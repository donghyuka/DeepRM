import gc
import argparse
import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
from utils.utils import parse_refflat


def transcript_to_chromosomal_coordinate(coord,chrom,exon_starts,exon_ends,strand,exon_cumsum):

    try:

        if strand == "+":
            exon_index = np.searchsorted(exon_cumsum, coord, side="right") - 1
            chrom_coord = exon_starts[exon_index] + (coord - exon_cumsum[exon_index])
        elif strand == "-":
            exon_index = np.searchsorted(exon_cumsum, coord, side="right") - 1
            chrom_coord = exon_ends[-exon_index-1] - (coord - exon_cumsum[exon_index]) - 1
        else:
            raise ValueError("Invalid strand")

        genome_id = f"{chrom}:{strand}:{chrom_coord}"

    except IndexError:
        ## This is Poly(A) tail
        return None

    return genome_id


def process_dorado_inferece(data_path):
    if data_path is None:
        return None
    elif data_path.endswith(".bed"):
        data_df = pd.read_csv(data_path, quoting = 3, sep = "\t", header = None, dtype=str)
        data_df = data_df[[0, 1, 3, 4, 9]]
        data_df.columns = ["transcript_id", "transcript_pos", "modbase", "depth", "dorado_dom"]
        data_df["depth"] = data_df["depth"].astype(int)
        data_df = data_df[data_df["depth"] >= 5].copy()
        data_df["dorado_dom"] = data_df["dorado_dom"].str.split(" ").str[1].astype(float) / 100
        data_df = data_df[data_df["dorado_dom"] > 0.0].copy()
        data_df["modbase"] = data_df["modbase"].map({"a":"m6A","17802":"pseU"})
        data_df["transcript_id"] = data_df["transcript_id"].str.split(".").str[0]
        data_df["transcript_pos"] = data_df["transcript_pos"].astype(int)
        data_df.to_pickle(data_path.replace(".bed", ".pkl"))
    elif data_path.endswith(".pkl"):
        data_df = pd.read_pickle(data_path)
    else:
        raise ValueError("Invalid data_path")

    return data_df


def worker(data_df_list, refflat_df, return_list):
    local_collect = []

    for data_df in tqdm.tqdm(data_df_list, desc="Processing depth_df"):
        nmid = data_df["transcript_id"].values[0]
        try:
            refflat_row = refflat_df.loc[nmid]
        except KeyError:
            continue
        if len(refflat_row) == 0:
            continue
        if type(refflat_row) == pd.DataFrame:
            refflat_row = refflat_row.iloc[0]

        exon_starts = refflat_row["exonStarts"]
        exon_ends = refflat_row["exonEnds"]
        strand = refflat_row["strand"]
        chrom = refflat_row["chr"]

        if strand == "+":
            exon_cumsum=np.concatenate(([0],np.cumsum(exon_ends-exon_starts)))

        elif strand == "-":
            exon_cumsum=np.concatenate(([0],np.cumsum(np.flip(exon_ends-exon_starts, axis=0))))

        data_df["genome_id"] = data_df.apply(lambda row: transcript_to_chromosomal_coordinate(row["transcript_pos"], chrom,exon_starts,exon_ends,strand,exon_cumsum), axis=1)
        data_df["transcriptome_id"] = data_df["transcript_id"] + ":" + data_df["transcript_pos"].astype(str)
        data_df = data_df[["genome_id", "transcriptome_id", "modbase", "depth", "dorado_dom"]].copy()

        data_df = data_df.dropna()
        local_collect.append(data_df)

    if len(local_collect) == 0:
        return None

    data_df = pd.concat(local_collect)
    return_list.append(data_df)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, required = True, nargs="+")
    parser.add_argument("--refflat", type = str, default="/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.gtf.refflat.txt")
    parser.add_argument("--cpu", type = int, default = 120)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    cpu = args.cpu

    for data_path in args.input:
        data_df = process_dorado_inferece(data_path)
        data_df_groupby = data_df.groupby("transcript_id")

        refflat_df = parse_refflat(args.refflat, drop_y=False, drop_m=True, drop_unk=True, drop_ver=True, reindex=True)

        ## split into ncpu chunks, respecting the nmid groupby
        data_df_split = [[] for i in range(cpu)]

        ## sort groups according to length
        groups = [group for name, group in data_df_groupby]
        len_groups =len(groups)
        groups = sorted(groups, key = lambda x: len(x), reverse = True)

        for idx, group in tqdm.tqdm(enumerate(groups), desc="Splitting depth_df", total=len_groups):
            nmid = group["transcript_id"].values[0]
            split_idx = idx % (2*cpu)
            if split_idx >= cpu:
                split_idx = 2*cpu - split_idx - 1
            data_df_group = data_df_groupby.get_group(nmid)
            data_df_split[split_idx].append(data_df_group)

        del data_df, data_df_groupby

        proc_list = []
        man = mp.Manager()
        return_list = man.list()

        for data_df_split_chunk in data_df_split:
            if len(data_df_split_chunk) == 0:
                continue
            proc = mp.Process(target=worker, args=(data_df_split_chunk, refflat_df, return_list))
            proc.start()
            proc_list.append(proc)

        del data_df_split
        gc.collect()

        for proc in proc_list:
            proc.join()

        return_list = list(return_list)
        datid_df = pd.concat(return_list).reset_index(drop=True)
        datid_df = datid_df.dropna()
        man.shutdown()
        del return_list
        gc.collect()

        datid_df.to_pickle(data_path[:-4] + ".genomic.pkl")
    return None


if __name__ == "__main__":
    main()