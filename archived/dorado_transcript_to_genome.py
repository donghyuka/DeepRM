import gc

import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
from utils.utils import parse_refflat

## TODO: Refactor to remove these fixed paths.
DATA_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado_m6a/dorado_m6a_basecalled.pileup.bed"
REFFLAT_PATH = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.gtf.refflat.txt"

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
    data_df = pd.read_csv(data_path, quoting = 3, sep = "\t", header = None, dtype=str)
    ## Keep column 0, 1, 4, 9
    data_df = data_df[[0, 1, 9]]
    data_df.columns = ["nmid", "pos", "pred"]
    data_df["nmid"] = data_df["nmid"].str.split(".").str[0]
    data_df["depth_m6a"] = data_df["pred"].str.split(" ").str[2].astype(int)
    data_df["depth_ca"] = data_df["pred"].str.split(" ").str[3].astype(int)
    data_df["pos"] = data_df["pos"].astype(int)
    data_df.drop(columns = ["pred"], inplace = True)
    return data_df


def worker(data_df_list, refflat_df, return_list):
    local_collect = []

    for data_df in tqdm.tqdm(data_df_list, desc="Processing depth_df"):
        nmid = data_df["nmid"].values[0]
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

        data_df["genome_id"] = data_df.apply(lambda row: transcript_to_chromosomal_coordinate(row["pos"], chrom,exon_starts,exon_ends,strand,exon_cumsum), axis=1)
        data_df = data_df.dropna()
        local_collect.append(data_df)

    if len(local_collect) == 0:
        return None

    data_df = pd.concat(local_collect)
    return_list.append(data_df)
    return None


def main(cpu = 120):
    data_df = process_dorado_inferece(DATA_PATH)
    data_df_groupby = data_df.groupby("nmid")

    refflat_df = parse_refflat(REFFLAT_PATH, drop_y=True, drop_m=True)

    ## split into ncpu chunks, respecting the nmid groupby
    data_df_split = [[] for i in range(cpu)]

    ## sort groups according to length
    groups = [group for name, group in data_df_groupby]
    len_groups =len(groups)
    groups = sorted(groups, key = lambda x: len(x), reverse = True)

    for idx, group in tqdm.tqdm(enumerate(groups), desc="Splitting depth_df", total=len_groups):
        nmid = group["nmid"].values[0]
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
    datid_df = pd.concat(return_list)
    datid_df = datid_df.dropna()
    man.shutdown()
    del return_list
    gc.collect()

    datid_df.to_pickle(DATA_PATH.replace(".bed", ".genomic.pkl"))
    return None


if __name__ == "__main__":
    main()