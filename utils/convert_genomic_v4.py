import argparse
import gc

import pandas as pd
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
from utils.convert_dorado_genomic import transcript_to_chromosomal_coordinate
from utils.utils import parse_refflat_v2, reformat_transcript_id

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", "-c", type=int, default=int(os.cpu_count()*0.95), help="Number of CPUs")
    args.add_argument("--input", "-i", type=str, required=True, help="Data path")
    args.add_argument("--output", "-o", type=str, default = None, help="Output path")
    args.add_argument("--refflat", "-r", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/anno/agat_refflat.base0.pkl", help="Refflat pickle path")
    args = args.parse_args()
    if args.output is None:
        args.output = f"{args.input}.genomic.pkl"
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    return args


def worker(df, refflat_df, collect_list):

    df["pos"] = df["label_id"].apply(lambda x: x.split(":")[1]).astype(int)
    df["isoforms"] = 1

    df = df.groupby("transcript_id")

    local_collect = []
    for nmid, gene_df in tqdm(df, total=len(df)):
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

        gene_df["genome_id"] = gene_df.apply(lambda row: transcript_to_chromosomal_coordinate(row["pos"], chrom,exon_starts,exon_ends,strand,exon_cumsum), axis=1)
        gene_df = gene_df.dropna()
        local_collect.append(gene_df)

    gene_df = pd.concat(local_collect)
    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first",
                                                "kl_div_neg": "sum", "kl_div_pos": "sum",
                                                "count_all": "sum",
                                                "logsum_1_p_pos": "sum", "count_pos": "sum",
                                                "gene_id": "first", "isoforms": "sum", "coding": "max"})
    gene_df = gene_df.reset_index(drop=True)
    collect_list.append(gene_df.copy())

    return None

def load_split_data(data_path, cpu, refflat_df):
    if data_path.endswith(".pkl"):
        data_df = pd.read_pickle(data_path)
    elif data_path.endswith(".npz"):
        with np.load(data_path, allow_pickle=True) as data:
            data_df = pd.DataFrame({k: data[k] for k in data.keys()})
    else:
        raise ValueError("Invalid data format: must be .pkl or .npz")

    print(data_df)
    data_df["transcript_id"] = data_df["label_id"].apply(lambda x: reformat_transcript_id(x.split(":")[0]))

    geneid_table = refflat_df[["gene_id", "transcript_id", "coding"]]
    data_df = data_df.merge(geneid_table, how="left", on="transcript_id")
    data_df.dropna(inplace=True)

    print(data_df)

    data_df = data_df.groupby("gene_id")
    ## sort by size
    data_df = sorted(data_df, key=lambda x: len(x[1]), reverse=True)

    df_list_split = [[] for _ in range(cpu)]
    for idx, (gene, gene_df) in tqdm(enumerate(data_df), total=len(data_df)):
        split_idx = idx % (2*cpu)
        if split_idx >= cpu:
            split_idx = 2*cpu - split_idx - 1
        df_list_split[split_idx].append(gene_df)

    df_list_split = [pd.concat(x, axis=0) for x in df_list_split]

    gc.collect()


    return df_list_split


def main():
    args = parse_args()
    man = mp.Manager()
    collect_list = man.list()
    proc_list = []
    refflat_df = parse_refflat_v2(args.refflat)
    df_list_split = load_split_data(args.input, args.cpu, refflat_df)
    refflat_df.set_index("transcript_id",inplace=True)

    for pid, df in enumerate(df_list_split):
        proc = mp.Process(target=worker, args=(df, refflat_df, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()

    gene_df = pd.concat(collect_list)

    gc.collect()

    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first",
                                                "kl_div_neg": "sum", "kl_div_pos": "sum",
                                                "count_all": "sum",
                                                "logsum_1_p_pos": "sum", "count_pos": "sum",
                                                "gene_id": "first", "isoforms": "sum", "coding": "max"})

    gene_df["dom"] = gene_df["kl_div_pos"] / (gene_df["kl_div_neg"] + gene_df["kl_div_pos"])
    gene_df["pm6a"] = -(2-gene_df["dom"])*gene_df["logsum_1_p_pos"]/gene_df["count_all"] + ((1-gene_df["dom"])*np.log10(np.clip(1-gene_df["dom"],1e-30,1)) + gene_df["dom"] * np.log10(np.clip(gene_df["dom"],1e-30,1)))*(gene_df["count_pos"]/gene_df["count_all"])
    gene_df.rename({"gene_id":"gene_symbol"}, axis=1, inplace=True)

    keys = ["genome_id", "gene_symbol", "coding", "isoforms", "pm6a", "dom", "count_all", "logsum_1_p_pos", "kl_div_pos", "kl_div_neg"]
    gene_df = gene_df[keys]

    print(gene_df)
    if args.output.endswith(".pkl"):
        gene_df.to_pickle(args.output)
    elif args.output.endswith(".npz"):
        np.savez_compressed(args.output, **{key: gene_df[key].values for key in gene_df.columns})
    else:
        raise ValueError("Invalid data format: must be .pkl or .npz")
    gc.collect()

    return None



if __name__ == "__main__":
    main()