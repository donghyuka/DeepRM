import argparse
import gc

import pandas as pd
import numpy as np
import multiprocessing as mp
import os

from sympy.external.tests.test_numpy import array
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
    gene_df = gene_df.groupby(["genome_id","gene_id","5mer"]).agg({
                                                "genome_id": "first",
                                                "gene_id" : "first",
                                                "5mer": "first",
                                                "depth": "sum",
                                                "coding": "max",
                                                "drach": "max",
                                                "isoforms": "sum",
                                                })
    gene_df = gene_df.reset_index(drop=True)
    collect_list.append(gene_df.copy())

    return None

def load_split_data(data_path, cpu):
    if data_path.endswith(".pkl"):
        data_df = pd.read_pickle(data_path)
    elif data_path.endswith(".npz"):
        with np.load(data_path, allow_pickle=True) as data:
            data_df = pd.DataFrame({k: data[k] for k in data.keys()})
    else:
        raise ValueError("Invalid data format: must be .pkl or .npz")

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
    df_list_split = load_split_data(args.input, args.cpu)

    refflat_df = parse_refflat_v2(args.refflat)
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

    print(gene_df)

    gc.collect()

    gene_df = gene_df.reset_index(drop=True)

    gene_df.to_pickle(args.output)

    # gene_df = pd.read_pickle(args.output)

    print(gene_df)

    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first",''
                                                "depth": "sum",
                                                "5mer": "unique",
                                                "gene_id": "unique",
                                                "coding": "max",
                                                "drach": "max",
                                                "isoforms": "sum",
                                              })

    gene_df.to_pickle(args.output)

    return None



if __name__ == "__main__":
    main()