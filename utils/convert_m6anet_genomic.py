import argparse
import gc

import pandas as pd
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
from archived.misc.dorado_transcript_to_genome import transcript_to_chromosomal_coordinate
from utils.utils import parse_refflat_v2, reformat_transcript_id

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--data", type=str, required=True, help="Data path")
    args.add_argument("--output", type=str, required=True, help="Output path")
    args = args.parse_args()
    return args


def process_m6anet_inferece(data_path):
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].apply(reformat_transcript_id)
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"id": "label_id", "probability_modified": "pm6a", "mod_ratio": "dom", "n_reads": "count_pm6a"}, inplace = True)
    data_df = data_df[["label_id", "pm6a", "dom", "count_pm6a"]].copy()
    return data_df


def worker(df, refflat_df, collect_list, epsilon = 1e-6):

    df["count_m6a"] = (df["count_pm6a"] * df["dom"]).astype(int)
    df["count_ca"] = df["count_pm6a"] - df["count_m6a"]
    df["pm6a"] = np.clip(df["pm6a"], 0.0, 1.0 - epsilon)
    df["pm6a"] = np.log10(1-df["pm6a"]) * df["count_pm6a"]
    df["pos"] = df["label_id"].apply(lambda x: x.split(":")[1]).astype(int)

    df = df.groupby("gene")

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
    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first", "count_m6a": "sum",
                                                "count_ca": "sum", "pm6a": "sum", "count_pm6a": "sum", "gene_id": "first",
                                                "coding": "max"})
    gene_df = gene_df.reset_index(drop=True)
    collect_list.append(gene_df.copy())

    return None

def load_split_data(data_path, output_path, cpu, refflat_df):
    data_df = process_m6anet_inferece(data_path)
    print(data_df)
    data_df["gene"] = data_df["label_id"].apply(lambda x: x.split(":")[0])
    geneid_table = refflat_df[["gene_id", "transcript_id", "coding"]]
    data_df = data_df.merge(geneid_table, how="left", left_on="gene", right_on="transcript_id")
    data_df.dropna(inplace=True)

    print(data_df)

    data_df.to_pickle(output_path + "/data_df.pkl")

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
    refflat_df = parse_refflat_v2("/extdata4/baeklab/Hyeonseo/m6A/anno/agat_refflat.base0.pkl")
    print(refflat_df)
    df_list_split = load_split_data(args.data, args.output, args.cpu, refflat_df)
    refflat_df.set_index("transcript_id", inplace=True)
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

    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first", "count_m6a": "sum", "count_ca": "sum",
                                                "pm6a": "sum", "count_pm6a": "sum", "gene_id": "first",
                                                "coding": "max"})

    gene_df["pm6a"] = 1 - 10**(gene_df["pm6a"] / gene_df["count_pm6a"])
    gene_df["dom"] = gene_df["count_m6a"] / (gene_df["count_m6a"] + gene_df["count_ca"])
    gene_df["count_dom"] = gene_df["count_m6a"] + gene_df["count_ca"]
    gene_df = gene_df[["genome_id", "gene_id", "dom", "pm6a", "count_dom", "count_pm6a", "coding"]].copy()
    gene_df.rename({"gene_id":"gene_symbol"}, axis=1, inplace=True)

    print(gene_df)

    gene_df.to_pickle(args.output + "/gene_df.pkl")
    gc.collect()

    return None



if __name__ == "__main__":
    main()