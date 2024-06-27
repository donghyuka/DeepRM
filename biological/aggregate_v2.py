import argparse
import gc

import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import glob
from tqdm import tqdm
from evaluate.dorado_transcript_to_genome import transcript_to_chromosomal_coordinate
from utils.utils import is_drach, parse_refflat

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--data", type=str, required=True, help="Data path")
    args.add_argument("--output", type=str, required=True, help="Output path")
    args.add_argument("--label", type=str, required=True, help="Label path")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def worker(pid, df, refflat_df, output_path, collect_list, epsilon = 1e-6):

    df["drach"] = df["5mer"].apply(is_drach)
    df["count_m6a"] = (df["depth_dom"] * df["pred_dom"]).astype(int)
    df["count_ca"] = df["depth_dom"] - df["count_m6a"]
    df["pred_pm6a"] = np.clip(df["pred_pm6a"], 0.0, 1.0 - epsilon)
    df["pred_pm6a"] = np.log10(1-df["pred_pm6a"]) * df["depth_pm6a"]
    df["pos"] = df["label_id"].apply(lambda x: x.split(":")[1]).astype(int)
    df["coding"] = df["gene"].apply(lambda x: x.startswith("NM"))

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
    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first", "drach": "max", "count_m6a": "sum", "count_ca": "sum", "pred_pm6a": "sum", "depth_pm6a": "sum", "gene_id": "first", "depth": "sum", "coding": "max"})
    gene_df = gene_df.reset_index(drop=True)
    collect_list.append(gene_df.copy())

    # gene_df["pred_pm6a"] = 1 - (gene_df["pred_pm6a"] / gene_df["depth_pm6a"])**10
    # gene_df["pred_dom"] = gene_df["count_m6a"] / (gene_df["count_m6a"] + gene_df["count_ca"])
    # gene_df = gene_df[["genome_id", "gene_id", "pred_dom", "pred_pm6a", "drach", "depth"]]
    # gene_df.rename({"gene_id":"gene_symbol"}, axis=1, inplace=True)
    #
    # gene_df.to_pickle(output_path + f"/gene_df_{pid}.pkl")

    return None

def load_split_data(data_path, output_path, label_path, cpu):
    data_df = pd.concat([pd.read_pickle(data_path) for data_path in glob.glob(data_path + "/*.pkl")])
    print(data_df)

    label_df = pd.read_pickle(label_path)

    label_df = label_df[["id","5mer"]]
    label_df.rename({"id":"label_id"}, axis=1, inplace=True)
    data_df = data_df.merge(label_df, how="left", on="label_id")
    data_df["gene"] = data_df["label_id"].apply(lambda x: x.split(":")[0])

    geneid_table = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.convert_table.pkl")
    geneid_table.rename({"transcript_id":"gene"}, axis=1, inplace=True)
    data_df = data_df.merge(geneid_table, how="left", on="gene")
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
    df_list_split = load_split_data(args.data, args.output, args.label, args.cpu)
    refflat_df = parse_refflat(drop_y=True, drop_m=True)
    for pid, df in enumerate(df_list_split):
        proc = mp.Process(target=worker, args=(pid, df, refflat_df, args.output, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()

    gene_df = pd.concat(collect_list)

    gc.collect()

    gene_df = gene_df.groupby("genome_id").agg({"genome_id": "first", "drach": "max", "count_m6a": "sum", "count_ca": "sum", "pred_pm6a": "sum", "depth_pm6a": "sum", "gene_id": "first", "depth": "sum", "coding": "max"})

    gene_df["pred_pm6a"] = 1 - 10**(gene_df["pred_pm6a"] / gene_df["depth_pm6a"])
    gene_df["pred_dom"] = gene_df["count_m6a"] / (gene_df["count_m6a"] + gene_df["count_ca"])
    gene_df["count_dom"] = gene_df["count_m6a"] + gene_df["count_ca"]
    gene_df = gene_df[["genome_id", "gene_id", "dom", "drach", "depth", "count_dom", "coding"]].copy()
    gene_df.rename({"gene_id":"gene_symbol"}, axis=1, inplace=True)

    print(gene_df)

    gene_df.to_pickle(args.output + "/gene_df_final.pkl")
    gene_df.to_csv(args.output + "/gene_df_final.tsv", sep="\t", index=False)
    gc.collect()

    return None



if __name__ == "__main__":
    main()