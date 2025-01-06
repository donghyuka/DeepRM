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
    args.add_argument("--label", "-l", type=str, required=True, help="Label path")
    args.add_argument("--output", "-o", type=str, default = None, help="Output path")
    args = args.parse_args()
    if args.output is None:
        args.output = f"{args.input}.geneid.pkl"
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    return args


def worker(df, collect_list):
    df["isoforms"] = 1
    gene_df = df.groupby(["genome_id","gene_id"]).agg({"genome_id": "first", "gene_id": "first",
                                                "kl_div_neg": "sum", "kl_div_pos": "sum",
                                                "count_all": "sum",
                                                "logsum_1_p_pos": "sum", "count_pos": "sum",
                                                "isoforms": "sum", "coding": "max", "drach": "max"})
    gene_df = gene_df.reset_index(drop=True)
    collect_list.append(gene_df.copy())

    return None

def load_split_data(data_path, cpu, label_path):
    if data_path.endswith(".pkl"):
        data_df = pd.read_pickle(data_path)
    elif data_path.endswith(".npz"):
        with np.load(data_path, allow_pickle=True) as data:
            data_df = pd.DataFrame({k: data[k] for k in data.keys()})
    else:
        raise ValueError("Invalid data format: must be .pkl or .npz")

    label_df = pd.read_pickle(label_path)[["label_id", "gene_id", "genome_id", "coding", "drach"]]
    data_df = data_df[["label_id", "pm6a", "dom", "count_all", "count_pos", "kl_div_neg", "kl_div_pos","logsum_1_p_pos"]]
    print(data_df)
    data_df = data_df.merge(label_df, on="label_id", how="inner")
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
    df_list_split = load_split_data(args.input, args.cpu, args.label)

    for pid, df in enumerate(df_list_split):
        proc = mp.Process(target=worker, args=(df, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()

    gene_df = pd.concat(collect_list)

    gc.collect()

    gene_df = gene_df.groupby(["genome_id","gene_id"]).agg({"genome_id": "first", "gene_id": "first",
                                                "kl_div_neg": "sum", "kl_div_pos": "sum",
                                                "count_all": "sum",
                                                "logsum_1_p_pos": "sum", "count_pos": "sum",
                                                "isoforms": "sum", "coding": "max", "drach": "max"})

    gene_df["dom"] = gene_df["kl_div_pos"] / (gene_df["kl_div_neg"] + gene_df["kl_div_pos"])
    gene_df["pm6a"] = -(2-gene_df["dom"])*gene_df["logsum_1_p_pos"]/gene_df["count_all"] + ((1-gene_df["dom"])*np.log10(np.clip(1-gene_df["dom"],1e-30,1)) + gene_df["dom"] * np.log10(np.clip(gene_df["dom"],1e-30,1)))*(gene_df["count_pos"]/gene_df["count_all"])
    keys = ["genome_id", "gene_id", "coding", "drach", "isoforms", "pm6a", "dom", "count_all"]
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