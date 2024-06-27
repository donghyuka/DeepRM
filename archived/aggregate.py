import argparse
import gc

import pandas as pd
import numpy as np
import multiprocessing
import os
import glob
from tqdm import tqdm

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--data", type=str, required=True, help="Data path")
    args.add_argument("--output", type=str, required=True, help="Output path")
    args.add_argument("--dom_pos", type=int, default = 0.7, help="DOM Positive Threshold")
    args.add_argument("--dom_neg", type=int, default = 0.3, help="DOM Negative Threshold")
    args.add_argument("--p_pos", type=int, default = 0.98, help="pm6A Positive Threshold")
    args.add_argument("--p_neg", type=int, default = 0.10, help="pm6A Negative Threshold")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def worker(pid, save_path, df, dom_pos, dom_neg, p_pos, p_neg, epsilon=1e-6, min_depth = 5):
    ## Since first and last groups can be truncated, collect them separately.

    df["count"] = 1
    df_p = df[(df["pred"] >= p_pos) | (df["pred"] <= p_neg)].copy()
    df_p["pred_log10"] = np.log10(1 - np.clip(df_p["pred"].to_numpy(), 0.0, 1 - epsilon))
    df_d = df[(df["pred"] >= dom_pos) | (df["pred"] <= dom_neg)].copy()
    df_d["pred_bin"] = (df_d["pred"] >= 0.5)

    df = df.groupby("label_id")
    df_p = df_p.groupby("label_id")
    df_d = df_d.groupby("label_id")

    id_list = []
    p_mean_list = []
    d_mean_list = []
    depth_list = []
    p_depth_list = []
    d_depth_list = []

    for label_id, df_group in tqdm(df):

        try:
            df_p_group = df_p.get_group(label_id)
        except KeyError:
            continue

        try:
            df_d_group = df_d.get_group(label_id)
        except KeyError:
            continue

        if min(len(df_p_group), len(df_d_group)) < min_depth:
            continue

        p_mean = df_p_group["pred_log10"].mean()
        d_mean = df_d_group["pred_bin"].mean()
        depth = len(df_group)
        p_depth = len(df_p_group)
        d_depth = len(df_d_group)

        id_list.append(label_id)
        p_mean_list.append(p_mean)
        d_mean_list.append(d_mean)
        depth_list.append(depth)
        p_depth_list.append(p_depth)
        d_depth_list.append(d_depth)

    p_mean_list = 1 - 10**np.clip(np.array(p_mean_list), None, 0)

    df = pd.DataFrame({"label_id":id_list, "pred_pm6a":p_mean_list, "pred_dom":d_mean_list, "depth":depth_list, "depth_pm6a":p_depth_list, "depth_dom":d_depth_list})
    df_dom_mask = df["pred_pm6a"] >= 0.5
    df["pred_dom"] = df["pred_dom"] * df_dom_mask
    df.to_pickle(f"{save_path}/df_{pid}.pkl")

    gc.collect()
    return None

def load_split_data(data_path, cpu):
    df_list = []
    for file in tqdm(glob.glob(f"{data_path}/*.tsv")):
        df = pd.read_csv(file, sep="\t")
        df_list.append(df)
    df_list = pd.concat(df_list)
    df_list["gene"] = df_list["label_id"].apply(lambda x: x.split(":")[0])
    df_list = df_list.groupby("gene")
    ## sort by size
    df_list = sorted(df_list, key=lambda x: len(x[1]), reverse=True)

    df_list_split = [[] for _ in range(cpu)]
    for idx, (gene, gene_df) in tqdm(enumerate(df_list), total=len(df_list)):
        split_idx = idx % (2*cpu)
        if split_idx >= cpu:
            split_idx = 2*cpu - split_idx - 1
        df_list_split[split_idx].append(gene_df)

    df_list_split = [pd.concat(x, axis=0) for x in df_list_split]

    del df_list
    gc.collect()

    return df_list_split


def main():
    args = parse_args()
    df_list = load_split_data(args.data, args.cpu)
    gc.collect()
    proc_list = []
    for pid, df in enumerate(df_list):
        proc = multiprocessing.Process(target=worker, args=(pid, args.output, df.copy(), args.dom_pos, args.dom_neg, args.p_pos, args.p_neg))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    gc.collect()

    return None



if __name__ == "__main__":
    main()