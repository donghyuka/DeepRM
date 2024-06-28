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


def worker(pid, save_path, df, dom_pos, dom_neg, p_pos, p_neg, collect_list, epsilon=1e-6, min_depth = 5):
    ## Since first and last groups can be truncated, collect them separately.

    df["count"] = 1
    df["pred_log10"] = np.log10(1 - np.clip(df["pred"].to_numpy(), 0.0, 1 - epsilon))
    df["pred_bin"] = df["pred"] >= 0.5

    first_group_name = df["label_id"].iloc[0]
    last_group_name = df["label_id"].iloc[-1]

    df = df.groupby("label_id")
    first_group = df.get_group(first_group_name)
    last_group = df.get_group(last_group_name)

    id_list = []
    p_mean_list = []
    d_mean_list = []
    depth_list = []


    idx = 0
    for label_id, df_group in tqdm(df):
        if idx == 0 or idx == len(df) - 1:
            idx += 1
            continue
        idx += 1

        df_p = df_group[(df_group["pred"] >= p_pos) | (df_group["pred"] <= p_neg)].copy()
        df_d = df_group[(df_group["pred"] >= dom_pos) | (df_group["pred"] <= dom_neg)].copy()

        if min(len(df_p), len(df_d)) < min_depth:
            continue

        p_mean = df_p["pred_log10"].mean()
        d_mean = df_d["pred_bin"].mean()
        depth = len(df_group)

        id_list.append(label_id)
        p_mean_list.append(p_mean)
        d_mean_list.append(d_mean)
        depth_list.append(depth)

    p_mean_list = 1 - 10**np.clip(np.array(p_mean_list), None, 0)

    df = pd.DataFrame({"label_id":id_list, "pred_pm6a":p_mean_list, "pred_dom":d_mean_list, "depth":depth_list})
    df_dom_mask = df["pred_pm6a"] >= 0.5
    df["pred_dom"] = df["pred_dom"] * df_dom_mask
    df.to_pickle(f"{save_path}/df_{pid}.pkl")

    gc.collect()

    collect_list.append(first_group)
    collect_list.append(last_group)
    return None

def load_split_data(data_path, cpu):
    df_list = []
    for file in tqdm(glob.glob(f"{data_path}/*.tsv")):
        df = pd.read_csv(file, sep="\t")
        df_list.append(df)
    df_list = pd.concat(df_list)
    df_list = df_list.sort_values("label_id", ascending=False).reset_index(drop=True)
    df_list = np.array_split(df_list, cpu)
    return df_list


def final_worker(pid, save_path, df, dom_pos, dom_neg, p_pos, p_neg, min_depth = 5):
    df = pd.concat(df)
    df = df.groupby("label_id")

    id_list = []
    p_mean_list = []
    d_mean_list = []
    depth_list = []

    for label_id, df_group in tqdm(df):
        df_p = df_group[(df_group["pred"] >= p_pos) | (df_group["pred"] <= p_neg)].copy()
        df_d = df_group[(df_group["pred"] >= dom_pos) | (df_group["pred"] <= dom_neg)].copy()

        if min(len(df_p), len(df_d)) < min_depth:
            continue

        p_mean = df_p["pred_log10"].mean()
        d_mean = df_d["pred_bin"].mean()
        depth = len(df_group)

        id_list.append(label_id)
        p_mean_list.append(p_mean)
        d_mean_list.append(d_mean)
        depth_list.append(depth)

    p_mean_list = 1 - 10**np.clip(np.array(p_mean_list), None, 0)

    df = pd.DataFrame({"label_id":id_list, "pred_pm6a":p_mean_list, "pred_dom":d_mean_list, "depth":depth_list})
    df_dom_mask = df["pred_pm6a"] >= 0.5
    df["pred_dom"] = df["pred_dom"] * df_dom_mask
    df.to_pickle(f"{save_path}/df_{pid}.pkl")

    return None


def main():
    args = parse_args()
    df_list = load_split_data(args.data, args.cpu)
    gc.collect()
    man = multiprocessing.Manager()
    collect_list = man.list()
    proc_list = []
    for pid, df in enumerate(df_list):
        proc = multiprocessing.Process(target=worker, args=(pid, args.output, df.copy(), args.dom_pos, args.dom_neg, args.p_pos, args.p_neg, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()
    gc.collect()

    final_worker(args.cpu, args.output, collect_list, args.dom_pos, args.dom_neg, args.p_pos, args.p_neg)
    gc.collect()

    return None



if __name__ == "__main__":
    main()