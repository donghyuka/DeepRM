import argparse
import gc
import glob
import multiprocessing as mp
import os
import pickle
import sys
import numpy as np
import pandas as pd
import pod5
import pysam
import atexit
import tqdm
from utils.utils import oom_killer, printmessage
import toml

## Warning: This script has heavy parallel I/O operations and large memory usage.
## > 1TB Read / Write operations and > 100 GB RAM usage is expected (> 4GB/s disk write was observed).
## Running this on NFS may cause significant performance degradation.




def segment_normalize_signal(seg_df_path, signal_path_arr):



    for signal_path in tqdm.tqdm(signal_path_arr, total=len(signal_path_arr), desc="Segmenting and Tokenizing Signals"):
        oom_killer()
        file_id = signal_path.split('/')[-1]

        if not os.path.exists(f"{seg_df_path}/intermediates/move_df_split/{file_id}"):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/block_df_split/{file_id}"):
            continue

        out_path = f"{seg_df_path}/distance_analysis/temp/{file_id}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}")
        signal_df = signal_df[["read_id", "mv"]]
        signal_df["read_len"] = signal_df["mv"].apply(lambda x: np.sum(x[1:])+1)
        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        block_df = block_df[block_df["penalty"] == 0]

        if len(block_df) == 0:
            continue

        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()


        signal_df.rename(columns={"pos_RM": "5p_dist"}, inplace=True)
        signal_df["3p_dist"] = signal_df["read_len"] - signal_df["5p_dist"] - 1

        signal_df = signal_df[["5p_dist", "3p_dist"]]

        save_path = f"{out_path.replace('.pkl','')}.pkl"
        signal_df.to_pickle(save_path)

    return None


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--output", "-o", type=str, required=True, nargs="+", help="Output path")
    args = parser.parse_args()
    return args


def assign_block_id(block_df):
    index = 0
    read_id_prev = ""
    block_id = []
    for read_id in block_df["read_id"]:
        if read_id != read_id_prev:
            index = 0
        else:
            index += 1
        block_id.append(index)
        read_id_prev = read_id
    block_df["block_id"] = block_id
    return block_df



def split_block_df(args, signal_path_dict, signal_path_arr, intermediate_path):

    printmessage("Reading Block Dataframe. It may take a while.", msg_type="info")
    block_df = pd.read_pickle(args.block)
    block_df = assign_block_id(block_df)
    block_df["signal_path"] = block_df["read_id"].map(signal_path_dict)

    ## Groupby read_id and make dict
    block_df_groupby = block_df.groupby("signal_path")

    del block_df
    gc.collect()

    for signal_path, group_df in tqdm.tqdm(block_df_groupby, total = len(signal_path_arr), desc="Splitting Block Dataframe"):
        group_df.to_pickle(f"{intermediate_path}/block_df_split/{signal_path}")

    del block_df_groupby
    gc.collect()

    return None


def parse_toml(toml_path):
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59

    if toml_path is None:
        printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
        printmessage("Using default values for standardisation.", msg_type="warning")
        return norm_factor_default

    if not os.path.exists(toml_path):
        printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
        printmessage("Using default values for standardisation.", msg_type="warning")
        return norm_factor_default

    toml_dict = toml.load(toml_path)
    if "normalisation" not in toml_dict:
        printmessage("normalisation section not found in the TOML file. Check Dorado model version.", msg_type="error", error=ValueError)
        printmessage("Using default values for standardisation.", msg_type="warning")
        return norm_factor_default

    printmessage("Normalisation parameters found in TOML file.", msg_type="info")

    std_dict = toml_dict["normalisation"]
    norm_factor = {}
    norm_factor["quantile_a"] = std_dict.get("quantile_a")
    norm_factor["quantile_b"] = std_dict.get("quantile_b")
    norm_factor["shift_mult"] = std_dict.get("shift_multiplier")
    norm_factor["scale_mult"] = std_dict.get("scale_multiplier")

    ## sanitize
    for key in norm_factor.keys():
        if norm_factor[key] is None:
            printmessage(f"Key {key} not found in TOML file. Falling back to default value.", msg_type="warning")
            norm_factor[key] = norm_factor_default[key]

    return norm_factor


def main():
    args = parse_args()

    for output_path in args.output:
        token_output_path = f"{output_path}/distance_analysis/"
        temp_path = f"{token_output_path}/temp/"
        intermediate_path = f"{output_path}/intermediates/"
        signal_index_path = f"{intermediate_path}/signal_index.pkl"

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)

        with open(signal_index_path, "rb") as infile:
            index_dict = pickle.load(infile)
        signal_path_arr = list(index_dict.keys())
        del index_dict
        gc.collect()

        np.random.shuffle(signal_path_arr)
        signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

        proc_list = []
        for signal_paths in signal_path_arr_split:
            proc = mp.Process(target=segment_normalize_signal,
                              args=(output_path, signal_paths))
            proc_list.append(proc)
            proc.start()

        del signal_path_arr_split
        gc.collect()

        for proc in proc_list:
            proc.join()

        orig_df = read_path(token_output_path)
        orig_df.to_pickle(f"{token_output_path}/result.pkl")

        printmessage("Saved to: " +output_path, msg_type="success")

    return None


def read_path(path, ncpu = 120):
    path_list = glob.glob(f"{path}/temp/*.pkl")
    man = mp.Manager()
    return_list = man.list()
    path_list = np.array_split(path_list, ncpu)
    proc_list = []
    for path in path_list:
        proc = mp.Process(target = read_file, args = (path, return_list))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_list = list(return_list)
    df = pd.concat(return_list)
    del return_list
    gc.collect()
    return df

def read_file(path_list, return_list):
    df_list = []
    for path in tqdm.tqdm(path_list):
        df = pd.read_pickle(path)
        df_list.append(df)
    df_list = pd.concat(df_list)
    return_list.append(df_list)
    return None


if __name__ == "__main__":
    main()
