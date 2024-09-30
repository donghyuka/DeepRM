import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
import glob
import os
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", "-c", type=int, default=int(mp.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--input", "-i", type=str, required=True, help="Input path")
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/histogram", help="Output path")
    args = args.parse_args()
    if args.input.endswith("/"):
        args.input = args.input[:-1]
    if args.output.endswith("/"):
        args.output = args.output[:-1]

    args.output = os.path.join(args.output, os.path.basename(args.input))
    os.makedirs(args.output, exist_ok=True)
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    file_list = glob.glob(f"{args.input}/*.pkl")
    ncpu = min(len(file_list), args.cpu)
    manager = mp.Manager()
    return_list = manager.list()
    proc_list = []
    split_file_list = np.array_split(file_list, ncpu)

    for i in range(ncpu):
        p = mp.Process(target=worker, args=(i, split_file_list[i], return_list))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    return_list = list(return_list)
    manager.shutdown()
    return_list = np.array(return_list)
    return_list = np.sum(return_list, axis=0)
    draw_histogram(return_list, args.output)
    np.save(f"{args.output}/histogram.npy", return_list)
    return None


def worker(pid, file_paths, return_list):
    df_list = []
    for path in tqdm.tqdm(file_paths):
        data_df =  pd.read_pickle(path)
        df_list.append(data_df["pred"].values)
    val_arr = np.concatenate(df_list)
    hist, bin_edges = np.histogram(val_arr, bins = 1000, range = (0, 1))
    return_list.append(hist)
    return None


def draw_histogram(hist_arr, out_path):
    bin_edges = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(len(hist_arr)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        ax.fill_between([bin_start, bin_end], hist_arr[i], color = "blue", alpha = 0.5)
    ax.set_title("Prediction Histogram")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Frequency")
    plt.savefig(f"{out_path}/histogram.png")
    return None


if __name__ == "__main__":
    main()


