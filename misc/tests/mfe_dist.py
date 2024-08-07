import numpy as np
from tqdm import tqdm
import RNA
import argparse
import multiprocessing as mp
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", "-i", type=str, required=True, help="Input path")
    args.add_argument("--threads", "-t", type=int, default=120, help="Number of threads")
    args = args.parse_args()
    return args

def seq_to_mfe(seq):
    mfe = RNA.fold(seq)
    return mfe[1]


def main():
    args = parse_args()

    path_list = list(glob.glob(f"{args.input}/train/neg/*.pkl"))
    path_list = np.random.choice(path_list, 10)
    result_list = []
    for path in tqdm(path_list, desc="Calculating MFE"):
        data = pd.read_pickle(path)
        data["mfe"] = data["kmer_token"].apply(seq_to_mfe)
        result_list.append(data)
    neg_result = pd.concat(result_list)

    path_list = list(glob.glob(f"{args.input}/train/pos/*.pkl"))
    path_list = np.random.choice(path_list, 10)
    result_list = []
    for path in tqdm(path_list, desc="Calculating MFE"):
        data = pd.read_pickle(path)
        data["mfe"] = data["kmer_token"].apply(seq_to_mfe)
        result_list.append(data)
    pos_result = pd.concat(result_list)

    ## Plot histogram

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(neg_result["mfe"], ax=ax[0], kde=True, binwidth=0.25, binrange=(-20, 0), stat="density")
    ax[0].set_title("Negative")
    sns.histplot(pos_result["mfe"], ax=ax[1], kde=True, binwidth=0.25, binrange=(-20, 0), stat="density")
    ax[1].set_title("Positive")

    for ax in ax:
        ax.set_xlabel("MFE")
        ax.set_ylabel("Count")
        ax.set_xlim(-20, 0)
        ax.set_ylim(0, 2.0)

    plt.tight_layout()
    plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/mfe_dist.png")


    return None


if __name__ == "__main__":
    main()