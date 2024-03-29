
import pysam

import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from utils.utils import mean_phred

def read_bam(path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/intermediates/dorado_output.aligned.sorted.bam"):
    pentamer_rl = []
    trimer_rl = []
    with pysam.AlignmentFile(path, "rb") as bam:
        with tqdm.tqdm(total=bam.mapped) as pbar:
            for read in bam:
                ref = read.reference_name
                if ref == "trimer":
                    trimer_rl.append(read.query_length)
                elif ref == "pentamer":
                    pentamer_rl.append(read.query_length)
                pbar.update(1)
    return pentamer_rl, trimer_rl


def plot_rl_dist(pentamer_rl, trimer_rl, path = "/extdata4/baeklab/Hyeonseo/m6A/plot/misc"):
    rl_cutoff = 1000
    pentamer_rl = [x for x in pentamer_rl if x < rl_cutoff]
    trimer_rl = [x for x in trimer_rl if x < rl_cutoff]
    pentamer_rl = np.array(pentamer_rl)
    trimer_rl = np.array(trimer_rl)
    pentamer_df = pd.DataFrame(pentamer_rl, columns=["Read Length"])
    trimer_df = pd.DataFrame(trimer_rl, columns=["Read Length"])
    pentamer_df["Reference"] = f"Pentamer (n={len(pentamer_rl):,})"
    trimer_df["Reference"] = f"Trimer (n={len(trimer_rl):,})"
    df = pd.concat([pentamer_df, trimer_df])
    plt.rcParams.update({'font.size': 36})
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.kdeplot(data=df, x="Read Length", hue="Reference", ax=ax, fill=True, common_norm=True, common_grid=True,
                palette = ["royalblue", "tomato"], linewidth=6)
    ## vline at 287 and 461
    ax.axvline(287, color="tomato", linestyle='--', linewidth=3)
    ax.axvline(461, color="royalblue", linestyle='--', linewidth=3)
    plt.savefig(f"{path}/read_length_distribution.png", dpi=300)
    return None


def get_block_count_per_read(df):
    block_count = df.groupby("read_id").size().reset_index(name="block_count")
    return block_count

def main2():
    path_trimer = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/trimer_block.pkl"
    path_pentamer = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/pentamer_block.pkl"
    trimer_df = pd.read_pickle(path_trimer)
    pentamer_df = pd.read_pickle(path_pentamer)

    trimer_block_count = get_block_count_per_read(trimer_df)
    pentamer_block_count = get_block_count_per_read(pentamer_df)

    trimer_df["mean_bq"] = trimer_df["bq"].apply(lambda x: mean_phred(np.array(x, dtype=int)))
    pentamer_df["mean_bq"] = pentamer_df["bq"].apply(lambda x: mean_phred(np.array(x, dtype=int)))

    trimer_df["block_score"] = trimer_df["penalty"].apply(lambda x: 1-(x/15))
    pentamer_df["block_score"] = pentamer_df["penalty"].apply(lambda x: 1-(x/15))

    trimer_df["type"] = "Trimer"
    pentamer_df["type"] = "Pentamer"

    joint_df = pd.concat([trimer_df, pentamer_df])
    print(joint_df)

    trimer_block_count["type"] = "Trimer"
    pentamer_block_count["type"] = "Pentamer"

    ## create zero rows for trimer_block_count
    trimer_zero_df = pd.DataFrame({"read_id": ["NA"]*(1000000-len(trimer_block_count)), "block_count": [0]*(1000000-len(trimer_block_count)), "type": "Trimer"})
    trimer_block_count = pd.concat([trimer_block_count, trimer_zero_df])

    ## create zero rows for pentamer_block_count
    pentamer_zero_df = pd.DataFrame({"read_id": ["NA"]*(1000000-len(pentamer_block_count)), "block_count": [0]*(1000000-len(pentamer_block_count)), "type": "Pentamer"})
    pentamer_block_count = pd.concat([pentamer_block_count, pentamer_zero_df])

    joint_block_count = pd.concat([trimer_block_count, pentamer_block_count])
    print(joint_block_count)

    joint_block_count.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/joint_block_count.pkl")
    joint_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/joint_df.pkl")

    plot_block_count(joint_block_count)
    plot_bq(joint_df)
    plot_block_score(joint_df)

    return None


def plot_block_count(joint_block_count, path = "/extdata4/baeklab/Hyeonseo/m6A/plot/misc"):
    plt.rcParams.update({'font.size': 36})
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.histplot(data=joint_block_count, x="block_count", hue="type", ax=ax, kde = False, discrete=True, binwidth=1,
                 palette = ["royalblue", "tomato"], linewidth=3,
                 element="step", fill=True)
    trimer_mean = joint_block_count[joint_block_count["type"] == "Trimer"]["block_count"].mean()
    pentamer_mean = joint_block_count[joint_block_count["type"] == "Pentamer"]["block_count"].mean()
    ax.axvline(trimer_mean, color="royalblue", linestyle='--', linewidth=3)
    ax.axvline(pentamer_mean, color="tomato", linestyle='--', linewidth=3)
    ax.set_xlim(0, 20)
    plt.savefig(f"{path}/block_count_distribution.png", dpi=300)

    return None


def plot_bq(joint_df, path = "/extdata4/baeklab/Hyeonseo/m6A/plot/misc"):
    plt.rcParams.update({'font.size': 36})
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.kdeplot(data=joint_df, x="mean_bq", hue="type", ax=ax, fill=True, common_norm=False, common_grid=True,
                palette = ["royalblue", "tomato"], linewidth=6)
    trimer_mean = joint_df[joint_df["type"] == "Trimer"]["mean_bq"].mean()
    pentamer_mean = joint_df[joint_df["type"] == "Pentamer"]["mean_bq"].mean()
    ax.axvline(trimer_mean, color="royalblue", linestyle='--', linewidth=3)
    ax.axvline(pentamer_mean, color="tomato", linestyle='--', linewidth=3)
    ax.set_xlim(0, 40)
    plt.savefig(f"{path}/bq_distribution.png", dpi=300)

    return None


def plot_block_score(joint_df, path = "/extdata4/baeklab/Hyeonseo/m6A/plot/misc"):
    plt.rcParams.update({'font.size': 36})
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.kdeplot(data=joint_df, x="block_score", hue="type", ax=ax, fill=True, common_norm=False, common_grid=True,
                palette = ["royalblue", "tomato"], linewidth=6, bw_adjust=8)
    trimer_mean = joint_df[joint_df["type"] == "Trimer"]["block_score"].mean()
    pentamer_mean = joint_df[joint_df["type"] == "Pentamer"]["block_score"].mean()
    ax.axvline(trimer_mean, color="royalblue", linestyle='--', linewidth=3)
    ax.axvline(pentamer_mean, color="tomato", linestyle='--', linewidth=3)
    ax.set_xlim(0, 1)
    plt.savefig(f"{path}/block_score_distribution.png", dpi=300)

    return None




def main():
    pentamer_rl, trimer_rl = read_bam()
    plot_rl_dist(pentamer_rl, trimer_rl)
    return None


if __name__ == '__main__':
    main2()
