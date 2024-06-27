
import pandas as pd
import numpy as np
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from collections import defaultdict
from utils.utils import mean_phred, seq_to_onehot
import pysam
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pickle


def main(penalty_cutoff = 10, bq_cutoff=7, max_penalty = 15):
    prefix = "/extdata4/baeklab/Hyeonseo/m6A/plot/dataset3/"
    os.makedirs(prefix, exist_ok=True)
    # neg_bam_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/dorado_output.bam"
    pos_bam_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado/intermediates/dorado_output.bam"
    pos_cnt, pos_pass_cnt = count_bam(pos_bam_path, bq_cutoff)
    # neg_cnt, neg_pass_cnt = count_bam(neg_bam_path, bq_cutoff)
    print(f"positive reads passed: {pos_pass_cnt:,} / {pos_cnt:,}")


    # print(f"negative reads passed: {neg_pass_cnt:,} / {neg_cnt:,}")
    # pos_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/block_df.pkl"
    # neg_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado/intermediates/block_df.pkl"
    # pos_df = pd.read_pickle(pos_df_path)
    # neg_df = pd.read_pickle(neg_df_path)
    # pos_df = pos_df.sample(frac=0.01)
    # neg_df = neg_df.sample(frac=0.01)
    # gc.collect()
    # print(f"positive before filtering: {len(pos_df):,}")
    # print(f"negative before filtering: {len(neg_df):,}")
    # pos_df["block_score"] = pos_df["penalty"].apply(lambda x: 1-(x/max_penalty))
    # neg_df["block_score"] = neg_df["penalty"].apply(lambda x: 1-(x/max_penalty))
    # block_score_distribution(pos_df, neg_df, prefix)
    # pos_df = pos_df[pos_df["penalty"] <= penalty_cutoff]
    # neg_df = neg_df[neg_df["penalty"] <= penalty_cutoff]
    # print(f"positive after filtering: {len(pos_df):,}")
    # print(f"negative after filtering: {len(neg_df):,}")
    # pos_df.to_pickle(f"{prefix}/pos_df_samp.pkl")
    # neg_df.to_pickle(f"{prefix}/neg_df_samp.pkl")
    # pos_df = pd.read_pickle(f"{prefix}/pos_df_samp.pkl")
    # neg_df = pd.read_pickle(f"{prefix}/neg_df_samp.pkl")
    # print(pos_df)
    # print(neg_df)
    # os.makedirs(prefix, exist_ok=True)
    # motif_cdf(pos_df, neg_df, prefix)
    # motif_composition(pos_df, neg_df, prefix)
    # bq_plot(pos_df, neg_df, prefix)
    return None


def count_bam(bam_path, bq_cutoff):
    all_cnt = 0
    pass_cnt = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=100) as input_bam:
        for record in tqdm(input_bam):
            qscore = mean_phred(np.array(record.query_qualities, dtype=int))
            if qscore >= bq_cutoff :
                pass_cnt += 1
            all_cnt += 1
    return all_cnt, pass_cnt


def motif_cdf(pos_df,neg_df, prefix):
    pos_motif = pos_df["motif"].apply(lambda x: x[8:13]).value_counts()
    neg_motif = neg_df["motif"].apply(lambda x: x[8:13]).value_counts()
    pos_motif_cnt_arr = np.array([pos_motif.get(motif,0) for motif in pos_motif.index])
    neg_motif_cnt_arr = np.array([neg_motif.get(motif,0) for motif in pos_motif.index])
    pos_motif_cnt_sorted = np.sort(pos_motif_cnt_arr)[::-1]
    neg_motif_cnt_sorted = np.sort(neg_motif_cnt_arr)[::-1]
    pos_motif_cdf = np.cumsum(pos_motif_cnt_sorted)/np.sum(pos_motif_cnt_sorted)
    neg_motif_cdf = np.cumsum(neg_motif_cnt_sorted)/np.sum(neg_motif_cnt_sorted)

    with open(f"{prefix}/motif_cdf.pickle", "wb") as f:
        pickle.dump([pos_motif_cdf, neg_motif_cdf], f)


    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.plot(pos_motif_cdf, label=f"m6A(n={len(pos_df)})", color="royalblue")
    ax.plot(neg_motif_cdf, label= f"cA(n={len(neg_df)})", color="tomato")
    ax.set_title("5-mer Motif CDF")
    ax.set_xlabel("Motif")
    ax.set_ylabel("CDF")
    ax.legend()
    plt.savefig(f"{prefix}/motif_cdf.png", dpi=300)
    return None


def motif_composition(pos_df, neg_df, prefix):
    ## Plot ratio of nucleotides in each position
    ## Each nucleotide is represented as a box, and the height of the box is the ratio of the nucleotide
    pos_motif = pos_df["motif"].apply(lambda x: seq_to_onehot(x))
    neg_motif = neg_df["motif"].apply(lambda x: seq_to_onehot(x))
    pos_motif_sum = np.sum(pos_motif.to_numpy(), axis=0)
    neg_motif_sum = np.sum(neg_motif.to_numpy(), axis=0)
    pos_motif_sum = pos_motif_sum/np.sum(pos_motif_sum)
    neg_motif_sum = neg_motif_sum/np.sum(neg_motif_sum)
    pos_motif_sum = pd.DataFrame(pos_motif_sum, columns=["A","C","G","U"])
    neg_motif_sum = pd.DataFrame(neg_motif_sum, columns=["A","C","G","U"])

    with open(f"{prefix}/motif_composition.pickle", "wb") as f:
        pickle.dump([pos_motif_sum, neg_motif_sum], f)

    fig, axes = plt.subplots(2, 1, figsize=(20,20))
    for ax, composition, title in zip(axes, [pos_motif_sum, neg_motif_sum],
                                      [f"motif composition m6A (n={len(pos_df)})", f"motif composition cA (n={len(neg_df)})"]):
        ## Stacked bar plot for each base position
        composition.plot(kind="bar", stacked=True, ax=ax, color=["royalblue", "tomato", "forestgreen", "gold"])
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Ratio")
        ax.legend()
    plt.savefig(f"{prefix}/motif_composition.png", dpi=300)

    return None


def bq_plot(pos_df, neg_df, prefix):
    ## Plot the distribution of base quality. Plot position-wise mean with CI95.

    pos_bq_arr = np.stack(pos_df["bq"].values, axis=0)
    neg_bq_arr = np.stack(neg_df["bq"].values, axis=0)
    print(pos_bq_arr)
    print(neg_bq_arr)
    pos_bq_mean = np.mean(pos_bq_arr, axis=0)
    neg_bq_mean = np.mean(neg_bq_arr, axis=0)
    pos_bq_std = np.std(pos_bq_arr, axis=0)
    neg_bq_std = np.std(neg_bq_arr, axis=0)
    pos_bq_ci95 = 1.96*pos_bq_std/np.sqrt(len(pos_bq_arr))
    neg_bq_ci95 = 1.96*neg_bq_std/np.sqrt(len(neg_bq_arr))

    with open(f"{prefix}/bq_plot.pickle", "wb") as f:
        pickle.dump([pos_bq_mean, neg_bq_mean, pos_bq_ci95, neg_bq_ci95], f)

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(pos_bq_mean, label=f"m6A (n={len(pos_df)})", color="royalblue")
    ax.fill_between(np.arange(len(pos_bq_mean)), pos_bq_mean-pos_bq_ci95, pos_bq_mean+pos_bq_ci95, color="royalblue", alpha=0.3)
    ax.plot(neg_bq_mean, label=f"cA (n={len(neg_df)})", color="tomato")
    ax.fill_between(np.arange(len(neg_bq_mean)), neg_bq_mean-neg_bq_ci95, neg_bq_mean+neg_bq_ci95, color="tomato", alpha=0.3)
    ax.set_title("Base Quality Distribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Base Quality")
    ax.legend()
    plt.savefig(f"{prefix}/bq_plot.png", dpi=300)
    return None


def length_distribution(pos_meta_df, neg_meta_df, prefix):
    ## Plot the distribution of read length
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    sns.histplot(pos_meta_df["token_len"], ax=ax, kde=True, stat="density", bins=100, color="royalblue", label=f"m6A (n={len(pos_meta_df)}")
    sns.histplot(neg_meta_df["token_len"], ax=ax, kde=True, stat="density", bins=100, color="tomato", label=f"cA (n={len(neg_meta_df)}")
    ax.set_title("Token Length Distribution")
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"{prefix}/length_distribution.png", dpi=300)
    return None


def block_score_distribution(pos_meta_df, neg_meta_df, prefix):
    ## Plot the distribution of block score
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    sample_pos = pos_meta_df.sample(frac=0.01)
    sample_neg = neg_meta_df.sample(frac=0.01)
    sns.kdeplot(sample_pos["block_score"], ax=ax, color="royalblue", label=f"m6A (n={len(pos_meta_df)}", bw_adjust=4)
    sns.kdeplot(sample_neg["block_score"], ax=ax, color="tomato", label=f"cA (n={len(neg_meta_df)}", bw_adjust=4)
    ax.set_title("Block Score Distribution")
    ax.set_xlabel("Block Score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"{prefix}/block_score_distribution.png", dpi=300)
    return None


def motif_cdf_load( prefix):

    with open(f"{prefix}/motif_cdf.pickle", "rb") as f:
        pos_motif_cdf, neg_motif_cdf = pickle.load(f)

    pos_motif_cnt = []
    neg_motif_cnt = []

    prev_pos = 0
    prev_neg = 0
    for motif in pos_motif_cdf:
        pos_motif_cnt.append(motif-prev_pos)
        prev_pos = motif
    for motif in neg_motif_cdf:
        neg_motif_cnt.append(motif-prev_neg)
        prev_neg = motif
    pos_motif_cnt = np.array(pos_motif_cnt)*256
    neg_motif_cnt = np.array(neg_motif_cnt)*256
    ## plot motif ratio distribution
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    sns.kdeplot(pos_motif_cnt, ax=ax, color="royalblue", label=f"m6A")
    sns.kdeplot(neg_motif_cnt, ax=ax, color="tomato", label=f"cA")
    ax.set_title("5-mer Motif Proportion Distribution")
    ax.set_xlabel("Motif Proportion (1 = expected proportion)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"{prefix}/motif_cdf_prop.png", dpi=300)

    return None


if __name__ == "__main__":
    main()