## Open a bam file
## Get the stats of read
## 1. Read length distribution
## 2. Quality score distribution

import argparse
import pickle
import os
import numpy as np
import pysam
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.utils import mean_phred, printmessage


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--in", "-i", dest="bam_path", type=str, required=True, help="Input bam file")
    args.add_argument("--out","-o", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--cpu","-c", dest="cpu", type=int, default=8, help="Number of CPUs")
    args.add_argument("--bq", "-q", dest="bq_thres", type=int, default=7, help="Base quality threshold")
    args.add_argument("--bb", "-b", dest="bb_length", type=int, default=87, help="BB length")
    args.add_argument("--mrna", "-m", action="store_true", help="mRNA mode")
    args = args.parse_args()
    return args

def plot_read_len_oligo(read_len_arr, mean_qual_arr, bq_thres, out_path, bb_length):

    read_len_arr_passed = read_len_arr[mean_qual_arr >= bq_thres]
    read_len_arr_failed = read_len_arr[mean_qual_arr < bq_thres]

    ## plot read length KDE
    fig, ax = plt.subplots(figsize=(10,10))
    read_len_max = np.percentile(read_len_arr, 99.9)
    binrange = (0, read_len_max)
    binwidth = 10

    if bb_length is not None:
        for i in (2,3,4,5, 6):
            ligate_length = bb_length * i
            ax.axvline(ligate_length, color="grey", linestyle="-", linewidth=2)

    sns.histplot(read_len_arr_passed, ax=ax, color="royalblue", label=f"Passed (n={len(read_len_arr_passed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step",  stat='density')
    sns.histplot(read_len_arr_failed, ax=ax, color="tomato", label=f"Failed (n={len(read_len_arr_failed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step", stat='density')

    ## Median
    ax.axvline(np.median(read_len_arr_passed), color="royalblue", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_passed), 0.9 * ax.get_ylim()[1], f"Passed median = {np.median(read_len_arr_passed):.0f}", color="black")
    ax.axvline(np.median(read_len_arr_failed), color="tomato", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_failed), 0.8 * ax.get_ylim()[1], f"Failed median = {np.median(read_len_arr_failed):.0f}", color="black")

    ax.set_title(f"Read Length Distribution (n={len(read_len_arr):,})")
    ax.set_xlabel("Read Length")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1000)
    ax.legend()

    ## Peak Detection


    ## Vline at median
    fig.savefig(f"{out_path}/read_len_hist.png", dpi=300)
    plt.close(fig)
    return None


def plot_read_len_mrna(read_len_arr, mean_qual_arr, bq_thres, out_path):

    read_len_arr_passed = read_len_arr[mean_qual_arr >= bq_thres]
    read_len_arr_failed = read_len_arr[mean_qual_arr < bq_thres]

    ## plot read length KDE
    fig, ax = plt.subplots(figsize=(10,10))
    read_len_max = np.percentile(read_len_arr, 99.9)
    binrange = (0, read_len_max)
    binwidth = 10

    sns.histplot(read_len_arr_passed, ax=ax, color="royalblue", label=f"Passed (n={len(read_len_arr_passed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step",  stat='density')
    sns.histplot(read_len_arr_failed, ax=ax, color="tomato", label=f"Failed (n={len(read_len_arr_failed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step", stat='density')

    ## Median
    ax.axvline(np.median(read_len_arr_passed), color="royalblue", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_passed), 0.9 * ax.get_ylim()[1], f"Passed median = {np.median(read_len_arr_passed):.0f}", color="black")
    ax.axvline(np.median(read_len_arr_failed), color="tomato", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_failed), 0.8 * ax.get_ylim()[1], f"Failed median = {np.median(read_len_arr_failed):.0f}", color="black")

    ax.set_title(f"Read Length Distribution (n={len(read_len_arr):,})")
    ax.set_xlabel("Read Length")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 3000)
    ax.legend()

    ## Vline at median
    fig.savefig(f"{out_path}/read_len_hist.png", dpi=300)
    plt.close(fig)
    return None

def plot_polya_len(read_len_arr, mean_qual_arr, bq_thres, out_path):

    read_len_arr_passed = read_len_arr[mean_qual_arr >= bq_thres]
    read_len_arr_failed = read_len_arr[mean_qual_arr < bq_thres]

    ## plot read length KDE
    fig, ax = plt.subplots(figsize=(10,10))
    read_len_max = np.percentile(read_len_arr, 99.9)
    binrange = (0, read_len_max)
    binwidth = 10

    sns.histplot(read_len_arr_passed, ax=ax, color="royalblue", label=f"Passed (n={len(read_len_arr_passed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step",  stat='density')
    sns.histplot(read_len_arr_failed, ax=ax, color="tomato", label=f"Failed (n={len(read_len_arr_failed):,})", binwidth=binwidth, binrange=binrange, fill=False, lw=5, element="step", stat='density')

    ## Median
    ax.axvline(np.median(read_len_arr_passed), color="black", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_passed), 0.9 * ax.get_ylim()[1], f"Passed median = {np.median(read_len_arr_passed):.0f}", color="black")
    ax.axvline(np.median(read_len_arr_failed), color="black", linestyle="--", linewidth=2)
    ax.text(np.median(read_len_arr_failed), 0.8 * ax.get_ylim()[1], f"Failed median = {np.median(read_len_arr_failed):.0f}", color="black")

    ax.set_title(f"Poly(A) Length Distribution (n={len(read_len_arr):,})")
    ax.set_xlabel("Poly(A) Length")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 300)
    ax.legend()
    ## Vline at median
    fig.savefig(f"{out_path}/polya_len_hist.png", dpi=300)
    plt.close(fig)
    return None


def plot_qual(mean_qual_arr, out_path, bq_thres = 7, max_bq = 30):
    ## plot mean quality score KDE with histogram
    fig, ax = plt.subplots(figsize=(10,10))
    pass_arr = mean_qual_arr[mean_qual_arr >= bq_thres]
    fail_arr = mean_qual_arr[mean_qual_arr < bq_thres]
    ax.set_title(f"Read Mean Base Quality Distribution (n={len(mean_qual_arr):,})")
    pass_percent = len(pass_arr) / len(mean_qual_arr) * 100
    fail_percent = len(fail_arr) / len(mean_qual_arr) * 100
    sns.histplot(data=pass_arr, ax=ax, color = "royalblue", label=f"Pass (n={len(pass_arr):,}, {pass_percent:.2f}%)", binwidth=0.1, binrange=(0, max_bq))
    sns.histplot(data=fail_arr, ax=ax, color = "tomato", label=f"Fail (n={len(fail_arr):,}, {fail_percent:.2f}%)", binwidth=0.1, binrange=(0, max_bq))
    ## vline at median
    ax.axvline(np.median(mean_qual_arr), color="black", linestyle="--", linewidth=2)
    ax.legend()
    ax.set_xlim(0, max_bq)
    fig.savefig(f"{out_path}/mean_qual_hist.png", dpi=300)
    plt.close(fig)
    return None


def main():
    args = parse_args()

    load_success = False

    if os.path.exists(args.out_path):
        printmessage("Output directory already exists. Attempting to load pickle")
        try:
            with open(f"{args.out_path}/read_len.pkl", "rb") as f:
                read_len_arr = pickle.load(f)
            with open(f"{args.out_path}/mean_qual.pkl", "rb") as f:
                mean_qual_arr = pickle.load(f)
            with open(f"{args.out_path}/polya_len.pkl", "rb") as f:
                polya_len_arr = pickle.load(f)
            load_success = True
        except:
            printmessage("Pickle loading failed. Re-run with a different output directory")
            load_success = False

    if not load_success:
        os.makedirs(args.out_path, exist_ok=True)

        bam_file = pysam.AlignmentFile(args.bam_path, "rb", check_sq=False, threads=args.cpu)
        read_len_arr = []
        qual_arr = []
        polya_len_arr = []
        printmessage("Reading BAM file")

        for read in tqdm(bam_file, total = bam_file.mapped + bam_file.unmapped):
            if read.is_secondary:
                continue
            if read.has_tag("pi"):
                continue
            try:
                bq = mean_phred(np.array(read.query_qualities, dtype=int))
                qual_arr.append(bq)
            except:
                continue
            try:
                polya_len_arr.append(read.get_tag("pt"))
            except:
                polya_len_arr.append(0)
            read_len_arr.append(read.query_length)

        read_len_arr = np.array(read_len_arr)
        mean_qual_arr = np.array(qual_arr)
        polya_len_arr = np.array(polya_len_arr)

        printmessage("Saving pickle")
        ## save pickle
        with open(f"{args.out_path}/read_len.pkl", "wb") as f:
            pickle.dump(read_len_arr, f)
        with open(f"{args.out_path}/mean_qual.pkl", "wb") as f:
            pickle.dump(mean_qual_arr, f)
        with open(f"{args.out_path}/polya_len.pkl", "wb") as f:
            pickle.dump(polya_len_arr, f)

    printmessage("Plotting")
    if not args.mrna:
        plot_read_len_oligo(read_len_arr, mean_qual_arr, args.bq_thres, args.out_path, args.bb_length)
    else:
        plot_read_len_mrna(read_len_arr, mean_qual_arr, args.bq_thres, args.out_path)
    plot_qual(mean_qual_arr, args.out_path, bq_thres=args.bq_thres)
    plot_polya_len(polya_len_arr, mean_qual_arr, args.bq_thres, args.out_path)

    return None


if __name__ == "__main__":
    main()