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
from deeprm.utils.utils import mean_phred, printmessage
import multiprocessing as mp
from collections import deque

plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 0.5, "legend.frameon": 1, "lines.linewidth": 2})

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--in", "-i", dest="bam_path", type=str, required=True, help="Input bam file")
    args.add_argument("--out","-o", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--process", "-p", dest="process", type=int, default=int(mp.cpu_count()*0.95//4), help="Number of processes")
    args.add_argument("--threads", "-t", dest="threads", type=int, default=4, help="Number of threads")
    args.add_argument("--bq", "-q", dest="bq_thres", type=int, default=7, help="Base quality threshold")
    args.add_argument("--bb", "-b", dest="bb_length", type=int, default=71, help="BB length")
    args.add_argument("--mrna", "-m", action="store_true", help="mRNA mode")
    args.add_argument("--len", "-l", dest="len_cutoff", type=int, default=200, help="Length cutoff")
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
        for i in range(1,1000//bb_length+1):
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
    # ## vline at median
    # ax.axvline(np.median(mean_qual_arr), color="black", linestyle="--", linewidth=2)
    # ax.text(np.median(mean_qual_arr), 0.9 * ax.get_ylim()[1], f"Median = {np.median(mean_qual_arr):.2f}", color="black")
    ## vline at passed median
    ax.axvline(np.median(pass_arr), color="black", linestyle="--", linewidth=2)
    ax.text(np.median(pass_arr), 0.8 * ax.get_ylim()[1], f"Passed Median = {np.median(pass_arr):.2f}", color="black")
    ## vline at failed median
    ax.axvline(np.median(fail_arr), color="black", linestyle="--", linewidth=2)
    ax.text(np.median(fail_arr), 0.7 * ax.get_ylim()[1], f"Failed Median = {np.median(fail_arr):.2f}", color="black")
    ax.legend()
    ax.set_xlim(0, max_bq)
    fig.savefig(f"{out_path}/mean_qual_hist.png", dpi=300)
    plt.close(fig)
    return None


def read_bam_worker(args, pid, collect_dict):

    bam_file = pysam.AlignmentFile(args.bam_path, "rb", check_sq=False, threads=args.threads)
    total = bam_file.mapped + bam_file.unmapped
    proc_len = (total // args.process) + 1
    read_len_arr = deque(maxlen=proc_len)
    qual_arr = deque(maxlen=proc_len)
    polya_len_arr = deque(maxlen=proc_len)

    for i, read in tqdm(enumerate(bam_file), total = total):
        if i % args.process == pid:
            if read.is_secondary:
                continue
            if read.has_tag("pi"):
                continue
            if args.len_cutoff > 0:
                if read.query_length < args.len_cutoff:
                    continue

            bq = np.array(read.query_qualities, dtype=int)
            rl = read.query_length

            if read.has_tag("TL") and read.has_tag("TR"):
                trim_5p = read.get_tag("TL")
                trim_3p = read.get_tag("TR")
                if trim_3p - trim_5p <= 0:
                    continue
                bq = read.query_qualities[trim_5p:trim_3p]
                rl = trim_3p - trim_5p

            qual_arr.append(mean_phred(bq))
            read_len_arr.append(rl)

            try:
                polya_len_arr.append(read.get_tag("pt"))
            except:
                polya_len_arr.append(0)

    collect_dict["read_len_arr"].append(np.array(read_len_arr))
    collect_dict["qual_arr"].append(np.array(qual_arr))
    collect_dict["polya_len_arr"].append(np.array(polya_len_arr))
    return None


def main():
    args = parse_args()
    ## Check if BAM is indexed
    if not os.path.exists(args.bam_path + ".bai"):
        printmessage("BAM file is not indexed. Indexing BAM file...")
        pysam.index(args.bam_path, nthreads=args.threads * args.process)
        printmessage("BAM file indexed.")

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
        manager = mp.Manager()
        collect_dict = manager.dict()
        collect_dict["read_len_arr"] = manager.list()
        collect_dict["qual_arr"] = manager.list()
        collect_dict["polya_len_arr"] = manager.list()

        printmessage("Reading BAM file")
        processes = []
        for pid in range(args.process):
            p = mp.Process(target=read_bam_worker, args=(args, pid, collect_dict))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        printmessage("Collecting results")
        read_len_arr = np.concatenate(collect_dict["read_len_arr"])
        mean_qual_arr = np.concatenate(collect_dict["qual_arr"])
        polya_len_arr = np.concatenate(collect_dict["polya_len_arr"])

        manager.shutdown()

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