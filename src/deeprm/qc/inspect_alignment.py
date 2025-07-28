import argparse
import os, sys, re, pickle
import pysam
import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from deeprm.utils.utils import mean_phred
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
from deeprm.utils.logging import get_logger
log = get_logger(__name__)

plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 1, "legend.frameon": 1, "lines.linewidth": 2})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", "-i", type=str, dest="input", help="Input BAM file path", required=True)
    parser.add_argument("--out","-o", type=str, dest="output", help="Output Directory", required=True)
    parser.add_argument("--process","-p", type=int, dest="process", help="Number of processes",  default=int(mp.cpu_count()*0.95//4))
    parser.add_argument("--thread","-t", type=int, dest="thread", help="Number of threads per process", default=4)
    parser.add_argument("--mapq","-m", type=int, dest="mapq", help="MAPQ cutoff", default=30)
    parser.add_argument("--bq", "-b", type=int, dest="bq", help="BQ cutoff", default=7)
    parser.add_argument("--len", "-l", type=int, dest="len_cutoff", help="Length cutoff", default=0)
    args = parser.parse_args()
    return args


def extract_cigar_worker(pid, args, error_dict):
    bamfile = pysam.AlignmentFile(args.input, "rb", check_sq=False, threads=args.thread)
    total = bamfile.mapped + bamfile.unmapped
    error_rates = deque(maxlen=total//args.process+1)
    for i, read in tqdm(enumerate(bamfile), total = total):
        if i % args.process == pid:
            if args.len_cutoff > 0:
                if read.query_length < args.len_cutoff:
                    continue
            if read.is_unmapped or read.is_secondary:
                continue
            if mean_phred(np.array(read.query_qualities, dtype=int)) < args.bq:
                continue
            if read.mapping_quality < args.mapq:
                continue
            if not read.has_tag("MD"):
                continue
            error_rates.append(get_error_rate_func(read.cigarstring, read.get_tag("MD"), use_md=True))
    bamfile.close()
    error_dict[pid] = np.stack(error_rates, axis=0)
    return None


def extract_cigar_master(args):
    manager = mp.Manager()
    error_dict = manager.dict()
    proc_list = []
    for pid in range(args.process):
        proc = mp.Process(target=extract_cigar_worker, args=(pid, args, error_dict))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    error_df = np.concatenate(list(error_dict.values()), axis=0)
    error_df = pd.DataFrame(error_df, columns=["MIS_RATE", "INS_RATE", "DEL_RATE"])
    error_df["ERROR_RATE"] = error_df["MIS_RATE"] + error_df["INS_RATE"] + error_df["DEL_RATE"]
    return error_df


def md_to_mismatch_arr(md):
    """
    Convert MD tag to mismatch array.
    1 = mismatch, 0 = match. Deletions are ignored (filled as matches).
    """
    # Pattern: numbers (matches), ^[A-Z]+ (deletions), [A-Z] (mismatches)
    pattern = re.compile(r'(\d+)|(\^[A-Z]+)|([A-Z])')
    result = []
    for match in pattern.finditer(md):
        if match.group(1):  # Match run
            result.append(np.zeros(int(match.group(1)), dtype=int))
        elif match.group(2):  # Deletion (e.g., ^AC)
            # Deletions are not counted as mismatches, fill with 0s for length of deletion
            result.append(np.zeros(len(match.group(2)[1:]), dtype=int))
        elif match.group(3):  # Mismatch
            result.append(np.ones(1, dtype=int))
    return np.concatenate(result, axis=0)


def get_error_rate_func(cigar, md, use_md = True):

    cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
    mismatch = 0
    insertion = 0
    deletion = 0
    ref_length = 0

    for length, match in cigar_list:
        if match in ["=","M"]:
            ref_length += int(length)
        elif match == "X":
            mismatch += int(length)
            ref_length += int(length)
        elif match == "I":
            insertion += int(length)
        elif match == "D":
            deletion += int(length)
            ref_length += int(length)

    mis_rate = mismatch/ref_length
    ins_rate = insertion/ref_length
    del_rate = deletion/ref_length

    if use_md:
        mis_arr = md_to_mismatch_arr(md)
        mis_rate = np.sum(mis_arr) / ref_length

    error_rates = np.array((mis_rate, ins_rate, del_rate))
    return error_rates


def plot_kde(df_error, args):

    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xlabel("Read Alignment accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0.8, 1.0)
    sns.histplot(1-df_error["ERROR_RATE"], ax=ax, color="royalblue",
                 label=f"Pass (n={len(df_error):,})", binwidth=0.001, binrange=(0.7, 1.0), kde=False, stat="density")
    ## vline at median
    median = np.median(1 - df_error["ERROR_RATE"])
    ax.axvline(median, color="royalblue", linestyle='--', linewidth=3)
    ax.text(median, 0.9 * ax.get_ylim()[1], f"{median:.3f}", color="royalblue")
    ax.legend()
    plt.savefig(f"{args.output}/per_read_phred_error_kde.png", dpi=300)
    plt.close()
    return None


def plot_boxplot(df_error, args):
    ## Plot mis, ins, del rate
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20,20))
    ## Use whiskers, no fliers
    sns.boxplot(data=df_error[["MIS_RATE", "INS_RATE", "DEL_RATE"]], ax=ax, palette="Set2", linewidth=3, fliersize=0)
    ax.set_xlabel("Error type")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 0.03)
    ax.legend()
    plt.savefig(f"{args.output}/per_read_error_boxplot.png", dpi=300)
    plt.close()
    return None


def main():
    args = parse_args()
    run_flag = True

    if os.path.exists(args.output):
        log.info("Output directory already exists. Attempting to load pickle")
        try:
            error_df = pd.read_pickle(f"{args.output}/error_rate.pkl")
            run_flag = False
        except:
            "Error rate pickle does not exist. Re-running alignment and error rate calculation"
            run_flag = True

    if run_flag:
        os.makedirs(args.output, exist_ok=True)

        log.info("Extracting CIGAR string")
        error_df = extract_cigar_master(args)
        error_df.to_pickle(f"{args.output}/error_rate.pkl")


    assert len(error_df) > 0, "Error rate dataframe is empty. Check input BAM file"

    log.info("Plotting error rate")
    plot_kde(error_df, args)
    plot_boxplot(error_df, args)

    return None



if __name__ == "__main__":
    main()