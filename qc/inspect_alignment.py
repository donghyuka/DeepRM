import argparse
import os, sys, re, pickle
import pysam
import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from utils.utils import mean_phred, printmessage
from matplotlib import pyplot as plt
import seaborn as sns


REF_PATH = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, dest="input", help="Input BAM file path", required=True)
    parser.add_argument("--out", type=str, dest="output", help="Output Directory", required=True)
    parser.add_argument("--ref", type=str, dest="ref", help="Reference FASTA file path", default=REF_PATH)
    parser.add_argument("--cpu", type=int, dest="cpu", help="Number of CPUs",  default=int(os.cpu_count()*0.9))
    parser.add_argument("--mapq", type=int, dest="mapq", help="MAPQ cutoff", default=30)
    parser.add_argument("--bq", type=int, dest="bq", help="BQ cutoff", default=7)
    args = parser.parse_args()
    return args

def align_bam(args):
    basename = os.path.basename(args.input)
    aln_sam_path = os.path.join(args.output, basename)
    aln_sam_path = aln_sam_path.replace(".bam", ".aligned.sam")
    aln_bam_path = aln_sam_path.replace(".sam", ".fltered.bam")

    if os.path.exists(aln_bam_path):
        print(f"Aligned BAM file {aln_bam_path} already exists. Skipping alignment")

    else:
        ## 1. Run minimap2
        cmd = f"samtools fastq -TXX,YY {args.input} | minimap2 --eqx -y -N 1 -ax map-ont -t {args.cpu} {args.ref} - > {aln_sam_path}"
        os.system(cmd)

        ## 2. Convert to BAM, filter, sort, and index.
        cmd = f"samtools view -bhS -F 4095 -q {args.mapq} {aln_sam_path} | samtools sort -@ {args.cpu} -o {aln_bam_path}"
        os.system(cmd)
        cmd = f"samtools index -@ {args.cpu} {aln_bam_path}"
        os.system(cmd)

    return aln_bam_path


def extract_cigar(args):
    bamfile = pysam.AlignmentFile(args.input, "rb", check_sq=False, threads=args.cpu)
    cigar_list = []
    for read in tqdm(bamfile):
        if read.is_unmapped or read.is_secondary:
            continue
        mean_bq = mean_phred(np.array(read.query_qualities, dtype=int))
        if mean_bq < args.bq:
            continue
        mapq = read.mapping_quality
        if mapq < args.mapq:
            continue
        cigar = read.cigarstring
        cigar_list.append(cigar)
    return cigar_list



def get_error_rate_func(cigar):

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

    return mis_rate, ins_rate, del_rate


def get_error_rate_worker(cigar_list, man_df_list):
    mis_rate_list = []
    ins_rate_list = []
    del_rate_list = []
    for cigar in cigar_list:
        mis_rate, ins_rate, del_rate = get_error_rate_func(cigar)
        mis_rate_list.append(mis_rate)
        ins_rate_list.append(ins_rate)
        del_rate_list.append(del_rate)
    df = pd.DataFrame({"MIS_RATE":mis_rate_list, "INS_RATE":ins_rate_list, "DEL_RATE":del_rate_list})
    man_df_list.append(df)
    return None


def get_error_rate_master(cigar_list, args):

    cigar_list_split = np.array_split(cigar_list, args.cpu)
    proc_list = []
    man = mp.Manager()
    man_df_list = man.list()

    for cigar_list in cigar_list_split:
        proc = mp.Process(target=get_error_rate_worker, args=(cigar_list, man_df_list))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    man_df_list = list(man_df_list)
    df_error = pd.concat(man_df_list, axis=0)
    df_error["ERROR_RATE"] = df_error["MIS_RATE"] + df_error["INS_RATE"] + df_error["DEL_RATE"]
    return df_error



def plot_kde(df_error, args):

    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xlabel("Read Alignment accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0.7, 1.0)
    sns.histplot(1-df_error["ERROR_RATE"], ax=ax, color="royalblue",
                 label=f"Pass (n={len(df_error):,})", binwidth=0.001, binrange=(0.7, 1.0), kde=True, stat="density")
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
    ax.set_ylim(0, 0.08)
    ax.legend()
    plt.savefig(f"{args.output}/per_read_error_boxplot.png", dpi=300)
    plt.close()
    return None



def main():
    args = parse_args()
    df_error = []
    run_flag = True

    if os.path.exists(args.output):
        printmessage("Output directory already exists. Attempting to load pickle")
        try:
            df_error = pd.read_pickle(f"{args.output}/error_rate.pkl")
            run_flag = False
        except:
            "Error rate pickle does not exist. Re-running alignment and error rate calculation"
            run_flag = True

    if run_flag:
        os.makedirs(args.output, exist_ok=True)

        printmessage("Extracting CIGAR string")
        cigar_list = extract_cigar(args)
        with open(f"{args.output}/cigar_list.pkl", "wb") as f:
            pickle.dump(cigar_list, f)

        printmessage("Calculating error rate")
        df_error = get_error_rate_master(cigar_list, args)
        df_error.to_pickle(f"{args.output}/error_rate.pkl")

    assert len(df_error) > 0, "Error rate dataframe is empty. Check input BAM file"

    printmessage("Plotting error rate")
    plot_kde(df_error, args)
    plot_boxplot(df_error, args)

    return None

def plot_error_rate(error_dict, label_dict):
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xlabel("Read Alignment accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0.7, 1.0)
    clist = ["royalblue", "tomato", "mediumseagreen", "gold"]
    for i, exp in enumerate(error_dict):
        color = clist.pop()
        error_arr = error_dict[exp]
        sns.kdeplot(1-error_arr, ax=ax, label=label_dict[exp]+f" (n={len(error_arr):,})", linewidth=6, color=color, shade=False, )
        median = np.median(1 - error_dict[exp])
        ax.axvline(median, color=color, linestyle='--', linewidth=3)
        ax.text(median, 20-3*i, f"{median:.3f}", color=color)
    ax.legend()
    plt.savefig(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/error_rate.png", dpi=300)
    plt.close()
    return None



def main2():
    label_dict = {"ON0074":"ON0074 (RNA002/minION/HEK)", "ON0086":"ON0086 (RNA004/minION/HEK)",
                  "ON0090":"ON0090 (RNA004/P2solo/HEK)", "ON0091":"ON0091 (RNA004/P2Solo/HeLa)"}
    df_1 = pd.read_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0074/ON0074/save_path/align/per_read_phred_error.tsv", sep='\t')
    df_2 = pd.read_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0086/ON0086/save_path/align/per_read_phred_error.tsv", sep='\t')
    error_1 = df_1["ERROR_RATE"]
    error_2 = df_2["ERROR_RATE"]
    df_3 = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/alignstats/error_rate.pkl")
    df_4 = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/alignstats/error_rate.pkl")
    error_3 = df_3["ERROR_RATE"]
    error_4 = df_4["ERROR_RATE"]
    error_dict = {"ON0074":error_1, "ON0086":error_2, "ON0090":error_3, "ON0091":error_4}
    with open("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/error_rate.pkl", "wb") as f:
        pickle.dump(error_dict, f)
    plot_error_rate(error_dict, label_dict)
    return None

if __name__ == "__main__":
    main()