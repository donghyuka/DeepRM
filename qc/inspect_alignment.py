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


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", "-i", type=str, dest="input", help="Input BAM file path", required=True)
    parser.add_argument("--out","-o", type=str, dest="output", help="Output Directory", required=True)
    parser.add_argument("--ref","-r", type=str, dest="ref", help="Reference FASTA file path", required=True)
    parser.add_argument("--cpu","-c", type=int, dest="cpu", help="Number of CPUs",  default=int(os.cpu_count()*0.9))
    parser.add_argument("--mapq","-m", type=int, dest="mapq", help="MAPQ cutoff", default=30)
    parser.add_argument("--bq", "-b", type=int, dest="bq", help="BQ cutoff", default=7)
    args = parser.parse_args()
    return args


def md_to_mismatch_arr(md):
    """
    Convert MD tag to mismatch array.

    Args:
        md (str): MD tag from BAM file.

    Returns:
        np.ndarray: A numpy array with shape (len(read),) with 1 for mismatch and 0 for match.
    """
    mis_arr = []
    digit_buffer = ""
    del_flag = False
    del_count = 0
    for char in md:
        if char.isdigit():
            if del_flag:
                del_flag = False
                if del_count > 0:
                    mis_arr += [0] * del_count
                    del_count = 0
            digit_buffer += char
        else:
            if del_flag:
                del_count += 1
                continue
            if len(digit_buffer) > 0:
                digit_buffer = int(digit_buffer)
                if digit_buffer > 0:
                    mis_arr += [0] * digit_buffer
                digit_buffer = ""
            if char == "^":
                del_flag = True
            else:
                mis_arr.append(1)

    if del_flag:
        if del_count > 0:
            mis_arr += [0] * del_count
    if len(digit_buffer) > 0:
        digit_buffer = int(digit_buffer)
        if digit_buffer > 0:
            mis_arr += [0] * digit_buffer

    mis_arr = np.array(mis_arr, dtype=int)
    return mis_arr


def align_bam(args):
    """
    Aligns BAM file using minimap2 and processes the alignment.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: Path to the aligned BAM file.
    """
    basename = os.path.basename(args.input)
    aln_sam_path = os.path.join(args.output, basename)
    aln_sam_path = aln_sam_path.replace(".bam", ".aligned.sam")
    aln_bam_path = aln_sam_path.replace(".sam", ".fltered.bam")

    if os.path.exists(aln_bam_path):
        print(f"Aligned BAM file {aln_bam_path} already exists. Skipping alignment")

    else:
        # 1. Run minimap2
        cmd = f"samtools fastq -TXX,YY {args.input} | minimap2 --eqx -y -N 1 -ax map-ont -t {args.cpu} {args.ref} - > {aln_sam_path}"
        os.system(cmd)

        # 2. Convert to BAM, filter, sort, and index.
        cmd = f"samtools view -bhS -F 4095 -q {args.mapq} {aln_sam_path} | samtools sort -@ {args.cpu} -o {aln_bam_path}"
        os.system(cmd)
        cmd = f"samtools index -@ {args.cpu} {aln_bam_path}"
        os.system(cmd)

    return aln_bam_path


def extract_cigar(args):
    """
    Extracts CIGAR strings and MD tags from the BAM file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: A tuple containing a list of CIGAR strings and a list of MD tags.
    """
    bamfile = pysam.AlignmentFile(args.input, "rb", check_sq=False, threads=args.cpu)
    cigar_list = []
    md_list = []
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
        md = read.get_tag("MD")
        md_list.append(md)

    return cigar_list, md_list


def get_error_rate_func(cigar, md, use_md=True):
    """
    Calculates error rates from CIGAR string and MD tag.

    Args:
        cigar (str): CIGAR string.
        md (str): MD tag.
        use_md (bool, optional): Whether to use MD tag for mismatch calculation. Defaults to True.

    Returns:
        tuple: A tuple containing mismatch rate, insertion rate, and deletion rate.
    """
    cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
    mismatch = 0
    insertion = 0
    deletion = 0
    ref_length = 0

    for length, match in cigar_list:
        if match in ["=", "M"]:
            ref_length += int(length)
        elif match == "X":
            mismatch += int(length)
            ref_length += int(length)
        elif match == "I":
            insertion += int(length)
        elif match == "D":
            deletion += int(length)
            ref_length += int(length)

    mis_rate = mismatch / ref_length
    ins_rate = insertion / ref_length
    del_rate = deletion / ref_length

    if use_md:
        mis_arr = md_to_mismatch_arr(md)
        mis_rate = np.sum(mis_arr) / ref_length

    return mis_rate, ins_rate, del_rate


def get_error_rate_worker(cigar_list, md_list, man_df_list, use_md=True):
    """
    Worker function to calculate error rates for a list of CIGAR strings and MD tags.

    Args:
        cigar_list (list): List of CIGAR strings.
        md_list (list): List of MD tags.
        man_df_list (list): Manager list to store results.
        use_md (bool, optional): Whether to use MD tag for mismatch calculation. Defaults to True.

    Returns:
        None
    """
    mis_rate_list = []
    ins_rate_list = []
    del_rate_list = []
    for cigar, md in zip(cigar_list, md_list):
        mis_rate, ins_rate, del_rate = get_error_rate_func(cigar, md, use_md)
        ins_rate_list.append(ins_rate)
        del_rate_list.append(del_rate)
        mis_rate_list.append(mis_rate)
    df = pd.DataFrame({"MIS_RATE": mis_rate_list, "INS_RATE": ins_rate_list, "DEL_RATE": del_rate_list})
    man_df_list.append(df)
    return None


def get_error_rate_master(cigar_list, md_list, args):
    """
    Master function to calculate error rates using multiprocessing.

    Args:
        cigar_list (list): List of CIGAR strings.
        md_list (list): List of MD tags.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        pd.DataFrame: DataFrame containing error rates.
    """
    cigar_list_split = np.array_split(cigar_list, args.cpu)
    md_list_split = np.array_split(md_list, args.cpu)
    proc_list = []
    man = mp.Manager()
    man_df_list = man.list()

    for (cigar_list, md_list) in zip(cigar_list_split, md_list_split):
        proc = mp.Process(target=get_error_rate_worker, args=(cigar_list, md_list, man_df_list))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    man_df_list = list(man_df_list)
    df_error = pd.concat(man_df_list, axis=0)
    df_error["ERROR_RATE"] = df_error["MIS_RATE"] + df_error["INS_RATE"] + df_error["DEL_RATE"]
    return df_error


def plot_kde(df_error, args):
    """
    Plots a KDE plot of read alignment accuracy.

    Args:
        df_error (pd.DataFrame): DataFrame containing error rates.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlabel("Read Alignment accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0.7, 1.0)
    sns.histplot(1 - df_error["ERROR_RATE"], ax=ax, color="royalblue",
                 label=f"Pass (n={len(df_error):,})", binwidth=0.001, binrange=(0.7, 1.0), kde=True, stat="density")
    # vline at median
    median = np.median(1 - df_error["ERROR_RATE"])
    ax.axvline(median, color="royalblue", linestyle='--', linewidth=3)
    ax.text(median, 0.9 * ax.get_ylim()[1], f"{median:.3f}", color="royalblue")
    ax.legend()
    plt.savefig(f"{args.output}/per_read_phred_error_kde.png", dpi=300)
    plt.close()
    return None


def plot_boxplot(df_error, args):
    """
    Plots a boxplot of mismatch, insertion, and deletion rates.

    Args:
        df_error (pd.DataFrame): DataFrame containing error rates.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(figsize=(20, 20))
    # Use whiskers, no fliers
    sns.boxplot(data=df_error[["MIS_RATE", "INS_RATE", "DEL_RATE"]], ax=ax, palette="Set2", linewidth=3, fliersize=0)
    ax.set_xlabel("Error type")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 0.08)
    ax.legend()
    plt.savefig(f"{args.output}/per_read_error_boxplot.png", dpi=300)
    plt.close()
    return None


def main():
    """
    Main function to parse arguments, calculate error rates, and plot results.

    Returns:
        None
    """
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
        cigar_list, md_list = extract_cigar(args)
        with open(f"{args.output}/cigar_list.pkl", "wb") as f:
            pickle.dump(cigar_list, f)

        printmessage("Calculating error rate")
        df_error = get_error_rate_master(cigar_list, md_list, args)
        df_error.to_pickle(f"{args.output}/error_rate.pkl")

    assert len(df_error) > 0, "Error rate dataframe is empty. Check input BAM file"

    printmessage("Plotting error rate")
    plot_kde(df_error, args)
    plot_boxplot(df_error, args)

    return None


if __name__ == "__main__":
    main()