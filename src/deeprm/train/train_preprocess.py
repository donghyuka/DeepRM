"""
DeepRM Training Data Preprocessing Module

This module provides functions for preprocessing training data for DeepRM.
It includes functions for extracting move tags from BAM files, preprocessing POD5 files,
and segmenting and normalizing signal data.
"""

import argparse
import atexit
import gc
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pod5
import pysam
import tqdm

from deeprm.train.extract_block import extract_block
from deeprm.utils.logging import get_logger
from deeprm.utils.memory import start_mem_watchdog

log = get_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.
    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    num_cpu = os.cpu_count()
    parser.add_argument("--bam", "-b", dest="input", type=str, required=True, help="Input BAM file")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--output", "-o", dest="output", type=str, required=True)
    parser.add_argument("--threads", "-t", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    ## DAG extraction parameters
    parser.add_argument("--it", dest="indel_tolerance", type=int, default=3)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=3)
    parser.add_argument("--cst", dest="cb_size_tolerance", type=int, default=3)
    parser.add_argument("--kst", dest="skip_size_tolerance", type=int, default=4)
    parser.add_argument("--amp", dest="anchor_mismatch_penalty", type=int, default=6)
    parser.add_argument("--smt", dest="spacer_mismatch_tolerance", type=int, default=3)
    parser.add_argument("--smp", dest="spacer_mismatch_penalty", type=int, default=2)
    parser.add_argument("--sst", dest="spacer_size_tolerance", type=int, default=1)
    parser.add_argument("--ac", dest="anchor_list", type=str, nargs="+", default=["A", "A", "A"])
    parser.add_argument(
        "--sp",
        dest="spacer_list",
        type=str,
        nargs="+",
        default=["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"],
    )
    parser.add_argument("--ss", dest="spacer_size", type=int, default=6)
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cb", dest="cb_per_bb", type=int, default=3)
    parser.add_argument("--rbq", dest="read_bq_cutoff", type=int, default=7)
    parser.add_argument("--cbq", dest="cb_bq_cutoff", type=int, default=0)
    parser.add_argument("--fi", dest="flush_interval", type=int, default=1000)
    parser.add_argument("--max", dest="max_read_length", type=int, default=1000)
    parser.add_argument("--min", dest="min_read_length", type=int, default=0)
    parser.add_argument("--sample", dest="sample", type=int, default=None)
    parser.add_argument(
        "--cfg",
        dest="config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "dag_config.json"),
        help="Path to the configuration file in JSON format. \
                        If provided, it will override the command-line arguments.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        type=str,
        default=None,
        help="Continue from previous run. Provide the path to the previous output.",
    )

    ## Signal preprocessing parameters
    parser.add_argument("--chunk", "-n", type=int, default=500, help="POD5 Chunk size")
    parser.add_argument("--max-size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min-size", "-i", type=int, default=10, help="Minimum POD5 dataframe size in MB")
    parser.add_argument("--keep", action="store_true", help="Keep intermediate files", default=True)
    parser.add_argument("--postfix", "-x", type=str, default="data", help="Output file postfix")
    return None


def main(args: argparse.Namespace):
    """
    Main function to extract context blocks from a basecalled BAM file
    using a directed acyclic graph (DAG).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """

    if args.config is not None:
        with open(args.config) as config_file:
            config_dict = json.load(config_file)
            for key, value in config_dict.items():
                setattr(args, key, value)
            log.info(f"Loaded configuration from: {args.config}")
            assert len(args.anchor_list) == args.cb_per_bb
    assert len(args.spacer_list) == args.cb_per_bb + 1
    assert args.skip_size_tolerance >= args.cb_size_tolerance

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"ERROR! {args.resume} does not exist.")

    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input POD5 directory {args.pod5} does not exist")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input BAM file {args.input} does not exist")
    if os.path.exists(args.output):
        raise FileExistsError(
            f"Output directory {args.output} already exists. \
        Please choose a different output directory or remove the existing one."
        )

    os.makedirs(args.output, exist_ok=True)

    norm_factor = get_norm_factor()

    token_output_path = f"{args.output}/{args.postfix}/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    os.makedirs(f"{intermediate_path}/block_df_split", exist_ok=True)
    atexit.register(lambda: os.system(f"rm -r {intermediate_path}"))

    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.ncpu, args.chunk, args.max_size, args.min_size)
    signal_path_arr = list(index_dict.keys())
    signal_name_arr = [x.split("/")[-1] for x in signal_path_arr]
    gc.collect()

    if len(signal_path_arr) == 0:
        log.error("No valid signal files found. Exiting.")
        raise FileNotFoundError("No valid signal files found in the provided POD5 directory.")

    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(
        index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"
    ):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path.split("/")[-1]

    del index_dict
    gc.collect()

    if args.resume is not None:
        flush_path = args.resume

    else:
        flush_path = f"{args.output}/block_flush_{time.strftime('%Y%m%d%H%M%S')}/"
        os.makedirs(flush_path, exist_ok=True)
        atexit.register(os.system, f"rm -r {flush_path}")

    args_dict = vars(args)
    block_df = extract_block(**args_dict, flush_path=flush_path)

    split_block_df(signal_path_dict, signal_name_arr, intermediate_path, block_df)
    del block_df
    gc.collect()

    extract_move(args.input, args.ncpu, signal_path_dict, signal_name_arr, intermediate_path)

    del signal_path_dict, signal_name_arr
    gc.collect()

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.ncpu))

    proc_list = []
    for signal_paths in signal_path_arr_split:
        proc = mp.Process(
            target=segment_normalize_signal,
            args=(args.output, args.postfix, signal_paths, norm_factor),
        )
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    log.info("Signal Segmentation and Tokenization Complete")
    log.info("Saved to: " + args.output)
    return None


def extract_move(bam_path, ncpu, signal_path_dict, signal_path_arr, intermediate_path):
    """
    Extracts the 'mv' tag from a BAM file and saves it to separate files.

    Args:
        bam_path (str): Path to the BAM file.
        ncpu (int): Number of CPU threads to use.
        signal_path_dict (dict): Dictionary mapping read IDs to signal paths.
        signal_path_arr (list): List of signal paths.
        intermediate_path (str): Path to save intermediate files.

    Returns:
        None
    """
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": []} for x in signal_path_arr}
    count = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                else:
                    read_id = str(read.query_name)

                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except Exception:
                    continue

                if read.has_tag("mv"):
                    mv = read.get_tag("mv")
                else:
                    continue

                if read.has_tag("ts"):
                    ts = read.get_tag("ts")
                else:
                    ts = 0

                if read.has_tag("ns"):
                    ns = read.get_tag("ns")
                else:
                    ns = 0

                if read.has_tag("sp"):
                    sp = read.get_tag("sp")
                else:
                    sp = 0

                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["mv"].append(mv)
                data["sp"].append(sp)
                count += 1

                pbar.update(1)

    log.info(f"Valid read count: {count}")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None


def preprocess_pod5(pod5_path, save_path, ncpu, chunk, max_mb, min_mb):
    """
    Exports POD5 files to DataFrame format.

    Args:
        pod5_path (str): Path to the POD5 files.
        save_path (str): Path to save the DataFrame files.
        ncpu (int): Number of CPU threads to use.
        chunk (int): Chunk size for processing.
        max_mb (int): Maximum size of the DataFrame in MB.
        min_mb (int): Minimum size of the DataFrame in MB.

    Returns:
        dict: Dictionary mapping file paths to read IDs.
    """
    pod5_path_list = glob.glob(os.path.join(pod5_path, "*.pod5"))
    proc_list = []
    np.random.shuffle(pod5_path_list)
    pod5_path_list_split = np.array_split(pod5_path_list, ncpu)

    man = mp.Manager()
    index_list = man.list()

    for pid in range(ncpu):
        proc = mp.Process(
            target=extract_signal_proc,
            args=(pod5_path_list_split[pid], save_path, pid, index_list, chunk, max_mb, min_mb),
        )
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    index_dict = {}
    for local_index_dict in index_list:
        index_dict.update(local_index_dict)

    man.shutdown()
    gc.collect()

    return index_dict


def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_list, chunk, max_mb, min_mb):
    """
    Extracts signal data from POD5 files and processes it.

    Args:
        pod5_path_list (list): List of POD5 file paths.
        signal_df_path (str): Path to save the signal data.
        pid (int): Process ID.
        index_list (list): List to store the index data.
        chunk (int): Chunk size for processing.
        max_mb (int): Maximum size of the dataframe in MB.
        min_mb (int): Minimum size of the dataframe in MB.

    Returns:
        None
    """
    index_dict_local = {}
    chunk_buffer = []
    pod5_idx = 0
    start_mem_watchdog()

    for pod5_idx, pod5_path in tqdm.tqdm(
        enumerate(pod5_path_list), total=len(pod5_path_list), desc="Parsing POD5 Files"
    ):
        signal_list = []
        offset_list = []
        scale_list = []
        id_list = []
        try:
            with pod5.Reader(pod5_path) as reader:
                skipped = 0
                for record in reader:
                    try:
                        signal_arr = record.signal
                        offset = record.calibration.offset
                        scale = record.calibration.scale
                        id = str(record.read_id)
                    except Exception:
                        skipped += 1
                        continue

                    offset_list.append(offset)
                    scale_list.append(scale)
                    signal_list.append(signal_arr)
                    id_list.append(id)

            if skipped > 0:
                log.warning(f"Skipped {skipped} faulty records in: {pod5_path}")

        except Exception as e:
            ## Pod5 file is corrupted
            log.warning(f"{e}")
            log.warning(f"Corrupted POD5 file: {pod5_path} - Skipping")
            continue

        df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
        del signal_list, offset_list, scale_list, id_list
        gc.collect()

        save_idx = 0

        for chunk_idx in range(0, len(df) // chunk + 1):
            signal_df = df.iloc[chunk_idx * chunk : min((chunk_idx + 1) * chunk, len(df))].copy()
            df_size = sys.getsizeof(signal_df) / (1024**2)
            if len(signal_df) == chunk and df_size > min_mb:
                save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer.append(signal_df)
            gc.collect()

        if len(chunk_buffer) > 0:
            signal_df = pd.concat(chunk_buffer, ignore_index=True)
            df_size = sys.getsizeof(signal_df) / (1024**2)
            if df_size > min_mb:
                chunk_buffer = []
                write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer = [signal_df]
        gc.collect()

    if len(chunk_buffer) > 0:
        signal_df = pd.concat(chunk_buffer, ignore_index=True)
        save_idx = 0
        pod5_idx += 1
        write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
        gc.collect()

    index_list.append(index_dict_local)

    return None


def write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb):
    """
    Writes the signal dataframe to a file.

    Args:
        signal_df (pandas.DataFrame): Dataframe containing the signal data.
        signal_df_path (str): Path to save the signal data.
        pid (int): Process ID.
        pod5_idx (int): POD5 file index.
        save_idx (int): Save index.
        index_dict (dict): Dictionary to store the index data.
        max_mb (int): Maximum size of the dataframe in MB.

    Returns:
        int: Updated save index.
    """
    save_idx += 1
    df_size = sys.getsizeof(signal_df) / (1024**2)
    if df_size > max_mb and len(signal_df) > 1:
        ## Split the dataframe
        split_num = min(int(df_size // max_mb) + 1, len(signal_df))
        split_size = len(signal_df) // split_num
        for split_idx in range(split_num):
            split_df = signal_df.iloc[split_idx * split_size : min((split_idx + 1) * split_size, len(signal_df))].copy()
            save_idx = write_df(split_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb)
    else:
        save_path = f"{signal_df_path}/{pid}-{pod5_idx}-{save_idx}.pkl"
        if os.path.exists(save_path):
            raise FileExistsError(f"File {save_path} already exists")
        signal_df.to_pickle(save_path)
        id_list = signal_df["read_id"].tolist()
        index_dict[save_path] = id_list
    return save_idx


def sequence_to_kmer_token(seq, kmer):
    """
    Converts a DNA/RNA sequence to k-mer tokens.

    Args:
        seq (str): DNA/RNA sequence.
        kmer (int): Length of the k-mer.

    Returns:
        numpy.ndarray: Array of k-mer tokens.
    """
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace("A", "0")
    seq = seq.replace("C", "1")
    seq = seq.replace("G", "2")
    seq = seq.replace("T", "3")
    seq = seq.replace("U", "3")
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i : kmer + i] for i in range(len(seq) - kmer + 1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4 ** np.arange(kmer).reshape(-1, 1)
    seq = np.sum(seq * quaternary, axis=0) + 1  ## 0 is reserved for padding
    seq = seq.astype(np.int16)
    return seq


def create_segment_len_arr(segment_arr, sampling):
    """
    Creates an array of segment lengths.

    Args:
        segment_arr (list): List of segments.
        sampling (int): Sampling rate.

    Returns:
        numpy.ndarray: Array of segment lengths.
    """
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int) // sampling
    return segment_len_arr


def expand_token_to_segment(token_arr, segment_len_arr):
    """
    Expands tokens to segments.

    Args:
        token_arr (numpy.ndarray): Array of tokens.
        segment_len_arr (numpy.ndarray): Array of segment lengths.

    Returns:
        numpy.ndarray: Expanded array of tokens.
    """
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_move_token(segment_len_arr):
    """
    Creates move tokens.

    Args:
        segment_len_arr (numpy.ndarray): Array of segment lengths.

    Returns:
        numpy.ndarray: Array of move tokens.
    """
    token = np.arange(1, len(segment_len_arr) + 1, dtype=np.uint8)
    token = np.repeat(token, segment_len_arr)
    return token


def create_target_mask(segment_len_arr, lr_pad):
    """
    Creates a target mask.

    Args:
        segment_len_arr (numpy.ndarray): Array of segment lengths.
        lr_pad (int): Left-right padding.

    Returns:
        numpy.ndarray: Target mask.
    """
    binary_mask = np.zeros(2 * lr_pad + 1, dtype=np.uint8)
    binary_mask[lr_pad] = 1
    binary_mask = np.repeat(binary_mask, segment_len_arr)
    return binary_mask


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    """
    Segments and pads the signal.

    Args:
        signal_segmented (numpy.ndarray): Segmented signal.
        segment_len_arr (numpy.ndarray): Array of segment lengths.
        kmer (int): Length of the k-mer.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        pad_to (int): Padding size.

    Returns:
        numpy.ndarray: Padded signal.
    """
    try:
        kmer_pad = (kmer - 1) // 2
        lr_pad = (sig_window - 1) // 2
        l_skip = (np.sum(segment_len_arr[:kmer_pad]) - lr_pad) * sampling
        r_skip = (np.sum(segment_len_arr[-kmer_pad:]) - lr_pad) * sampling
        assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
        assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]

        padding = (pad_to + kmer - 1) * sampling - len(signal_segmented)
        if padding > 0:
            signal_segmented = np.pad(signal_segmented, (0, padding), mode="constant", constant_values=0)

    except Exception as e:
        log.error(f"Error in segmented_signal_to_block: {e}")
        return None
    return signal_segmented


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Converts move data to dwell time.

    Args:
        move (numpy.ndarray): Move data.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        numpy.ndarray: Dwell time data.
    """
    sampling = move[0]
    move = np.flip(move[1:]) * np.arange(1, len(move))
    move = move[move > 0]
    move = np.concatenate([np.zeros(1, dtype=int), move])
    move = move[1:] - move[:-1]
    move = move * sampling
    move = np.log10(move.astype(np.float32))
    quantile_a_value = np.quantile(move, quantile_a)
    quantile_b_value = np.quantile(move, quantile_b)
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    move = (move - q_shift) / q_scale
    return move


def trim_scale_segment_signal(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Trims and scales the signal.

    Args:
        signal (numpy.ndarray): Signal data.
        move (numpy.ndarray): Move data.
        sp (int): Start position.
        ts (int): Timestamp.
        ns (int): Number of samples.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        numpy.ndarray: Trimmed and scaled signal.
    """
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)

    quantile_a_value = np.quantile(signal, quantile_a)
    quantile_b_value = np.quantile(signal, quantile_b)

    q_shift = shift_mult * (quantile_a_value + quantile_b_value)
    q_scale = scale_mult * (quantile_b_value - quantile_a_value)
    signal = (signal - q_shift) / q_scale
    signal = signal.astype(np.float32)
    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def segment_normalize_signal(
    seg_df_path,
    postfix,
    signal_path_arr,
    norm_factor,
    kmer=5,
    cb_len=21,
    sampling=6,
    sig_window=5,
    max_penalty=10,
    chunk_size=1000,
    max_token_len=200,
    dwell_shift=10,
):
    """
    Segments and normalizes the signal data.

    Args:
        seg_df_path (str): Path to the segmented dataframe.
        postfix (str): Postfix for the output files.
        signal_path_arr (list): List of signal paths.
        norm_factor (dict): Normalization factors.
        kmer (int): Length of the k-mer. Defaults to 5. (optional)
        cb_len (int): Length of the codebook. Defaults to 21. (optional)
        sampling (int): Sampling rate. Defaults to 6. (optional)
        sig_window (int): Signal window size. Defaults to 5. (optional)
        max_penalty (int): Maximum penalty. Defaults to 10. (optional)
        chunk_size (int): Chunk size for processing. Defaults to 1000. (optional)
        max_token_len (int): Maximum token length. Defaults to 200. (optional)
        dwell_shift (int): Dwell shift. Defaults to 10. (optional)

    Returns:
        None
    """

    start_mem_watchdog()

    trim = kmer // 2

    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]
    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]

    for signal_path in tqdm.tqdm(signal_path_arr, total=len(signal_path_arr), desc="Segmenting and Tokenizing Signals"):
        file_id = signal_path.split("/")[-1]

        if not os.path.exists(signal_path):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/move_df_split/{file_id}"):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/block_df_split/{file_id}"):
            continue

        out_path = f"{seg_df_path}/{postfix}/{file_id}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(signal_path)
        move_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}")
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["dwell_token"] = signal_df["mv"].apply(lambda x: move_to_dwell(x, 0.2, 0.8, 0.5, 1.5))
        signal_df["signal"] = signal_df.apply(
            lambda x: trim_scale_segment_signal(
                x["signal"],
                x["mv"],
                x["sp"],
                x["ts"],
                x["ns"],
                quantile_a,
                quantile_b,
                shift_mult,
                scale_mult,
            ),
            axis=1,
        )

        signal_df = signal_df[signal_df["signal"].notnull()][["read_id", "signal", "dwell_token"]].copy()

        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        # block_df = block_df[block_df["penalty"]==0]
        if len(block_df) == 0:
            continue

        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()

        signal_df["signal_length"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["end_pos"] + dwell_shift - trim < signal_df["signal_length"]]

        if len(signal_df) == 0:
            continue

        signal_df["block_score"] = signal_df["penalty"].apply(lambda x: 1 - (x / max_penalty))
        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"] : x["end_pos"]], axis=1)
        signal_df["dwell_motor_token"] = signal_df.apply(
            lambda x: x["dwell_token"][(x["start_pos"] + dwell_shift + trim) : (x["end_pos"] + dwell_shift - trim)],
            axis=1,
        )
        signal_df["dwell_pore_token"] = signal_df.apply(
            lambda x: x["dwell_token"][(x["start_pos"] + trim) : (x["end_pos"] - trim)], axis=1
        )
        signal_df["bq"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))

        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[
            (signal_df["segment_len_arr"].apply(lambda x: len(x) == cb_len))
            & (signal_df["token_len"] <= max_token_len)
            & (signal_df["token_len"] > 0)
        ]
        try:
            signal_df["signal"] = signal_df.apply(
                lambda x: segmented_signal_to_block(
                    x["signal"], x["segment_len_arr"], kmer, sampling, sig_window, max_token_len
                ),
                axis=1,
            )
        except Exception as e:
            log.warning(f"{e}")
            log.warning(f"Signal Tokenization Error in: {signal_path} - Skipping")
            continue

        signal_df = signal_df[signal_df["signal"].notnull()]

        if len(signal_df) == 0:
            continue

        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])

        signal_df = signal_df[
            [
                "block_score",
                "segment_len_arr",
                "signal",
                "motif",
                "dwell_motor_token",
                "dwell_pore_token",
                "bq",
            ]
        ].copy()
        signal_df.rename(
            columns={"motif": "kmer_token", "signal": "signal_token", "bq": "bq_token"},
            inplace=True,
        )
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x.astype(np.uint8))
        signal_df["signal_token"] = signal_df["signal_token"].apply(lambda x: x.astype(np.float32))
        signal_df["kmer_token"] = signal_df["kmer_token"].apply(
            lambda x: np.array(list(x)).view(np.int32).astype(np.uint8)
        )
        signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: np.clip(x, 0, 60).astype(np.uint8))

        for chunk_idx in range(0, len(signal_df) // chunk_size + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk_size : min((chunk_idx + 1) * chunk_size, len(signal_df))].copy()
            save_path = f"{out_path.replace('.pkl','')}-{chunk_idx}.npz"
            save_npz(save_path, chunk_df)

        del signal_df, chunk_df
        gc.collect()

    return None


def save_npz(save_path, df):
    """
    Saves the dataframe to a compressed NPZ file.

    Args:
        save_path (str): Path to save the NPZ file.
        df (pandas.DataFrame): Dataframe containing the data to be saved.

    Returns:
        None
    """
    if len(df) > 0:
        segment_len_arr = np.stack(df["segment_len_arr"].values)
        signal_token = np.stack(df["signal_token"].values)
        kmer_token = np.stack(df["kmer_token"].values)
        dwell_motor_token = np.stack(df["dwell_motor_token"].values)
        dwell_pore_token = np.stack(df["dwell_pore_token"].values)
        bq_token = np.stack(df["bq_token"].values)
        block_score = df["block_score"].values
        np.savez_compressed(
            save_path,
            segment_len_arr=segment_len_arr,
            signal_token=signal_token,
            kmer_token=kmer_token,
            dwell_motor_token=dwell_motor_token,
            dwell_pore_token=dwell_pore_token,
            bq_token=bq_token,
            block_score=block_score,
        )

    return None


def assign_block_id(block_df):
    """
    Assigns block IDs to the dataframe.

    Args:
        block_df (pandas.DataFrame): Dataframe containing block data.

    Returns:
        pandas.DataFrame: Dataframe with assigned block IDs.
    """
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


def split_block_df(signal_path_dict, signal_path_arr, intermediate_path, block_df):
    """
    Splits the block dataframe into smaller dataframes based on signal paths.

    Args:
        signal_path_dict (dict): Dictionary mapping read IDs to signal paths.
        signal_path_arr (list): List of signal paths.
        intermediate_path (str): Path to save intermediate files.
        block_df (pandas.DataFrame): Dataframe containing block data.

    Returns:
        None
    """
    log.info("Reading Block Dataframe. It may take a while.")
    block_df = assign_block_id(block_df)
    block_df["signal_path"] = block_df["read_id"].map(signal_path_dict)

    ## Groupby read_id and make dict
    block_df_groupby = block_df.groupby("signal_path")

    del block_df
    gc.collect()

    for signal_path, group_df in tqdm.tqdm(
        block_df_groupby, total=len(signal_path_arr), desc="Splitting Block Dataframe"
    ):
        group_df.to_pickle(f"{intermediate_path}/block_df_split/{signal_path}")

    del block_df_groupby
    gc.collect()

    return None


def get_norm_factor():
    """
    Returns the default normalization factors.

    Returns:
        dict: Dictionary containing default normalization factors.
    """
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59

    return norm_factor_default
