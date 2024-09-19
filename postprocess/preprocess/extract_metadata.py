import argparse
import gc
import toml
import multiprocessing as mp
import os
import pickle
import re
import numpy as np
import pandas as pd
import tqdm
import pysam
from utils.utils import mean_phred, oom_killer, printmessage
from archived.train_eval.tokenize_transcript import preprocess_pod5

def md_to_mismatch_arr(md):
    """
    Convert MD tag to mismatch array.

    :param md: An array. MD tag from BAM file.

    :return A numpy array with shape (len(read),) with 1 for mismatch and 0 for match.
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


def parse_bam(bam_path, threads, bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path):
    """
    Parse BAM File and saves the pickled Pandas Dataframes to disk.
    Dataframes contain SAM tags (including MOVE), Sequence, Mapping, and BQ.

    :param bam_path: A string. Path to BAM file.
    :param threads: An integer. Number of threads.
    :param bq_cutoff: An integer. Minimum BQ score.
    :param signal_path_dict: A dictionary. Read ID to Signal Path mapping.
    :param signal_path_arr: A list. Signal Path list.
    :param intermediate_path: A string. Path to save the output.

    :return: None.
    Pickled Pandas Dataframes are written to disk.
    Each Dataframe is a part of BAM that corresponds to each POD5 file.
    """
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [],
                     "mapq": [], "flag": [], "md": [], "pi": [], "ref": [], "start": [], "cigar": []} for x in signal_path_arr}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=threads) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                pbar.update(1)
                pbar.set_postfix({"valid": valid_count, "invalid": missing_move + missing_bq + low_bq + missing_signal + unmapped})
                if read.is_unmapped:
                    unmapped += 1
                    continue

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                    pi = 1
                else:
                    read_id = str(read.query_name)
                    pi = 0

                try:
                    bq = np.array(read.query_qualities, dtype=int)
                    if mean_phred(bq) < bq_cutoff:
                        low_bq += 1
                        continue
                except:
                    missing_bq += 1
                    continue

                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except:
                    missing_signal += 1
                    continue

                if read.has_tag("mv"):
                    mv = read.get_tag("mv")
                else:
                    missing_move += 1
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

                if read.has_tag("MD"):
                    md = read.get_tag("MD")
                else:
                    md = ""

                data["mv"].append(mv)
                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["sp"].append(sp)
                data["seq"].append(str(read.query_sequence))
                data["bq"].append(bq)
                data["ref"].append(read.reference_name)
                data["start"].append(read.reference_start)
                data["cigar"].append(read.cigarstring)
                data["md"].append(md)
                data["pi"].append(pi)
                data["mapq"].append(read.mapping_quality)
                data["flag"].append(read.flag)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="warning")
    printmessage(f"Missing BQ: {missing_bq}", msg_type="warning")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="warning")
    printmessage(f"Missing Move: {missing_move}", msg_type="warning")
    printmessage(f"Unmapped: {unmapped}", msg_type="warning")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/meta_df_split/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None


def standardise_trim_segment_signal(signal,move,sp,ts,ns,offset,scale,mean,stdev):
    """
    Standardise and Trim the signal.
    :param signal:
    :param move:
    :param sp:
    :param ts:
    :param ns:
    :param offset:
    :param scale:
    :param mean:
    :param stdev:
    :return:
    """
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)
    signal = (signal + offset) * scale
    signal = (signal - mean) / stdev

    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def normalise_trim_segment_signal(signal,move,sp,ts,ns, quantile_a, quantile_b, shift_mult, scale_mult):
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

    q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value))
    signal = (signal - q_shift) / q_scale

    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def get_left_soft_clip(cigar):
    if "S" in cigar:
        left_soft_clip = cigar.split("S")[0]
        if left_soft_clip.isdigit():
            left_soft_clip = int(left_soft_clip)
        else:
            left_soft_clip = 0

    else:
        left_soft_clip = 0
    return left_soft_clip


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, label_df, pid, norm_mode, token_output_path,
                             cb_len = 21, kmer_len = 5, chunk_size = 10000, max_token_len = 200, sampling = 6, boi = "A"):


    trim = kmer_len//2
    mean, stdev, quantile_a, quantile_b, shift_mult, scale_mult = None, None, None, None, None, None

    if norm_mode == "normalise":
        shift_mult = norm_factor["shift_mult"]
        scale_mult = norm_factor["scale_mult"]
        quantile_a = norm_factor["quantile_a"]
        quantile_b = norm_factor["quantile_b"]
    elif norm_mode == "standardise":
        mean = norm_factor["mean"]
        stdev = norm_factor["stdev"]
    else:
        raise ValueError(f"Invalid norm_mode: {norm_mode}")


    cb_half_len = cb_len//2
    buffer = []

    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        out_path = f"{token_output_path}/{signal_path.split('/')[-1]}"

        if os.path.exists(out_path):
            printmessage(f"Already exists: {out_path}", msg_type="warning")
            continue
        move_path = f"{seg_df_path}/intermediates/meta_df_split/{signal_path.split('/')[-1]}"
        if not os.path.exists(move_path):
            printmessage(f"Move data not found: {move_path}", msg_type="warning")
            continue
        signal_df = pd.read_pickle(signal_path)
        if len(signal_df) == 0:
            printmessage(f"Empty signal data: {signal_path}", msg_type="warning")
            continue

        move_df = pd.read_pickle(move_path)
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df
        gc.collect()

        signal_df["query_len"] = signal_df["seq"].apply(len)

        signal_df[['ref_pos', 'query_pos', 'error', 'label', "dom"]] = signal_df.apply(lambda x: get_label_pos_list(
            x["ref"], x["start"], x["cigar"], x["query_len"], x["md"], label_df), axis=1, result_type="expand")

        signal_df = signal_df[signal_df["query_pos"].notnull()].copy()
        signal_df["left_soft_clip"] = signal_df["cigar"].apply(get_left_soft_clip)

        if len(signal_df) == 0:
            continue

        gc.collect()

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))

        if norm_mode == "normalise":
            signal_df["signal"] = signal_df.apply(lambda x: normalise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                          quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        elif norm_mode == "standardise":
            signal_df["signal"] = signal_df.apply(lambda x: standardise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                            x["offset"], x["scale"], mean, stdev), axis=1)

        signal_df = signal_df[["read_id", "ref", "bq", "seq", "ref_pos", "query_pos", "error", "label", "dom",
                               "mapq", "flag", "pi", "left_soft_clip", "signal"]].copy()
        signal_df["read_bq"] = signal_df["bq"].apply(mean_phred)
        gc.collect()

        signal_df = signal_df.explode(["ref_pos", "query_pos", "error", "label", "dom"], ignore_index=True)

        if len(signal_df) == 0:
            continue

        signal_df["centre_nuc"] = signal_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)
        signal_df = signal_df[signal_df["centre_nuc"] == boi]

        if len(signal_df) == 0:
            continue

        signal_df["label_id"] = signal_df["ref"].astype(str) + ":" + signal_df["ref_pos"].astype(str)
        signal_df["block_id"] = signal_df["read_id"] + ":" + signal_df["query_pos"].astype(str)
        signal_df["start_pos"] = signal_df["query_pos"] - cb_half_len
        signal_df["end_pos"] = signal_df["query_pos"] + cb_half_len + 1
        signal_df["query_len"] = signal_df["seq"].apply(len)

        signal_df = signal_df[(signal_df["start_pos"] >= 0) & (signal_df["end_pos"] <= signal_df["query_len"])]

        if len(signal_df) == 0:
            continue

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["block_bq"] = signal_df["bq"].apply(mean_phred)
        signal_df["base_bq"] = signal_df["bq"].apply(lambda x: x[cb_half_len])


        signal_df = signal_df[["label_id", "block_id", "label", "dom", "signal", "bq", "motif",
                               "mapq", "flag", "pi", "read_bq", "block_bq", "base_bq",
                               "error", "query_pos", "query_len", "left_soft_clip"]].copy()

        gc.collect()

        if len(buffer) > 0:
            signal_df = pd.concat([buffer, signal_df], ignore_index=True)
            buffer = []

        if len(signal_df) < chunk_size:
            buffer = signal_df

        else:
            for chunk_idx in range(0, len(signal_df) // chunk_size):
                chunk = signal_df.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

            chunk = signal_df.iloc[(len(signal_df) // chunk_size) * chunk_size:].copy()
            if len(chunk) > 0:
                buffer = chunk
            del signal_df

        gc.collect()


    out_path = f"{token_output_path}/last-{pid}.pkl"
    ## For last chunk, append zero data and save
    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // chunk_size):
            chunk = buffer.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

        chunk = buffer.iloc[(len(buffer) // chunk_size) * chunk_size:].copy()
        if len(chunk) > 0:
            chunk.to_pickle(f"{out_path.split('.')[0]}-last.pkl")

    return None



def ref_pos_to_query_pos(ref_pos_list, cigar, start_pos, query_len, md_tag, cb_pad = 10):
    cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
    mis_arr = md_to_mismatch_arr(md_tag)
    query_pos_dict = {}
    error_arr = np.zeros((query_len,3),dtype=bool)
    ## Channel 0: Mismatch, Channel 1: Insertion, Channel 2: Deletion

    idx_ref_prev = start_pos
    idx_ref = start_pos

    idx_query_prev = 0
    idx_query = 0

    for length, match in cigar_list:
        length = int(length)
        if match == "M":
            mis_arr_slice = mis_arr[idx_ref-start_pos:idx_ref+length-start_pos]
            error_arr[idx_query:idx_query+length, 0] = mis_arr_slice
            idx_ref += length
            idx_query += length
        elif match == "I":
            error_arr[idx_query:idx_query+length, 1] = 1
            idx_query += length
        elif match == 'D':
            error_arr[idx_query, 2] = 1
            idx_ref += length
        elif match == "S":
            idx_query += length
        elif match in ['H', 'P']:
            pass
        else:
            raise ValueError(f'unknown cigar: {match}')

        if len(query_pos_dict) < len(ref_pos_list):
            for save_idx, ref_pos in enumerate(ref_pos_list):
                if save_idx not in query_pos_dict:
                    if idx_ref > ref_pos:
                        if match == 'M' :
                            query_pos = idx_query_prev + (ref_pos - idx_ref_prev)
                        else:
                            query_pos = -1
                        query_pos_dict[save_idx] = query_pos

        idx_ref_prev = idx_ref
        idx_query_prev = idx_query

    assert len(error_arr) == query_len

    for save_idx in range(len(ref_pos_list)):
        if save_idx not in query_pos_dict:
            query_pos_dict[save_idx] = -1
        elif query_pos_dict[save_idx] != -1:
            if query_pos_dict[save_idx] < cb_pad:
                query_pos_dict[save_idx] = -1
            elif query_pos_dict[save_idx] >= query_len - cb_pad:
                query_pos_dict[save_idx] = -1
            else:
                pass
        else:
            pass

    query_pos_list = np.array([query_pos_dict[x] for x in range(len(ref_pos_list))], dtype=int)
    error_list = np.lib.stride_tricks.sliding_window_view(error_arr, window_shape=2*cb_pad+1, axis=0)

    error_list = error_list[query_pos_list - cb_pad]
    return query_pos_list, error_list


def get_label_pos_list(ref, start, cigar, query_len, md_tag, label_df):
    ref = ref.split(".")[0]

    try:
        label_df = label_df.get_group(ref)
    except KeyError:
        return None, None, None, None, None

    if len(label_df) == 0:
        return None, None, None, None, None

    label_list = label_df["label"].values
    dom_list = label_df["m6A_level"].values
    ref_pos_list = label_df["pos"].values
    query_pos_list, error_list = ref_pos_to_query_pos(ref_pos_list, cigar, start, query_len, md_tag)

    index_to_drop = [idx for idx, x in enumerate(query_pos_list) if x == -1]
    ref_pos_list = np.delete(ref_pos_list, index_to_drop)
    query_pos_list = np.delete(query_pos_list, index_to_drop)
    error_list = np.delete(error_list, index_to_drop, axis=0)
    label_list = np.delete(label_list, index_to_drop)
    dom_list = np.delete(dom_list, index_to_drop)

    if len(query_pos_list) == 0:
        return None, None, None, None, None

    return ref_pos_list, query_pos_list, error_list, label_list, dom_list


def parse_args():
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--wdir", "-w", type=str, default=None, help="Working directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=1000, help="Chunk size")
    parser.add_argument("--pod5_chunk", "-j", type=int, default=1000, help="POD5 Chunk size")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5  dataframe size in MB")
    parser.add_argument("--toml", "-t", type=str, default=None, help="Dorado Model TOML file")
    parser.add_argument("--norm_mode", "-n", type=str, required=True, help="Normalisation mode: normalise or standardise")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output files")
    parser.add_argument("--max_token_len", "-z", type=int, default=200, help="Maximum token length")
    parser.add_argument("--sampling", "-s", type=int, default=6, help="Sampling rate")
    parser.add_argument("--boi", "-y", type=str, default="A", help="Base of interest")
    parser.add_argument("--kmer_len", "-e", type=int, default=5, help="Kmer length")
    parser.add_argument("--cb_len", "-a", type=int, default=21, help="Context block length")
    args = parser.parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if args.wdir is None:
        args.wdir = f"{args.output}/intermediates/"
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wdir, exist_ok=True)
    if args.toml is None:
        if args.norm_mode == "standardise":
            args.toml = "/extdata3/baeklab/Hyeonseo/bin/dorado-0.7.2/model/rna004_130bps_sup@v5.0.0/config.toml"
        elif args.norm_mode == "normalise":
            args.toml = "/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3/model/rna004_130bps_sup@v3.0.1/config.toml"
        else:
            raise ValueError(f"Invalid norm_mode: {args.norm_mode}")

    return args


def parse_toml(toml_path, norm_mode):

    if norm_mode == "normalise":
        norm_factor_default = {}
        norm_factor_default["quantile_a"] = 0.2
        norm_factor_default["quantile_b"] = 0.8
        norm_factor_default["shift_mult"] = 0.48
        norm_factor_default["scale_mult"] = 0.59

        if not os.path.exists(toml_path):
            printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        toml_dict = toml.load(toml_path)
        if "normalisation" not in toml_dict:
            printmessage("normalisation section not found in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for normalisation.", msg_type="warning")
            return norm_factor_default

        printmessage("Normalisation parameters found in TOML file.", msg_type="info")

        std_dict = toml_dict["normalisation"]
        norm_factor = {}
        norm_factor["quantile_a"] = std_dict.get("quantile_a")
        norm_factor["quantile_b"] = std_dict.get("quantile_b")
        norm_factor["shift_mult"] = std_dict.get("shift_multiplier")
        norm_factor["scale_mult"] = std_dict.get("scale_multiplier")

        ## sanitize
        for key in norm_factor.keys():
            if norm_factor[key] is None:
                printmessage(f"Key {key} not found in TOML file. Falling back to default value.", msg_type="warning")
                norm_factor[key] = norm_factor_default[key]

    elif norm_mode == "standardise":

        norm_factor_default = {}
        norm_factor_default["mean"] = 80.8758975922949
        norm_factor_default["stdev"] = 17.26975967138176

        if not os.path.exists(toml_path):
            printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        toml_dict = toml.load(toml_path)
        if "standardisation" not in toml_dict:
            printmessage("standardisation section not found in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        printmessage("Normalisation parameters found in TOML file.", msg_type="info")

        std_dict = toml_dict["standardisation"]

        if not std_dict["standardise"]:
            printmessage("Standardisation is not enabled in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        norm_factor = {}
        norm_factor["mean"] = std_dict.get("mean")
        norm_factor["stdev"] = std_dict.get("stdev")

        ## sanitize
        for key in norm_factor.keys():
            if norm_factor[key] is None:
                printmessage(f"Key {key} not found in TOML file. Falling back to default value.", msg_type="warning")
                norm_factor[key] = norm_factor_default[key]

    else:
        printmessage(f"Mode {norm_mode} not recognised. Should be either 'normalise' or 'standardise'.", msg_type="error")
        raise ValueError()

    return norm_factor


def main():
    args = parse_args()
    token_output_path = f"{args.output}/token_{args.norm_mode}_{args.postfix}/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/meta_df_split", exist_ok=True)

    norm_factor = parse_toml(args.toml, args.norm_mode)

    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk, args.min_size, args.max_size)
    signal_path_arr = list(index_dict.keys())
    gc.collect()

    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)
    gc.collect()

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path.split('/')[-1]
    del index_dict
    gc.collect()

    parse_bam(args.bam, args.cpu, args.qcut, signal_path_dict, intermediate_path)

    del signal_path_dict, index_dict
    gc.collect()

    label_df = pd.read_csv(args.label, sep='\t')
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    label_df = label_df.groupby("nmid")
    for pid, signal_paths in enumerate(signal_path_arr_split):
        proc = mp.Process(target=segment_normalize_signal,
                          args=(args.output, signal_paths, norm_factor, label_df, pid, args.norm_mode, token_output_path,
                                args.cb_len, args.kmer_len, args.chunk, args.max_token_len, args.sampling, args.boi))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    gc.collect()

    return None


if __name__ == "__main__":
    main()
