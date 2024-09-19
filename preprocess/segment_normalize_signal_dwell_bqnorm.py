import argparse
import gc
import glob
import multiprocessing as mp
import os
import pickle
import sys
import numpy as np
import pandas as pd
import pod5
import pysam
import atexit
import tqdm
from utils.utils import oom_killer, printmessage
import toml

## Warning: This script has heavy parallel I/O operations and large memory usage.
## > 1TB Read / Write operations and > 100 GB RAM usage is expected (> 4GB/s disk write was observed).
## Running this on NFS may cause significant performance degradation.


def sequence_to_kmer_token(seq, kmer):
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i:kmer+i] for i in range(len(seq)-kmer+1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4**np.arange(kmer).reshape(-1,1)
    seq = np.sum(seq * quaternary, axis=0) + 1 ## 0 is reserved for padding
    seq = seq.astype(np.int16)
    return seq


def create_segment_len_arr(segment_arr, sampling):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int) // sampling
    return segment_len_arr


def expand_token_to_segment(token_arr, segment_len_arr):
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_move_token(segment_len_arr):
    token = np.arange(1, len(segment_len_arr)+1, dtype=np.uint8)
    token = np.repeat(token, segment_len_arr)
    return token


def create_target_mask(segment_len_arr, lr_pad):
    binary_mask = np.zeros(2*lr_pad+1, dtype=np.uint8)
    binary_mask[lr_pad] = 1
    binary_mask = np.repeat(binary_mask, segment_len_arr)
    return binary_mask


def extract_move(bam_path, ncpu, signal_path_dict, signal_path_arr, move_df_path, low_quantile = 0.2, high_quantile = 0.8):
    ## Extract mv tag from bam and save to separate file
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "bq_low": [], "bq_high": [],} for x in signal_path_arr}
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
                except:
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

                bq = read.query_qualities
                bq_low = np.quantile(bq, low_quantile)
                bq_high = np.quantile(bq, high_quantile)

                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["mv"].append(mv)
                data["sp"].append(sp)
                data["bq_low"].append(bq_low)
                data["bq_high"].append(bq_high)
                count += 1

                pbar.update(1)

    printmessage(f"Valid read count: {count}", msg_type="info")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{move_df_path}/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None

def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    try:
        kmer_pad = (kmer-1)//2
        lr_pad = (sig_window-1)//2
        l_skip = (np.sum(segment_len_arr[:kmer_pad])-lr_pad)*sampling
        r_skip = (np.sum(segment_len_arr[-kmer_pad:])-lr_pad)*sampling
        assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
        assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]
        # signal_segmented = np.lib.stride_tricks.sliding_window_view(signal_segmented, sig_window * sampling)[::sampling]

        padding = (pad_to+kmer-1) * sampling - len(signal_segmented)
        if padding > 0:
            signal_segmented = np.pad(signal_segmented, (0, padding), mode="constant", constant_values=0)

    except:
        return None
    return signal_segmented


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult):
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


def trim_scale_segment_signal(signal,move,sp,ts,ns, quantile_a, quantile_b, shift_mult, scale_mult):
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


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, kmer = 5, cb_len = 21, sampling = 6,
                             sig_window = 5, max_penalty = 10, chunk_size = 1000, max_token_len = 200, dwell_shift = 10):

    trim = kmer//2

    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]
    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]

    for signal_path in tqdm.tqdm(signal_path_arr, total=len(signal_path_arr), desc="Segmenting and Tokenizing Signals"):
        oom_killer()
        file_id = signal_path.split('/')[-1]

        if not os.path.exists(signal_path):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/move_df_split_bqnorm/{file_id}"):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/block_df_split/{file_id}"):
            continue

        out_path = f"{seg_df_path}/token_dwell_bqnorm/{file_id}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(signal_path)
        move_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split_bqnorm/{signal_path.split('/')[-1]}")
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["dwell_token"] = signal_df["mv"].apply(lambda x: move_to_dwell(x, 0.2, 0.8, 0.5, 1.5))
        signal_df["signal"] = signal_df.apply(lambda x: trim_scale_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                         quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        signal_df = signal_df[signal_df["signal"].notnull()][["read_id", "signal", "dwell_token", "bq_low", "bq_high"]].copy()

        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        block_df = block_df[block_df["penalty"]==0]
        if len(block_df) == 0:
            continue

        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()

        signal_df["signal_length"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["end_pos"] + dwell_shift - trim < signal_df["signal_length"]]

        if len(signal_df) == 0:
            continue

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["dwell_motor_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"]+dwell_shift+trim):(x["end_pos"]+dwell_shift-trim)], axis=1)
        signal_df["dwell_pore_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"]+trim):(x["end_pos"]-trim)], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: normalise_bq(x["bq"], x["bq_low"], x["bq_high"], trim, 0.5, 1.5), axis=1)
        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))

        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x)==cb_len)) &
                              (signal_df["token_len"] <= max_token_len) &
                              (signal_df["token_len"] > 0)]
        try:
            signal_df["signal"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"],
                                                                                      kmer, sampling, sig_window, max_token_len), axis=1)
        except:
            print(f"Signal Tokenization Error in: {signal_path} - Skipping")
            continue

        signal_df = signal_df[signal_df["signal"].notnull()]

        if len(signal_df) == 0:
            continue

        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])

        signal_df = signal_df[["segment_len_arr", "signal", "motif", "dwell_motor_token", "dwell_pore_token", "bq"]].copy()
        signal_df.rename(columns={"motif": "kmer_token", "signal": "signal_token", "bq": "bq_token"}, inplace=True)
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x.astype(np.uint8))
        signal_df["signal_token"] = signal_df["signal_token"].apply(lambda x: x.astype(np.float32))
        signal_df["kmer_token"] = signal_df["kmer_token"].apply(lambda x: np.array(list(x)).view(np.int32).astype(np.uint8))
        signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: np.clip(x,0,60).astype(np.uint8))

        for chunk_idx in range(0, len(signal_df) // chunk_size + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk_size:min((chunk_idx + 1) * chunk_size, len(signal_df))].copy()
            save_path = f"{out_path.replace('.pkl','')}-{chunk_idx}.npz"
            save_npz(save_path, chunk_df)

        del signal_df, chunk_df
        gc.collect()

    return None


def normalise_bq(bq, quantile_a, quantile_b, trim, shift_mult, scale_mult):
    bq = bq[trim:-trim]
    q_shift = shift_mult * (quantile_a + quantile_b)
    q_scale = max(1.0, scale_mult * (quantile_b - quantile_a))
    bq = (bq - q_shift) / q_scale
    return bq


def save_npz(save_path, df):

    if len(df) > 0:
        segment_len_arr = np.stack(df["segment_len_arr"].values)
        signal_token = np.stack(df["signal_token"].values)
        kmer_token = np.stack(df["kmer_token"].values)
        dwell_motor_token = np.stack(df["dwell_motor_token"].values)
        dwell_pore_token = np.stack(df["dwell_pore_token"].values)
        bq_token = np.stack(df["bq_token"].values)
        np.savez_compressed(save_path,
                            segment_len_arr=segment_len_arr,
                            signal_token=signal_token,
                            kmer_token=kmer_token,
                            dwell_motor_token=dwell_motor_token,
                            dwell_pore_token=dwell_pore_token,
                            bq_token=bq_token)

    return None



def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--toml", "-t", type=str, default=None, help="Dorado Model TOML file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory", nargs="+")
    args = parser.parse_args()
    return args


def assign_block_id(block_df):
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



def split_block_df(args, signal_path_dict, signal_path_arr, intermediate_path):

    printmessage("Reading Block Dataframe. It may take a while.", msg_type="info")
    block_df = pd.read_pickle(args.block)
    block_df = assign_block_id(block_df)
    block_df["signal_path"] = block_df["read_id"].map(signal_path_dict)

    ## Groupby read_id and make dict
    block_df_groupby = block_df.groupby("signal_path")

    del block_df
    gc.collect()

    for signal_path, group_df in tqdm.tqdm(block_df_groupby, total = len(signal_path_arr), desc="Splitting Block Dataframe"):
        group_df.to_pickle(f"{intermediate_path}/block_df_split/{signal_path}")

    del block_df_groupby
    gc.collect()

    return None


def parse_toml(toml_path):
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59

    if toml_path is None:
        printmessage("TOML file not provided. Using default values for standardisation.", msg_type="warning")
        return norm_factor_default


    if not os.path.exists(toml_path):
        printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
        printmessage("Using default values for standardisation.", msg_type="warning")
        return norm_factor_default

    toml_dict = toml.load(toml_path)
    if "normalisation" not in toml_dict:
        printmessage("normalisation section not found in the TOML file. Check Dorado model version.", msg_type="error", error=ValueError)
        printmessage("Using default values for standardisation.", msg_type="warning")
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

    return norm_factor


def main():
    args = parse_args()
    norm_factor = parse_toml(args.toml)

    for output in args.output:
        if output[-1] == "/":
            output = output[:-1]

        token_output_path = f"{output}/token_dwell_bqnorm/"
        intermediate_path = f"{output}/intermediates/"
        signal_index_path = f"{intermediate_path}/signal_index.pkl"
        move_df_path = f"{intermediate_path}/move_df_split_bqnorm"
        os.makedirs(move_df_path, exist_ok=True)

        os.makedirs(token_output_path, exist_ok=True)
        with open(signal_index_path, "rb") as infile:
            index_dict = pickle.load(infile)
        signal_path_arr = list(index_dict.keys())
        signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]

        signal_path_dict = {}
        for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
            for read_id in id_list:
                signal_path_dict[read_id] = signal_path.split('/')[-1]

        del index_dict
        gc.collect()

        np.random.shuffle(signal_path_arr)
        signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

        bam_path = os.path.join(os.path.dirname(output), "dorado_output.bam")

        if len(os.listdir(move_df_path)) == 0:
            extract_move(bam_path, args.cpu, signal_path_dict, signal_name_arr, move_df_path)

        proc_list = []
        for signal_paths in signal_path_arr_split:
            proc = mp.Process(target=segment_normalize_signal,
                              args=(output, signal_paths, norm_factor))
            proc_list.append(proc)
            proc.start()

        del signal_path_arr_split
        gc.collect()

        for proc in proc_list:
            proc.join()

        printmessage("Signal Segmentation and Tokenization Complete", msg_type="success")
        printmessage("Saved to: " + output, msg_type="success")

    return None



if __name__ == "__main__":
    main()
