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
from sklearn import preprocessing

## Warning: This script has heavy parallel I/O operations and large memory usage.
## > 1TB Read / Write operations and > 100 GB RAM usage is expected (> 4GB/s disk write was observed).
## Running this on NFS may cause significant performance degradation.


def extract_move(bam_path, ncpu, signal_path_dict, signal_path_arr, intermediate_path):
    ## Extract mv tag from bam and save to separate file
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

                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["mv"].append(mv)
                data["sp"].append(sp)
                count += 1

                pbar.update(1)

    printmessage(f"Valid read count: {count}", msg_type="info")

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
    # Export pod5 to csv
    pod5_path_list = glob.glob(pod5_path + "/*.pod5")
    proc_list = []
    np.random.shuffle(pod5_path_list)
    pod5_path_list_split = np.array_split(pod5_path_list, ncpu)

    man = mp.Manager()
    index_list = man.list()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_signal_proc, args=(pod5_path_list_split[pid], save_path, pid, index_list,
                                                            chunk, max_mb, min_mb))
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
    index_dict_local = {}
    chunk_buffer = []
    pod5_idx = 0

    for pod5_idx, pod5_path in tqdm.tqdm(enumerate(pod5_path_list), total=len(pod5_path_list), desc=f"Parsing POD5 Files"):
        oom_killer()
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
                    except:
                        skipped += 1
                        continue

                    offset_list.append(offset)
                    scale_list.append(scale)
                    signal_list.append(signal_arr)
                    id_list.append(id)

            if skipped > 0:
                printmessage(f"Skipped {skipped} faulty records in: {pod5_path}", msg_type="warning")

        except:
            ## Pod5 file is corrupted
            printmessage(f"Corrupted POD5 file: {pod5_path} - Skipping", msg_type="warning")
            continue

        df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
        del signal_list, offset_list, scale_list, id_list
        gc.collect()

        save_idx = 0

        for chunk_idx in range(0, len(df) // chunk + 1):
            signal_df = df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(df))].copy()
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if len(signal_df) == chunk and df_size > min_mb:
                save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer.append(signal_df)
            gc.collect()

        if len(chunk_buffer) > 0:
            signal_df = pd.concat(chunk_buffer, ignore_index=True)
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
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
    save_idx += 1
    df_size = sys.getsizeof(signal_df) / (1024 ** 2)
    if df_size > max_mb and len(signal_df) > 1:
        ## Split the dataframe
        split_num = min(int(df_size // max_mb) + 1 ,len(signal_df))
        split_size = len(signal_df) // split_num
        for split_idx in range(split_num):
            split_df = signal_df.iloc[split_idx * split_size:min((split_idx + 1) * split_size, len(signal_df))].copy()
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


def trim_signal(signal,sp,ts,ns):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)
    signal = signal.astype(np.float32)
    return signal

def get_move_idx(signal, move):
    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    return move_idx

def iqrnorm(signal, quantile_a, quantile_b, shift_mult, scale_mult):
    quantile_a_value = np.quantile(signal, quantile_a)
    quantile_b_value = np.quantile(signal, quantile_b)
    q_shift = shift_mult * (quantile_a_value + quantile_b_value)
    q_scale = scale_mult * (quantile_b_value - quantile_a_value)
    signal = (signal - q_shift) / q_scale
    return signal

def znorm(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal

def minmaxnorm(signal):
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return signal

def meannorm(signal):
    signal = (signal - np.mean(signal))/ (np.max(signal) - np.min(signal))
    return signal

def maxabsnorm(signal):
    signal = signal / np.max(np.abs(signal))
    return signal

def madnorm(signal):
    signal = (signal - np.median(signal)) / np.median(np.abs(signal - np.median(signal)))
    return signal

def sigmoidnorm(signal):
    signal = 1 / (1 + np.exp((-znorm(signal))))
    return signal

def tanhnorm(signal):
    signal = 0.5 * (np.tanh(0.01 * znorm(signal))+1)
    return signal

def quantile_transform_normal(signal):
    signal = signal * 0.001
    signal = preprocessing.QuantileTransformer(n_quantiles=min(len(signal), 1000),
        output_distribution='normal').fit_transform(signal.reshape(-1, 1)).flatten()
    return signal

def quantile_transform_uniform(signal):
    signal = signal * 0.001
    signal = preprocessing.QuantileTransformer(n_quantiles=min(len(signal), 1000),
        output_distribution='uniform').fit_transform(signal.reshape(-1, 1)).flatten()
    return signal

def power_transform(signal):
    signal = signal * 0.001
    signal = preprocessing.PowerTransformer().fit_transform(signal.reshape(-1, 1)).flatten()
    return signal




def segment_normalize_signal(seg_df_path, postfix, signal_path_arr, norm_factor, kmer = 5, cb_len = 21, sampling = 6,
                             sig_window = 5, max_penalty = 10, chunk_size = 1000, max_token_len = 200, dwell_shift = 10):

    test_key_list = ["signal", "signal_iqr", "signal_z", "signal_minmax", "signal_mean", "signal_maxabs", "signal_mad",
                     "signal_sigmoid", "signal_tanh", "signal_qt_normal", "signal_qt_uniform", "signal_power"]

    final_test_key_list = ["motif"] + [x + "_centre" for x in test_key_list] + [x + "_avg" for x in test_key_list]

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
        signal_df["signal"] = signal_df.apply(lambda x: trim_signal(x["signal"], x["sp"], x["ts"], x["ns"]), axis=1)
        signal_df = signal_df[signal_df["signal"].notnull()][["read_id", "signal", "mv"]].copy()
        signal_df["mv"] = signal_df.apply(lambda x: get_move_idx(x["signal"], x["mv"]), axis=1)

        signal_df["signal_iqr"] = signal_df["signal"].apply(lambda x: iqrnorm(x, quantile_a, quantile_b, shift_mult, scale_mult))
        signal_df["signal_z"] = signal_df["signal"].apply(lambda x: znorm(x))
        signal_df["signal_minmax"] = signal_df["signal"].apply(lambda x: minmaxnorm(x))
        signal_df["signal_mean"] = signal_df["signal"].apply(lambda x: meannorm(x))
        signal_df["signal_maxabs"] = signal_df["signal"].apply(lambda x: maxabsnorm(x))
        signal_df["signal_mad"] = signal_df["signal"].apply(lambda x: madnorm(x))
        signal_df["signal_sigmoid"] = signal_df["signal"].apply(lambda x: sigmoidnorm(x))
        signal_df["signal_tanh"] = signal_df["signal"].apply(lambda x: tanhnorm(x))
        signal_df["signal_qt_normal"] = signal_df["signal"].apply(lambda x: quantile_transform_normal(x))
        signal_df["signal_qt_uniform"] = signal_df["signal"].apply(lambda x: quantile_transform_uniform(x))
        signal_df["signal_power"] = signal_df["signal"].apply(lambda x: power_transform(x))


        for k in test_key_list:
            signal_df[k] = signal_df.apply(lambda x: np.array_split(x[k], x["mv"]), axis=1)

        signal_df.drop(columns=["mv"], inplace=True)

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

        for k in test_key_list:
            signal_df[k] = signal_df.apply(lambda x: x[k][x["start_pos"]:x["end_pos"]], axis=1)
            signal_df[k+"_centre"] = signal_df[k].apply(lambda x: np.mean(x[10]))
            signal_df[k+"_avg"] = signal_df[k].apply(lambda x: np.mean(np.concatenate(x)))


        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))
        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x)==cb_len)) &
                              (signal_df["token_len"] <= max_token_len) &
                              (signal_df["token_len"] > 0) &
                              (signal_df["signal"].notnull())][final_test_key_list]

        if len(signal_df) == 0:
            continue

        for chunk_idx in range(0, len(signal_df) // chunk_size + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk_size:min((chunk_idx + 1) * chunk_size, len(signal_df))].copy()
            save_path = f"{out_path.replace('.pkl','')}-{chunk_idx}.pkl"
            chunk_df.to_pickle(save_path)

        del signal_df, chunk_df
        gc.collect()

    return None


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, default=None, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, default=None, help="Dorado BAM file")
    parser.add_argument("--toml", "-t", type=str, default=None, help="Dorado Model TOML file")
    parser.add_argument("--block", "-k", type=str, default=None, help="Block dataframe path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--chunk", "-n", type=int, default=500, help="POD5 Chunk size")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5  dataframe size in MB")
    parser.add_argument("--keep_intermediate", "-ki", action="store_true", help="Keep intermediate files", default=True)
    parser.add_argument("--skip_intermediate", "-si", action="store_true", help="Skip intermediate files")
    parser.add_argument("--postfix", "-x", type=str, default="normalization_test_111724", help="Output file postfix")
    args = parser.parse_args()
    # if not os.path.exists(args.pod5):
    #     raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    # if not os.path.exists(args.bam):
    #     raise FileNotFoundError(f"BAM file {args.bam} does not exist")
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

    if not args.skip_intermediate:

        index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk, args.max_size, args.min_size)
        signal_path_arr = list(index_dict.keys())
        signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
        gc.collect()

        if len(signal_path_arr) == 0:
            printmessage("No valid signal files found. Exiting.", msg_type="error")
            return None

        with open(signal_index_path, "wb") as outfile:
            pickle.dump(index_dict, outfile)

        signal_path_dict = {}
        for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
            for read_id in id_list:
                signal_path_dict[read_id] = signal_path.split('/')[-1]

        del index_dict
        gc.collect()

        split_block_df(args, signal_path_dict, signal_name_arr, intermediate_path)
        extract_move(args.bam, args.cpu, signal_path_dict, signal_name_arr, intermediate_path)

        del signal_path_dict, signal_name_arr
        gc.collect()


    else:
        ## load
        with open(signal_index_path, "rb") as infile:
            index_dict = pickle.load(infile)
        signal_path_arr = list(index_dict.keys())
        del index_dict
        gc.collect()

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    for signal_paths in signal_path_arr_split:
        proc = mp.Process(target=segment_normalize_signal,
                          args=(args.output, args.postfix, signal_paths, norm_factor))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    printmessage("Signal Segmentation and Tokenization Complete", msg_type="success")
    printmessage("Saved to: " + args.output, msg_type="success")
    return None



if __name__ == "__main__":
    main()
