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


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, kmer = 5, cb_len = 21, sampling = 6,
                             sig_window = 5, max_penalty = 10, chunk_size = 1000000, max_token_len = 200):

    cb_lr_pad = (cb_len-kmer)//2
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

        out_path = f"{seg_df_path}/signal_analysis/temp/{file_id}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(signal_path)
        move_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}")
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["signal"] = signal_df.apply(lambda x: trim_scale_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                  quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        signal_df = signal_df[signal_df["signal"].notnull()][["read_id", "signal"]].copy()

        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        block_df = block_df[block_df["penalty"] == 0]

        if len(block_df) == 0:
            continue

        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]-3:x["end_pos"]+3], axis=1)

        signal_df = signal_df[signal_df["signal"].apply(lambda x: len(x) == cb_len + 6)]

        if len(signal_df) == 0:
            continue

        signal_df = signal_df[["signal", "bq", "motif", "cb_idx"]].copy()

        gc.collect()

        signal_df["signal_mean"] = signal_df["signal"].apply(lambda x: np.array([np.mean(y) for y in x]))
        signal_df["signal_median"] = signal_df["signal"].apply(lambda x: np.array([np.median(y) for y in x]))
        signal_df["signal_std"] = signal_df["signal"].apply(lambda x: np.array([np.std(y) for y in x]))
        signal_df["signal_len"] = signal_df["signal"].apply(lambda x: np.array([len(y) for y in x]))
        signal_df["signal_amp"] = signal_df["signal"].apply(lambda x: np.array([np.max(y) - np.min(y) for y in x]))
        signal_df["signal_rms"] = signal_df["signal"].apply(lambda x: np.array([np.sqrt(np.mean(np.square(y))) for y in x]))

        signal_df.drop("signal", axis=1, inplace=True)

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
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-n", type=int, default=500, help="POD5 Chunk size")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5  dataframe size in MB")
    parser.add_argument("--keep_intermediate", "-ki", action="store_true", help="Keep intermediate files")
    parser.add_argument("--skip_intermediate", "-sk", action="store_true", help="Skip intermediate files")
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
        printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
        printmessage("Using default values for standardisation.", msg_type="warning")
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


    token_output_path = f"{args.output}/signal_analysis/"
    temp_path = f"{token_output_path}/temp/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"


    norm_factor = parse_toml(args.toml)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    os.makedirs(f"{intermediate_path}/block_df_split", exist_ok=True)

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
                          args=(args.output, signal_paths, norm_factor))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    printmessage("Signal Segmentation and Tokenization Complete", msg_type="success")
    printmessage("Saved to: " + args.output, msg_type="success")
    orig_df = read_path(token_output_path)

    result_df_list = []

    for cb_idx in range(3):

        df = orig_df[orig_df["cb_idx"] == cb_idx]
        df["5mer"] = df["motif"].str[8:13]
        num_groups = len(df["5mer"].unique())
        print("Number of 5mers:", num_groups)
        df = df.groupby(["5mer"])
        ## sample 800 for each 5mer
        df_list = []
        group_lengths = []
        for name, group in df:
            group_lengths.append(len(group))

        for name, group in tqdm.tqdm(df, total = num_groups):
            df_list.append(group.sample(800))
        df = pd.concat(df_list)
        df.to_pickle(f"{token_output_path}/sampled_CB{cb_idx}.pkl")

        df = pd.read_pickle(f"{token_output_path}/sampled_CB{cb_idx}.pkl")
        df["signal_len_log10"] = df["signal_len"].apply(lambda x: np.log10(np.float32(x)))
        print(df)
        result_df_list.append(df)

        result_dict = {}
        column_list = ["signal_mean","signal_median","signal_std","signal_len_log10","signal_amp","signal_rms","bq"]
        for column in column_list:
            values = np.stack(df[column].values, axis = 0)
            mean = np.mean(values, axis = 0)
            std = np.std(values, axis = 0)
            ci95 = 1.96 * std / np.sqrt(len(df))
            result_dict[column] = {"mean":mean, "std":std, "ci95":ci95, "len":len(df)}

        result_df = pd.DataFrame(result_dict)
        result_df.to_pickle(f"{token_output_path}/result_CB{cb_idx}.pkl")


    df = pd.concat(result_df_list, axis = 0).reset_index()
    print(df)

    del result_df_list
    result_dict = {}
    column_list = ["signal_mean","signal_median","signal_std","signal_len_log10","signal_amp","signal_rms","bq"]
    for column in column_list:
        values = np.stack(df[column].values, axis = 0)
        mean = np.mean(values, axis = 0)
        std = np.std(values, axis = 0)
        ci95 = 1.96 * std / np.sqrt(len(df))
        result_dict[column] = {"mean":mean, "std":std, "ci95":ci95, "len":len(df)}

    result_df = pd.DataFrame(result_dict)
    result_df.to_pickle(f"{token_output_path}/result.pkl")

    return None


def read_path(path, ncpu = 120):
    path_list = glob.glob(f"{path}/temp/*.pkl")
    if "ON0092" in path or "ON0096" in path:
        path_list = np.random.choice(path_list, 10000, replace = False)
    man = mp.Manager()
    return_list = man.list()
    path_list = np.array_split(path_list, ncpu)
    proc_list = []
    for path in path_list:
        proc = mp.Process(target = read_file, args = (path, return_list))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_list = list(return_list)
    df = pd.concat(return_list)
    del return_list
    gc.collect()
    return df

def read_file(path_list, return_list):
    df_list = []
    for path in tqdm.tqdm(path_list):
        df = pd.read_pickle(path)
        df_list.append(df)
    df_list = pd.concat(df_list)
    return_list.append(df_list)
    return None


if __name__ == "__main__":
    main()
