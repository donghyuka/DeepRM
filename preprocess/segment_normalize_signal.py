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
import scipy
import tqdm
from utils.utils import oom_killer, printmessage


def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_list, chunk, max_mb, min_mb):
    index_dict_local = {}
    chunk_buffer = []
    pod5_idx = 0

    for pod5_idx, pod5_path in tqdm.tqdm(enumerate(pod5_path_list), total=len(pod5_path_list)):
        oom_killer()
        signal_list = []
        offset_list = []
        scale_list = []
        id_list = []
        try:
            with pod5.Reader(pod5_path) as reader:
                for record in reader:
                    signal_arr = record.signal
                    offset = record.calibration.offset
                    scale = record.calibration.scale
                    offset_list.append(offset)
                    scale_list.append(scale)
                    signal_list.append(signal_arr)
                    id_list.append(str(record.read_id))
        except:
            ## Pod5 file is corrupted
            printmessage(f"Corrupted POD5 file: {pod5_path}")
            continue
        gc.collect()
        df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})

        #
        # if len(chunk_buffer) > 0:
        #     chunk_buffer.append(df)
        #     df = pd.concat(chunk_buffer, ignore_index=True)
        #     chunk_buffer = []

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
    data_dict = {x: {"mv": [], "read_id": [], "sm": [], "sd": [], "ts": []} for x in signal_path_arr}

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                if read.has_tag("pi"):
                    continue
                read_id = str(read.query_name)
                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except:
                    continue
                data["read_id"].append(read_id)
                data["mv"].append(read.get_tag("mv"))
                data["sm"].append(read.get_tag("sm"))
                data["sd"].append(read.get_tag("sd"))
                data["ts"].append(read.get_tag("ts"))
                pbar.update(1)

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


def segment_signal(signal, move):
    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal_segmented = np.array_split(signal, move_idx)
    return signal_segmented


def segment_spectrogram(signal, move, filter, sampling = 4000, nperseg = 40, stride = 5):
    signal = scipy.signal.sosfilt(filter, signal)
    f, t, sxx = scipy.signal.spectrogram(signal, fs=sampling, nperseg=nperseg, noverlap=nperseg-stride,
                                         mode="magnitude", window="hann")
    sxx = sxx.T
    stride = move[0]
    move = move[1:]
    move=np.array(move, dtype=int)
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0) // stride
    sxx = np.array_split(sxx, move_idx, axis=0)
    return sxx


def segment_normalize_fft_signal(seg_df_path, signal_path_arr):
    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        if not os.path.exists(signal_path):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}"):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}"):
            continue
        out_path = f"{seg_df_path}/block/{signal_path.split('/')[-1]}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(signal_path)
        move_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}")
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df
        gc.collect()

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["signal_len"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["signal_len"] > signal_df["ts"]]
        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["ts"]:], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: np.flip(x["signal"], axis=0), axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] + x["offset"]) * x["scale"], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] - x["sm"]) / x["sd"], axis=1)

        signal_df = signal_df[["read_id", "signal", "mv"]].copy()
        gc.collect()

        signal_df["signal_seg"] = signal_df.apply(lambda x: segment_signal(x["signal"], x["mv"]), axis=1)
        filter = scipy.signal.butter(4, 100, btype="highpass", fs=4000, output="sos")
        signal_df["signal_fft"] = signal_df.apply(lambda x: segment_spectrogram(x["signal"], x["mv"], filter), axis=1)

        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()

        signal_df["signal_seg"] = signal_df.apply(lambda x: x["signal_seg"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["signal_fft"] = signal_df.apply(lambda x: x["signal_fft"][x["start_pos"]:x["end_pos"]], axis=1)

        signal_df.to_pickle(out_path)
        del signal_df
        gc.collect()

    return None


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--block", "-k", type=str, required=True, help="Block dataframe")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-n", type=int, default=500, help="Chunk size")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum dataframe size in MB")
    args = parser.parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    # if os.path.exists(args.output):
    #     raise FileExistsError(f"Output directory {args.output} already exists")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/block/", exist_ok=True)
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



def main():
    args = parse_args()
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    os.makedirs(f"{intermediate_path}/block_df_split", exist_ok=True)
    #
    # index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk, args.max_size, args.min_size)
    # signal_path_arr = list(index_dict.keys())
    # signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
    # gc.collect()
    #
    # with open(signal_index_path, "wb") as outfile:
    #     pickle.dump(index_dict, outfile)
    #
    # signal_path_dict = {}
    # for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Signal Path Dictionary"):
    #     for read_id in id_list:
    #         signal_path_dict[read_id] = signal_path.split('/')[-1]
    #
    # del index_dict
    # gc.collect()
    #
    # split_block_df(args, signal_path_dict, signal_name_arr, intermediate_path)
    # extract_move(args.bam, args.cpu, signal_path_dict, signal_name_arr, intermediate_path)
    #
    # del signal_path_dict, signal_name_arr
    # gc.collect()

    with open(signal_index_path, "rb") as infile:
        index_dict = pickle.load(infile)
    signal_path_arr = list(index_dict.keys())
    del index_dict
    gc.collect()

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    for signal_paths in signal_path_arr_split:
        proc = mp.Process(target=segment_normalize_fft_signal,
                          args=(args.output, signal_paths))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    return None


## TODO: update this script using evaluate/segment_transcript.py
## It contains several major performance improvements.


if __name__ == "__main__":
    main()
