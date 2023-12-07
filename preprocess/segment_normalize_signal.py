import argparse
import gc
import multiprocessing as mp

import glob
import pickle

import numpy as np
import os
import pandas as pd
import pod5
import pysam
import scipy
import tqdm

EXPIDS = ["ON0089", "ON0087"]
DIGITISATION = 8192


def mean_phred(phred):
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_dict):
    signal_list = []
    offset_list = []
    scale_list = []
    id_list = []
    for pod5_path in tqdm.tqdm(pod5_path_list):
        with pod5.Reader(pod5_path) as reader:
            for record in reader:
                signal_arr = record.signal
                offset = record.calibration.offset
                scale = record.calibration.scale
                offset_list.append(offset)
                scale_list.append(scale)
                signal_list.append(signal_arr)
                id_list.append(record.read_id)
        gc.collect()
    df = pd.DataFrame({"signal": signal_list, "id": id_list, "offset": offset_list, "scale": scale_list})
    df["id"] = df["id"].astype(str)
    df.to_pickle(f"{signal_df_path}/{pid}.pkl")
    index_dict[pid] = id_list
    return None


def extract_move(bam_path):
    ## Extract mv tag from bam and save to separate file
    bam_file = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    mv_list = []
    id_list = []
    sm_list = []
    sd_list = []
    ts_list = []
    qs_list = []
    sl_list = []

    for read in tqdm.tqdm(bam_file, total=bam_file.count()):
        if read.has_tag("pi"):
            continue
        mv_list.append(read.get_tag("mv"))
        id_list.append(read.query_name)
        sm_list.append(read.get_tag("sm"))
        sd_list.append(read.get_tag("sd"))
        ts_list.append(read.get_tag("ts"))
        qs_list.append(read.get_tag("qs"))
        sequence_length = len(read.query_sequence)
        sl_list.append(sequence_length)

    move_df = pd.DataFrame(
        {"mv": mv_list, "id": id_list, "qs": qs_list, "sm": sm_list, "sd": sd_list, "ts": ts_list, "sl": sl_list})
    move_df["id"] = move_df["id"].astype(str)

    del mv_list, id_list
    gc.collect()

    return move_df


def preprocess_pod5(pod5_path, save_path, ncpu):
    # Export pod5 to csv
    pod5_path_list = glob.glob(pod5_path + "*.pod5")
    proc_list = []
    pod5_path_list_split = np.array_split(pod5_path_list, ncpu)

    man = mp.Manager()
    index_dict = man.dict()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_signal_proc, args=(pod5_path_list_split[pid], save_path, pid, index_dict))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    index_dict = dict(index_dict)
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


def segment_spectrogram(signal, move, min_freq=0, max_freq=3000, step_size=10):
    f, t, sxx = scipy.signal.spectrogram(signal, fs=1, window="hann", nperseg=100, noverlap=50, mode="magnitude")
    ## refit sxx into min_freq ~ max_freq with step_size
    spectrogram = []
    for freq in range(min_freq, max_freq + step_size, step_size):
        idx = np.where((f >= freq) & (f < freq + step_size))[0]
        if len(idx) > 0:
            sxx_freq = np.sum(sxx[idx, :], axis=0)
            spectrogram.append(sxx_freq)
        else:
            spectrogram.append(np.zeros(len(t)))
    spectrogram = np.stack(spectrogram, axis=0)
    spectrogram = spectrogram.T

    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal_segmented = np.array_split(spectrogram, move_idx, axis=0)
    return signal_segmented


def segment_normalize_fft_signal(signal_df_path, seg_df_path, move_df, block_df, pid_arr):

    for pid in tqdm.tqdm(pid_arr):
        signal_df = pd.read_pickle(f"{signal_df_path}/{pid}.pkl")
        signal_df = signal_df.merge(move_df, on="id", how="inner")
        gc.collect()

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["signal_len"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["signal_len"] > signal_df["ts"]]
        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["ts"]:], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: np.flip(x["signal"], axis=0), axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] + x["offset"]) * x["scale"], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] - x["sm"]) / x["sd"], axis=1)
        signal_df["signal_seg"] = signal_df.apply(lambda x: segment_signal(x["signal"], x["mv"]), axis=1)
        signal_df["signal_fft"] = signal_df.apply(lambda x: segment_spectrogram(x["signal"], x["mv"]), axis=1)
        signal_df = signal_df[["id", "signal_seg", "signal_fft"]]
        signal_df.to_pickle(f"{seg_df_path}/segment/signal_segment_{pid}.pkl")

        block_df_proc = block_df.merge(signal_df, left_on="id", right_on="id", how="inner")
        block_df_proc["signal_seg"] = block_df_proc.apply(lambda row: row["signal_seg"][row["start_pos"]:row["end_pos"]], axis=1)
        block_df_proc["signal_fft"] = block_df_proc.apply(lambda row: row["signal_fft"][row["start_pos"]:row["end_pos"]], axis=1)

        block_df_proc.to_pickle(f"{seg_df_path}/block/signal_block_{pid}.pkl")
        del signal_df, block_df_proc
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
    args = parser.parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if os.path.exists(args.output):
        raise FileExistsError(f"Output directory {args.output} already exists")
    os.makedirs(args.output)
    os.makedirs(f"{args.output}/segment/")
    os.makedirs(f"{args.output}/block/")
    return args


def main():
    args = parse_args()
    intermediate_path = f"{args.output}/intermediates/"
    os.makedirs(intermediate_path, exist_ok=True)

    move_df = extract_move(args.bam)
    print(move_df)
    move_df.to_pickle(f"{intermediate_path}/move_df.pkl")

    signal_raw_path = f"{intermediate_path}/signal_raw/"
    os.makedirs(signal_raw_path, exist_ok=True)
    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu)

    signal_index_path = f"{intermediate_path}/signal_index.pkl"
    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)

    block_df = pd.read_pickle(args.block)

    proc_list = []
    thread_reduction_factor = 8 ## Not to blow up memory. It is a temporary fix and will be removed in the future.
    pid_arr_split = np.array_split(list(range(args.cpu)), min(1, args.cpu // thread_reduction_factor))
    move_df_proc_list = []
    block_df_proc_list = []

    for pid_arr in pid_arr_split:
        id_list = []
        for pid in pid_arr:
            id_list += index_dict[pid]
        move_df_proc = move_df[move_df["id"].isin(id_list)]
        move_df_proc_list.append(move_df_proc)
        block_df_proc = block_df[block_df["id"].isin(id_list)]
        block_df_proc_list.append(block_df_proc)

    del move_df
    gc.collect()

    for pid_arr, move_df_proc, block_df_proc in zip(pid_arr_split, move_df_proc_list, block_df_proc_list):
        proc = mp.Process(target=segment_normalize_fft_signal, args=(signal_raw_path, args.output, move_df_proc, block_df_proc, pid_arr))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return None


if __name__ == "__main__":
    main()
