import argparse
import gc
import glob
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd
import scipy
import tqdm
import pysam

from preprocess.segment_normalize_signal import preprocess_pod5, segment_signal, segment_spectrogram


def extract_move(bam_path,ncpu):
    ## Extract mv tag from bam and save to separate file
    bam_file = pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu)
    mv_list = []
    id_list = []
    sm_list = []
    sd_list = []
    ts_list = []
    qs_list = []
    sl_list = []
    sq_list = []

    for read in tqdm.tqdm(bam_file, total=bam_file.count()):
        if read.has_tag("pi"):
            continue
        mv_list.append(read.get_tag("mv"))
        id_list.append(read.query_name)
        sm_list.append(read.get_tag("sm"))
        sd_list.append(read.get_tag("sd"))
        ts_list.append(read.get_tag("ts"))
        qs_list.append(read.get_tag("qs"))
        sq = read.query_sequence
        sequence_length = len(sq)
        sl_list.append(sequence_length)
        sq_list.append(sq)

    move_df = pd.DataFrame(
        {"mv": mv_list, "read_id": id_list, "qs": qs_list, "sm": sm_list, "sd": sd_list, "ts": ts_list, "sl": sl_list, "seq": sq_list})
    move_df["read_id"] = move_df["read_id"].astype(str)

    del mv_list, id_list
    gc.collect()

    return move_df

def segment_normalize_fft_signal(seg_df_path, signal_path_arr):
    for signal_path in tqdm.tqdm(signal_path_arr):
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

        signal_df["signal_seg"] = signal_df.apply(lambda x: segment_signal(x["signal"], x["mv"]), axis=1)
        filter = scipy.signal.butter(4, 100, btype="highpass", fs=4000, output="sos")
        signal_df["signal_fft"] = signal_df.apply(lambda x: segment_spectrogram(x["signal"], x["mv"], filter), axis=1)

        signal_df["signal_seg"] = signal_df.apply(lambda x: x["signal_seg"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["signal_fft"] = signal_df.apply(lambda x: x["signal_fft"][x["start_pos"]:x["end_pos"]], axis=1)

        signal_zip = zip(signal_df["read_id"], signal_df["signal_seg"], signal_df["signal_fft"], signal_df["motif"])
        del signal_df
        gc.collect()
        df_list = [expand_row_to_blocks(read_id, signal_seg, signal_fft, seq) for read_id, signal_seg, signal_fft, seq in signal_zip]
        block_df = pd.concat(df_list)
        del df_list
        gc.collect()
        block_df.to_pickle(f"{seg_df_path}/{signal_path.split('/')[-1]}")
        del block_df
        gc.collect()


    return None


def expand_row_to_blocks(read_id, signal_seg, signal_fft, seq, boi="A", pad = 8):
    boi_pos_list = [i for i, x in enumerate(seq) if x == boi and i > pad and i < len(seq) - pad]
    block_id_list = [f"{read_id}-{i}" for i in range(len(boi_pos_list))]
    signal_seg_list = [signal_seg[i-pad:i+pad] for i in boi_pos_list]
    signal_fft_list = [signal_fft[i-pad:i+pad] for i in boi_pos_list]
    motif_list = [seq[i-pad:i+pad] for i in boi_pos_list]
    df = pd.DataFrame({"read_id": block_id_list, "signal_seg": signal_seg_list,
                       "signal_fft": signal_fft_list, "motif": motif_list})
    return df


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
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


def main():
    args = parse_args()
    intermediate_path = f"{args.output}/intermediates/"
    os.makedirs(intermediate_path, exist_ok=True)
    block_path = f"{args.output}/block/"
    os.makedirs(block_path, exist_ok=True)

    move_path = f"{intermediate_path}/move_df.pkl"
    move_df = extract_move(args.bam, args.cpu)
    move_df.to_pickle(move_path)

    signal_raw_path = f"{intermediate_path}/signal_raw/"
    os.makedirs(signal_raw_path, exist_ok=True)
    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu)

    signal_index_path = f"{intermediate_path}/signal_index.pkl"
    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)


    with open(signal_index_path, "rb") as infile:
        index_dict = pickle.load(infile)
    signal_path_arr_split = list(index_dict.keys())
    np.random.shuffle(signal_path_arr_split)
    signal_path_arr_split = np.array_split(signal_path_arr_split, max(1, args.cpu))


    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)

    for signal_path_arr in signal_path_arr_split:
        for signal_path in signal_path_arr:
            id_list = index_dict[signal_path]
            move_df_proc = move_df[move_df["read_id"].isin(id_list)]
            move_df_proc.to_pickle(f"{intermediate_path}/move_df_split/{signal_path.split('/')[-1]}")

    del move_df, index_dict, id_list
    gc.collect()

    proc_list = []
    for signal_path_arr in signal_path_arr_split:
        proc = mp.Process(target=segment_normalize_fft_signal,
                          args=(block_path, signal_path_arr))
        proc_list.append(proc)
        proc.start()

    del  signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    return None


if __name__ == "__main__":
    main()
