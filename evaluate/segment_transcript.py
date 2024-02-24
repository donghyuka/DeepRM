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
from utils.utils import mean_phred, oom_killer

def extract_move(bam_path,ncpu,bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path):
    ## Extract mv tag from bam and save to separate file

    data_dict = {x: {"mv": [], "read_id": [], "sm": [], "sd": [], "ts": [], "seq": [], "bq": []} for x in signal_path_arr}

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                if read.has_tag("pi"):
                    continue
                bq = np.array(read.query_qualities, dtype=int)
                if mean_phred(bq) < bq_cutoff:
                    continue
                read_id = str(read.query_name)
                signal_path = signal_path_dict[read_id]
                data = data_dict[signal_path]
                data["read_id"].append(read_id)
                data["seq"].append(str(read.query_sequence))
                data["mv"].append(read.get_tag("mv"))
                data["sm"].append(read.get_tag("sm"))
                data["sd"].append(read.get_tag("sd"))
                data["ts"].append(read.get_tag("ts"))
                data["bq"].append(bq)
                pbar.update(1)

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path.split('/')[-1]}")
        del move_df

    del data_dict

    gc.collect()

    return None

def segment_normalize_fft_signal(seg_df_path, signal_path_arr):
    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        out_path = f"{seg_df_path}/block/{signal_path.split('/')[-1]}"
        if os.path.exists(out_path):
            continue
        move_path = f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}"
        if not os.path.exists(move_path):
            continue
        signal_df = pd.read_pickle(signal_path)
        if len(signal_df) == 0:
            continue

        move_df = pd.read_pickle(move_path)
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
        gc.collect()

        signal_zip = zip(signal_df["read_id"], signal_df["signal_seg"], signal_df["signal_fft"], signal_df["seq"], signal_df["bq"])

        del signal_df, filter
        gc.collect()

        df_list = [expand_row_to_blocks(read_id, signal_seg, signal_fft, seq, bq) for
                   read_id, signal_seg, signal_fft, seq, bq in signal_zip]

        if len(df_list) == 0:
            continue

        block_df = pd.concat(df_list)
        del df_list, signal_zip

        gc.collect()
        block_df.to_pickle(out_path)

        del block_df
        gc.collect()

    return None


def expand_row_to_blocks(read_id, signal_seg, signal_fft, seq, bq, boi="A", pad = 10, zfill=6):
    boi_pos_list = [i for i, x in enumerate(seq) if x == boi and i >= pad and i < len(seq) - pad]
    block_id_list = [f"{read_id}:{str(i).zfill(zfill)}" for i in range(len(boi_pos_list))]
    signal_seg_list = [signal_seg[i-pad:i+pad+1] for i in boi_pos_list]
    signal_fft_list = [signal_fft[i-pad:i+pad+1] for i in boi_pos_list]
    motif_list = [seq[i-pad:i+pad+1] for i in boi_pos_list]
    bq_list = [bq[i-pad:i+pad+1] for i in boi_pos_list]
    df = pd.DataFrame({"read_id": block_id_list, "signal_seg": signal_seg_list,
                       "signal_fft": signal_fft_list, "motif": motif_list, "bq": bq_list})
    return df


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="BQ cutoff")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=200, help="Chunk size")
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
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)

    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk)
    signal_path_arr = list(index_dict.keys())
    gc.collect()

    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)
    gc.collect()

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Signal Path Dictionary"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path

    del index_dict
    gc.collect()


    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    extract_move(args.bam, args.cpu, args.qcut, signal_path_dict, signal_path_arr, intermediate_path)

    del signal_path_dict
    gc.collect()

    # with open(signal_index_path, "rb") as infile:
    #     signal_path_dict = pickle.load(infile)
    # signal_path_arr = list(signal_path_dict.keys())
    # del signal_path_dict
    # gc.collect()

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


if __name__ == "__main__":
    main()
