import argparse
import gc
import multiprocessing as mp
import os
import pickle
import re

import numpy as np
import pandas as pd
import tqdm
import pysam

from archived.misc.segment_normalize_signal import segment_signal
from utils.utils import mean_phred, oom_killer, printmessage
from archived.misc.tokenizer import create_segment_len_arr, sequence_to_kmer_token, expand_token_to_segment, \
    create_positional_token, create_move_token, segmented_signal_to_block, create_target_mask


def md_to_mismatch_arr(md):
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


def extract_move(bam_path, ncpu, signal_path_dict, signal_path_arr, intermediate_path):
    ## Extract mv tag from bam and save to separate file

    data_dict = {x: {"mv": [], "read_id": [], "sm": [], "sd": [], "ts": [], "seq": [], "bq": [], "mapq": [], "flag": [],
                     "ref": [], "start": [], "cigar": [], "md": [], "pi": []} for x in signal_path_arr}

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                pbar.update(1)

                read_id = str(read.query_name)

                try:
                    signal_path = signal_path_dict[read_id]
                except:
                    continue

                try:
                    bq = np.array(read.query_qualities, dtype=int)
                except:
                    continue

                if read.has_tag("pi"):
                    pi = 1
                else:
                    pi = 0

                data = data_dict[signal_path]
                data["read_id"].append(read_id)
                data["seq"].append(str(read.query_sequence))
                data["mv"].append(read.get_tag("mv"))
                data["sm"].append(read.get_tag("sm"))
                data["sd"].append(read.get_tag("sd"))
                data["ts"].append(read.get_tag("ts"))
                data["md"].append(read.get_tag("MD"))
                data["pi"].append(pi)
                data["bq"].append(bq)
                data["ref"].append(read.reference_name)
                data["start"].append(read.reference_start)
                data["cigar"].append(read.cigarstring)
                data["mapq"].append(read.mapping_quality)
                data["flag"].append(read.flag)

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split_v2/{signal_path.split('/')[-1]}")
        del move_df

    del data_dict

    gc.collect()

    return None

def segment_normalize_fft_signal(seg_df_path, pid, wdir_path, signal_path_arr, label_df, sampling_ratio = None,
                                 kmer = 5, cb_len = 21, sampling = 5, sig_window = 5, shard_size = 10000, boi = "A"):

    cb_half_len = cb_len//2
    cb_lr_pad = (cb_len-kmer)//2
    trim = kmer//2

    buffer = []

    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        printmessage("Point 1")
        out_path = f"{seg_df_path}/{signal_path.split('/')[-1]}"
        if os.path.exists(out_path):
            continue
        move_path = f"{wdir_path}/move_df_split_v2/{signal_path.split('/')[-1]}"
        if not os.path.exists(move_path):
            continue
        signal_df = pd.read_pickle(signal_path)
        if len(signal_df) == 0:
            continue

        printmessage("Point 2")
        move_df = pd.read_pickle(move_path)
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df
        gc.collect()

        signal_df["query_len"] = signal_df["seq"].apply(len)

        alignment_zip = zip(signal_df["ref"], signal_df["start"], signal_df["cigar"], signal_df["query_len"], signal_df["md"])

        printmessage("Point 3")

        pos_list = [get_label_pos_list(ref, start, cigar, query_len, md_tag, label_df) for ref, start, cigar, query_len, md_tag in alignment_zip]
        signal_df["ref_query_error_label"] = pos_list
        signal_df = signal_df[signal_df["ref_query_error_label"].notnull()].copy()
        signal_df = signal_df[signal_df["ref_query_error_label"].apply(lambda x: len(x) > 0)]
        if len(signal_df) == 0:
            continue
        del pos_list, alignment_zip
        gc.collect()

        printmessage("Point 4")

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["signal_len"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["signal_len"] > signal_df["ts"]]
        if len(signal_df) == 0:
            continue
        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["ts"]:], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: np.flip(x["signal"], axis=0), axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] + x["offset"]) * x["scale"], axis=1)
        signal_df["signal"] = signal_df.apply(lambda x: (x["signal"] - x["sm"]) / x["sd"], axis=1)

        signal_df["signal_seg"] = signal_df.apply(lambda x: segment_signal(x["signal"], x["mv"]), axis=1)
        gc.collect()

        signal_df = signal_df[["read_id", "ref", "bq", "seq", "signal_seg", "ref_query_error_label", "mapq", "flag", "pi"]].copy()
        signal_df["read_bq"] = signal_df["bq"].apply(mean_phred)
        gc.collect()
        signal_df = signal_df[signal_df["ref_query_error_label"].apply(lambda x: len(x) > 0)]
        signal_df = signal_df.explode("ref_query_error_label").reset_index(drop=True)

        if len(signal_df) == 0:
            continue

        printmessage("Point 5")

        signal_df["ref"] = signal_df["ref"].str.split(".").str[0]
        signal_df["ref_pos"] = signal_df["ref_query_error_label"].apply(lambda x: x[0])
        signal_df["query_pos"] = signal_df["ref_query_error_label"].apply(lambda x: x[1])
        signal_df["error"] = signal_df["ref_query_error_label"].apply(lambda x: x[2])
        signal_df["label"] = signal_df["ref_query_error_label"].apply(lambda x: x[3])
        signal_df.drop("ref_query_error_label", axis=1, inplace=True)

        signal_df["centre_nuc"] = signal_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)
        signal_df = signal_df[signal_df["centre_nuc"] == boi]

        if len(signal_df) == 0:
            continue

        signal_df["block_id"] = signal_df["read_id"] + ":" + signal_df["query_pos"].astype(str)
        signal_df["label_id"] = signal_df["ref"].astype(str) + ":" + signal_df["ref_pos"].astype(str)
        signal_df["start"] = signal_df["query_pos"] - cb_half_len
        signal_df["end"] = signal_df["query_pos"] + cb_half_len + 1
        signal_df["query_len"] = signal_df["seq"].apply(len)

        signal_df = signal_df[(signal_df["start"] >= 0) & (signal_df["end"] <= signal_df["query_len"])]

        if len(signal_df) == 0:
            continue

        printmessage("Point 6")

        signal_df["signal_seg"] = signal_df.apply(lambda x: x["signal_seg"][x["start"]:x["end"]], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start"]:x["end"]], axis=1)
        signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start"]:x["end"]], axis=1)

        signal_df = signal_df[['block_id', 'label_id', 'motif', 'signal_seg', 'bq', 'label', "mapq", "flag", "pi",
                               "read_bq", "error"]].copy()
        gc.collect()

        signal_df["bq"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        signal_df["segment_len_arr"] = signal_df["signal_seg"].apply(lambda x: create_segment_len_arr(x, sampling))
        signal_df["signal_token"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal_seg"], x["segment_len_arr"],
                                                                                        kmer, sampling, sig_window), axis=1)
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
        signal_df["kmer_token"] = signal_df["motif"].apply(lambda x: sequence_to_kmer_token(x, kmer))
        signal_df["kmer_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["kmer_token"], x["segment_len_arr"]), axis=1)
        signal_df["bq_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["bq"], x["segment_len_arr"]), axis=1)
        signal_df["position_token"] = signal_df["segment_len_arr"].apply(lambda x: create_positional_token(np.sum(x)))
        signal_df["move_token"] = signal_df["segment_len_arr"].apply(lambda x: create_move_token(x))
        signal_df["spectrogram_token"] = None
        signal_df["target_mask"] = signal_df["segment_len_arr"].apply(lambda x: create_target_mask(x, cb_lr_pad))

        printmessage("Point 7")

        signal_df = signal_df[["block_id", "label_id", "label", "motif", "kmer_token", "bq_token", "position_token", "signal_token",
                               "spectrogram_token", "move_token", "target_mask","mapq", "flag", "pi", "read_bq", "error"]][signal_df["signal_token"].notnull()].copy()

        signal_df["block_bq"] = signal_df["bq_token"].apply(mean_phred)
        signal_df["base_bq"] = signal_df["bq_token"].apply(lambda x: x[cb_half_len-trim])

        signal_df["center_mis"] = signal_df["error"].apply(lambda x: x[cb_half_len-trim,0])
        signal_df["center_ins"] = signal_df["error"].apply(lambda x: x[cb_half_len-trim,1])
        signal_df["center_del"] = signal_df["error"].apply(lambda x: x[cb_half_len-trim,2])

        printmessage("Point 8")

        gc.collect()

        if len(buffer) > 0:
            signal_df = pd.concat([buffer, signal_df], ignore_index=True)
            buffer = []

        if len(signal_df) < shard_size:
            buffer = signal_df

        else:
            for chunk_idx in range(0, len(signal_df) // shard_size):
                chunk = signal_df.iloc[chunk_idx * shard_size:(chunk_idx + 1) * shard_size]
                if sampling_ratio is not None:
                    chunk = chunk.sample(frac=sampling_ratio)
                chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

            chunk = signal_df.iloc[(len(signal_df) // shard_size) * shard_size:].copy()

            if len(chunk) > 0:
                buffer = chunk
            del signal_df

        printmessage("Point 9")

        gc.collect()

    out_path = f"{seg_df_path}/last-{pid}.pkl"
    ## For last chunk, append zero data and save
    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // shard_size):
            chunk = buffer.iloc[chunk_idx * shard_size:(chunk_idx + 1) * shard_size]
            if sampling_ratio is not None:
                chunk = chunk.sample(frac=sampling_ratio)
            chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

        chunk = buffer.iloc[(len(buffer) // shard_size) * shard_size:].copy()

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
                            query_pos = None
                        query_pos_dict[save_idx] = query_pos

        idx_ref_prev = idx_ref
        idx_query_prev = idx_query

    assert len(error_arr) == query_len

    for save_idx in range(len(ref_pos_list)):
        if save_idx not in query_pos_dict:
            query_pos_dict[save_idx] = None
        elif query_pos_dict[save_idx] is not None:
            if query_pos_dict[save_idx] < cb_pad:
                query_pos_dict[save_idx] = None
            elif query_pos_dict[save_idx] >= query_len - cb_pad:
                query_pos_dict[save_idx] = None
            else:
                pass
        else:
            pass

    query_pos_list = [query_pos_dict[x] for x in range(len(ref_pos_list))]

    query_pos_list_filt = []
    error_list_filt = []

    ## filter errors
    for query_pos in query_pos_list:

        if query_pos is None:
            query_pos_list_filt.append(None)
            error_list_filt.append(None)
        else:
            assert query_pos - cb_pad >= 0
            assert query_pos + cb_pad + 1 <= query_len
            error_slice = error_arr[query_pos - cb_pad : query_pos + cb_pad+1]

            query_pos_list_filt.append(query_pos)
            error_list_filt.append(error_slice)

    return query_pos_list_filt, error_list_filt


def get_label_pos_list(ref, start, cigar, query_len, md_tag, label_df):
    ref = ref.split(".")[0]
    label_df = label_df[label_df["nmid"] == ref]

    if len(label_df) == 0:
        return []

    label_list = label_df["label"].values
    ref_pos_list = label_df["pos"].values
    query_pos_list, error_list = ref_pos_to_query_pos(ref_pos_list, cigar, start, query_len, md_tag)

    assert len(ref_pos_list) == len(query_pos_list)
    assert len(ref_pos_list) == len(error_list)

    pos_tuple_list = list(zip(ref_pos_list, query_pos_list, error_list, label_list))
    pos_tuple_list_filtered = [x for x in pos_tuple_list if x[1] is not None]

    if len(pos_tuple_list_filtered) == 0:
        return None

    return pos_tuple_list_filtered


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--wdir", "-w", type=str, required=True, help="Working directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=1000, help="Chunk size")
    parser.add_argument("--ratio", "-r", type=float, default=None, help="Sampling ratio")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    args = parser.parse_args()
    # if not os.path.exists(args.pod5):
    #     raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    # if not os.path.exists(args.bam):
    #     raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if os.path.exists(args.output):
        ## check empty directory
        if len(os.listdir(args.output)) > 0:
            raise FileExistsError(f"Output directory {args.output} already exists")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wdir, exist_ok=True)
    return args


def main():
    args = parse_args()
    intermediate_path = args.wdir
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split_v2", exist_ok=True)

    # index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk)
    # signal_path_arr = list(index_dict.keys())
    # gc.collect()
    #
    # with open(signal_index_path, "wb") as outfile:
    #     pickle.dump(index_dict, outfile)
    # gc.collect()
    #
    # signal_path_dict = {}
    # for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Signal Path Dictionary"):
    #     for read_id in id_list:
    #         signal_path_dict[read_id] = signal_path
    #
    # del index_dict
    # gc.collect()

    with open(signal_index_path, "rb") as infile:
        index_dict = pickle.load(infile)

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Signal Path Dictionary"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path
    signal_path_arr = list(index_dict.keys())

    # extract_move(args.bam, args.cpu, signal_path_dict, signal_path_arr, intermediate_path)


    del signal_path_dict, index_dict
    gc.collect()

    label_df = pd.read_csv(args.label, sep="\t")
    # label_df = label_df[label_df["label"] != 0].copy()

    print(label_df)

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    for pid, signal_paths in enumerate(signal_path_arr_split):
        proc = mp.Process(target=segment_normalize_fft_signal,
                          args=(args.output, pid, intermediate_path, signal_paths, label_df, args.ratio))
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
