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


def extract_move(bam_path, ncpu, bq_cutoff, output):
    ## Extract mv tag from bam and save to separate file
    data_dict = {"read_id": [], "md": [], "ref": [], "start": [], "cigar": [], "query_len": []}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
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

                if read.has_tag("MD"):
                    md = read.get_tag("MD")
                else:
                    md = ""

                data_dict["read_id"].append(read_id)
                data_dict["ref"].append(read.reference_name)
                data_dict["start"].append(read.reference_start)
                data_dict["cigar"].append(read.cigarstring)
                data_dict["md"].append(md)
                data_dict["query_len"].append(read.query_length)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="warning")
    printmessage(f"Missing BQ: {missing_bq}", msg_type="warning")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="warning")
    printmessage(f"Missing Move: {missing_move}", msg_type="warning")
    printmessage(f"Unmapped: {unmapped}", msg_type="warning")

    data_df = pd.DataFrame(data_dict)
    data_df.to_pickle(f"{output}/move_df.pkl")

    gc.collect()
    return data_df


def standardise_trim_segment_signal(signal,move,sp,ts,ns,offset,scale,mean,stdev):
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


def segment_normalize_signal(data_df, label_df, pid, token_output_path):


    out_path = f"{token_output_path}/{pid}.pkl"

    data_df[['ref_pos', 'query_pos', 'error', 'label', "dom"]] = data_df.apply(lambda x: get_label_pos_list(
        x["ref"], x["start"], x["cigar"], x["query_len"], x["md"], label_df), axis=1, result_type="expand")
    data_df = data_df[data_df["query_pos"].notnull()].copy()

    data_df = data_df.explode(["query_pos", 'ref_pos', "error", "label", "dom"]).copy()

    # data_df["label_id"] = data_df["ref"].astype(str) + ":" + data_df["ref_pos"].astype(str)
    data_df["block_label_id"] = data_df["read_id"] + ":" + data_df["ref"].astype(str) + ":" + data_df["ref_pos"].astype(str)

    data_df = data_df[["block_label_id", "error"]].copy()

    data_df.to_pickle(out_path)

    return None



def ref_pos_to_query_pos(ref_pos_list, cigar, start_pos, query_len, md_tag, cb_pad = 10):

    if query_len < 2 * cb_pad + 1:
        return [], []

    cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
    mis_arr = md_to_mismatch_arr(md_tag)

    query_pos_dict = {}
    error_arr = np.zeros((query_len,3),dtype=bool)
    ## Channel 0: Mismatch, Channel 1: Insertion, Channel 2: Deletion

    idx_ref_prev = start_pos
    idx_ref = start_pos

    idx_query_prev = 0
    idx_query = 0

    insertion_before_match_flag = True

    for length, match in cigar_list:
        length = int(length)
        if match == "M":
            insertion_before_match_flag = False
            mis_arr_slice = mis_arr[idx_ref-start_pos:idx_ref+length-start_pos]
            error_arr[idx_query:idx_query+length, 0] = mis_arr_slice
            idx_ref += length
            idx_query += length
        elif match == "I":
            if not insertion_before_match_flag:
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
    placeholder = np.zeros((2*cb_pad+1,3),dtype=bool)
    error_list = [error_arr[x-cb_pad:x+cb_pad+1] if x != -1 else placeholder for x in query_pos_list]
    error_list = np.array(error_list)
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
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    args = parser.parse_args()
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if not os.path.exists(args.label):
        raise FileNotFoundError(f"label file {args.label} does not exist")
    return args

def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/meta_df_split", exist_ok=True)
    token_output_path = f"{args.output}/output"
    os.makedirs(token_output_path, exist_ok=True)

    data_df = extract_move(args.bam, args.cpu, args.qcut, args.output)
    print(data_df)

    data_df_split = np.array_split(data_df, args.cpu)
    label_df = pd.read_csv(args.label, sep="\t")
    proc_list = []
    label_df = label_df.groupby("nmid")

    for pid, data_df in enumerate(data_df_split):
        proc = mp.Process(target=segment_normalize_signal,
                          args=(data_df, label_df, pid, token_output_path))
        proc_list.append(proc)
        proc.start()

    gc.collect()

    for proc in proc_list:
        proc.join()

    gc.collect()

    return None


if __name__ == "__main__":
    main()
