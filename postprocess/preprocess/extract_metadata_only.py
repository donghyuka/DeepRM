import argparse
import gc
import multiprocessing as mp
import os
import glob
import re
import numpy as np
import pandas as pd
import tqdm
import pysam
from utils.utils import mean_phred, printmessage

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


def parse_bam(bam_path, threads, bq_cutoff):
    """
    Parse BAM File into Pandas Dataframe.
    Dataframe contain SAM tags (including MOVE), Sequence, Mapping, and BQ.

    :param bam_path: A string. Path to BAM file.
    :param threads: An integer. Number of threads.
    :param bq_cutoff: An integer. Minimum BQ score.

    :return A Pandas Dataframe.
    """
    cols_list = ["mv", "read_id", "ts", "ns", "sp", "seq", "bq", "mapq", "flag", "md", "pi", "ref", "start", "cigar", "read_index"]
    data_dict = {x:[] for x in cols_list}

    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=threads) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read_index, read in enumerate(input_bam):
                pbar.update(1)
                pbar.set_postfix({"valid": valid_count, "invalid": missing_move + missing_bq + low_bq + unmapped})
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


                data_dict["mv"].append(mv)
                data_dict["read_id"].append(read_id)
                data_dict["ts"].append(ts)
                data_dict["ns"].append(ns)
                data_dict["sp"].append(sp)
                data_dict["seq"].append(str(read.query_sequence))
                data_dict["bq"].append(bq)
                data_dict["ref"].append(read.reference_name)
                data_dict["start"].append(read.reference_start)
                data_dict["cigar"].append(read.cigarstring)
                data_dict["md"].append(md)
                data_dict["pi"].append(pi)
                data_dict["mapq"].append(read.mapping_quality)
                data_dict["flag"].append(read.flag)
                data_dict["read_index"].append(read_index)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="warning")
    printmessage(f"Missing BQ: {missing_bq}", msg_type="warning")
    printmessage(f"Missing Move: {missing_move}", msg_type="warning")
    printmessage(f"Unmapped: {unmapped}", msg_type="warning")

    bam_df = pd.DataFrame.from_dict(data_dict, orient="columns")

    return bam_df


def get_left_soft_clip(cigar):
    """
    Calculate left soft clip length from CIGAR string.

    :param cigar: A string. CIGAR string.

    :return: An integer. Length of left soft clip.
    """


    if "S" in cigar:
        left_soft_clip = cigar.split("S")[0]
        if left_soft_clip.isdigit():
            left_soft_clip = int(left_soft_clip)
        else:
            left_soft_clip = 0

    else:
        left_soft_clip = 0
    return left_soft_clip


def ref_pos_to_query_pos(ref_pos_list, cigar, start_pos, query_len, md_tag, cb_pad = 10):
    """
    Convert reference position to query position.
    Also returns error array for context block.

    Error Array:
        Channel 0: Mismatch, Channel 1: Insertion, Channel 2: Deletion

    :param ref_pos_list: An iterable. Positions in reference to convert.
    :param cigar: A string. CIGAR string.
    :param start_pos: An integer. Start position in reference.
    :param query_len: An integer. Length of query.
    :param md_tag: A string. MD tag.
    :param cb_pad: An integer. Padding for context block (Context block size / 2 -1).
    :return: (query_pos_arr, error_arr) A tuple of numpy arrays.
             query_pos_arr is the converted query positions. Shape (len(ref_pos_list),).
             error_arr is the error array for context block. Shape (len(ref_pos_list), 2*cb_pad+1, 3).
    """

    cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
    mis_arr = md_to_mismatch_arr(md_tag)
    query_pos_dict = {}
    error_arr = np.zeros((query_len,3),dtype=bool)

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

    query_pos_arr = np.array([query_pos_dict[x] for x in range(len(ref_pos_list))], dtype=int)
    error_arr = np.lib.stride_tricks.sliding_window_view(error_arr, window_shape=2*cb_pad+1, axis=0)

    error_arr = error_arr[query_pos_arr - cb_pad]
    return query_pos_arr, error_arr


def get_label_pos_list(ref, start, cigar, query_len, md_tag, label_df):
    """
    Extract context blocks from each read.
    Each context block correspond to each position in label.

    :param ref: A string. Name of Reference sequence.
    :param start: An integer. Start position in reference.
    :param cigar: A string. CIGAR string.
    :param query_len: An integer. Length of query.
    :param md_tag: A string. MD tag.
    :param label_df: A Pandas DataFrame. Label DataFrame.

    :return: (ref_pos_arr, query_pos_arr, error_arr, label_arr, dom_arr)
            A tuple of numpy arrays.
            ref_pos_arr is the reference positions.
            query_pos_arr is the query positions.
            error_arr is the error array for context block.
            label_arr is the label array.
            dom_arr is the domain array.
    """

    ref = ref.split(".")[0]

    try:
        label_df = label_df.get_group(ref)
    except KeyError:
        return None, None, None, None, None

    if len(label_df) == 0:
        return None, None, None, None, None

    label_arr = label_df["label"].values
    dom_arr = label_df["dom_level"].values
    ref_pos_arr = label_df["pos"].values
    query_pos_arr, error_arr = ref_pos_to_query_pos(ref_pos_arr, cigar, start, query_len, md_tag)

    index_to_drop = [idx for idx, x in enumerate(query_pos_arr) if x == -1]
    ref_pos_arr = np.delete(ref_pos_arr, index_to_drop)
    query_pos_arr = np.delete(query_pos_arr, index_to_drop)
    error_arr = np.delete(error_arr, index_to_drop, axis=0)
    label_arr = np.delete(label_arr, index_to_drop)
    dom_arr = np.delete(dom_arr, index_to_drop)

    index_keep = np.where(query_pos_arr != -1)[0]
    ref_pos_arr = ref_pos_arr[index_keep]
    query_pos_arr = query_pos_arr[index_keep]
    error_arr = error_arr[index_keep]
    label_arr = label_arr[index_keep]
    dom_arr = dom_arr[index_keep]


    if len(query_pos_arr) == 0:
        return None, None, None, None, None

    return ref_pos_arr, query_pos_arr, error_arr, label_arr, dom_arr


def extract_write_metadata(bam_df, label_df, pid, out_dir, cb_len = 21, boi = "A"):

    """
    Extract metadata from BAM Dataframe and write to Dataframe.
    Each rows in the output are the context blocks that correspond to the positions in the given label Dataframe.

    :param bam_df: A Pandas DataFrame. BAM Dataframe.
    :param label_df: A Pandas DataFrame. Label DataFrame.
    :param pid: An Integer. Process ID.
    :param out_dir: A string. Output directory.
    :param cb_len: An Integer. Context block length.
    :param boi: A string. Base of interest.
    :return: None. Output is written to file.
    """
    
    cb_half_len = cb_len//2
    out_path = f"{out_dir}/{pid}.pkl"

    if os.path.exists(out_path):
        printmessage(f"Already exists: {out_path}", msg_type="warning")
        return None

    bam_df["query_len"] = bam_df["seq"].apply(len)

    ## Extract positions in each query that corresponds to reference positions in label
    bam_df[['ref_pos', 'query_pos', 'error', 'label', "dom"]] = bam_df.apply(lambda x: get_label_pos_list(
        x["ref"], x["start"], x["cigar"], x["query_len"], x["md"], label_df), axis=1, result_type="expand")

    bam_df = bam_df[bam_df["query_pos"].notnull()].copy()
    bam_df["left_soft_clip"] = bam_df["cigar"].apply(get_left_soft_clip)

    if len(bam_df) == 0:
        return None

    gc.collect()

    bam_df["mv"] = bam_df["mv"].apply(lambda x: np.array(x, dtype=int))

    bam_df = bam_df[["read_index", "ref", "bq", "seq", "ref_pos", "query_pos", "error", "label", "dom",
                     "mapq", "flag", "pi", "left_soft_clip"]].copy()
    bam_df["read_bq"] = bam_df["bq"].apply(mean_phred)

    ## Before explode: Each row is a query.
    ## After explode: Each row is a context block.
    bam_df = bam_df.explode(["ref_pos", "query_pos", "error", "label", "dom"], ignore_index=True)

    if len(bam_df) == 0:
        return None

    gc.collect()

    bam_df["centre_nuc"] = bam_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)
    bam_df = bam_df[bam_df["centre_nuc"] == boi]

    if len(bam_df) == 0:
        return None

    bam_df["label_id"] = bam_df["ref"].astype(str) + ":" + bam_df["ref_pos"].astype(str)
    bam_df["start_pos"] = bam_df["query_pos"] - cb_half_len
    bam_df["end_pos"] = bam_df["query_pos"] + cb_half_len + 1
    bam_df["query_len"] = bam_df["seq"].apply(len)

    bam_df = bam_df[(bam_df["start_pos"] >= 0) & (bam_df["end_pos"] <= bam_df["query_len"])]

    if len(bam_df) == 0:
        return None

    bam_df["bq"] = bam_df.apply(lambda x: x["bq"][x["start_pos"]:x["end_pos"]], axis=1)
    bam_df["motif"] = bam_df.apply(lambda x: x["seq"][x["start_pos"]:x["end_pos"]], axis=1)
    bam_df["block_bq"] = bam_df["bq"].apply(mean_phred)
    bam_df["base_bq"] = bam_df["bq"].apply(lambda x: x[cb_half_len])

    bam_df = bam_df[["label_index", "read_index", "label", "dom", "bq", "motif",
                     "mapq", "flag", "pi", "read_bq", "block_bq", "base_bq",
                     "error", "query_pos", "query_len", "left_soft_clip"]].copy()

    bam_df.to_pickle(out_path)

    return None



def parse_args():
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--thread", "-t", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    parser.add_argument("--boi", "-i", type=str, default="A", help="Base of interest")
    parser.add_argument("--cb_len", "-c", type=int, default=21, help="Context block length")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")

    intermediate_path = f"{args.output}/intermediates/"
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)

    if not len(os.listdir(intermediate_path)) == 0:
        raise FileExistsError(f"Output directory {intermediate_path} is not empty")

    ## Read label and BAM files.\
    if args.label.endswith(".tsv"):
        label_df = pd.read_csv(args.label, sep='\t')
    elif args.label.endswith(".pkl"):
        label_df = pd.read_pickle(args.label)
    else:
        raise ValueError("Label file must be either .tsv or .pkl")
    label_df["nmid"] = label_df["label_id"].str.split(":").str[0]
    label_df["pos"] = label_df["label_id"].str.split(":").str[1].astype(int)
    label_df = label_df.groupby("nmid")
    bam_df = parse_bam(args.bam, args.thread, args.qcut)
    bam_df = np.array_split(bam_df, args.thread)

    ## Extract metadata from BAM and write to intermediate files.
    proc_list = []
    for pid, bam_df_split in enumerate(bam_df):
        proc = mp.Process(target=extract_write_metadata,
                          args=(bam_df_split, label_df, pid, intermediate_path, args.cb_len, args.boi))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    del bam_df, label_df
    gc.collect()

    intermediate_file_list = glob.glob(f"{intermediate_path}/*.pkl")
    if len(intermediate_file_list) == 0:
        raise FileNotFoundError("No intermediate files found")

    ## Merge intermediate files
    final_df = []
    for path in tqdm.tqdm(intermediate_file_list, desc="Merging Files"):
        final_df.append(pd.read_pickle(path))
        os.remove(path)
    final_df = pd.concat(final_df, axis=0).reset_index(drop=True)
    final_df.to_pickle(f"{args.output}/metadata.pkl")

    return None


if __name__ == "__main__":
    main()
