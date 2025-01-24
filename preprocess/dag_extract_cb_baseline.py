import argparse
import atexit
import gc
import itertools as it
import json
import multiprocessing as mp
import os
import glob
import time
import numpy as np
import pandas as pd
import polyleven as pl
import pysam
from tqdm import tqdm

from utils.utils import mean_phred, printmessage


def get_ed_kmers(spacer_size, spacer_list):
    nucs = "ACGT"
    possible_nucs = ["".join(x) for x in it.product(nucs, repeat=spacer_size)]
    ed_dict_list = []
    for spacer in tqdm(spacer_list, desc="Calculating Levenshtein"):
        ed_dict = {}
        for possible_kmer in possible_nucs:
            ed_dict[possible_kmer] = pl.levenshtein(spacer, possible_kmer)
        ed_dict_list.append(ed_dict)
    return ed_dict_list


def find_block_candidates(record_list, pos_idx_list, ed_dict_list, spacer_size, cb_pad, save_path, last_spacer, boi="A"):

    read_id_list = []
    central_pos_list = []
    penalty_list = []
    motif_list = []
    cb_idx_list = []

    for record in tqdm(record_list, desc="Finding block"):
        read_id, seq, read_length = record
        slice_idx = seq.rfind(last_spacer)
        if slice_idx == -1:
            continue
        seq = seq[:slice_idx+spacer_size]
        seq_len = len(seq)
        pos_thres = spacer_size+cb_pad
        for cb_idx, neg_pos in pos_idx_list:
            pos = seq_len - neg_pos
            if pos < pos_thres:
                break
            central = seq[pos]
            if central == boi:
                front = pos-cb_pad
                back = pos+cb_pad+1
                motif = seq[front:back]
                spacer_front = seq[front-spacer_size:front]
                spacer_back = seq[back:back+spacer_size]
                spacer_front_ed = ed_dict_list[cb_idx][spacer_front]
                spacer_back_ed = ed_dict_list[cb_idx+1][spacer_back]
                penalty = spacer_front_ed + spacer_back_ed
                motif_list.append(motif)
                read_id_list.append(read_id)
                central_pos_list.append(pos)
                penalty_list.append(penalty)
                cb_idx_list.append(cb_idx)

    del record_list
    gc.collect()

    block_df = pd.DataFrame({"read_id": read_id_list, "pos_RM": central_pos_list, "cb_idx": cb_idx_list, "penalty": penalty_list, "motif": motif_list,})
    block_df.to_pickle(save_path)

    del block_df
    gc.collect()

    return None


def get_central_pos(spacer_size, cb_pad, max_read_length, cb_per_bb):
    cb_size = 2 * cb_pad + 1
    bb_size = cb_size * cb_per_bb + spacer_size * (cb_per_bb + 1)
    max_bb_count = max_read_length // bb_size + 1
    pos_in_cb = np.arange(cb_per_bb) * (cb_size + spacer_size) + spacer_size + cb_pad
    pos_list = np.concatenate([pos_in_cb + i * bb_size for i in range(max_bb_count)])+1
    pos_idx_list = [(cb_per_bb-i%cb_per_bb-1, x) for i, x in enumerate(pos_list)]
    return pos_idx_list


def extract_blocks_from_read_list(input, output, max_read_length, min_read_length, spacer_list, spacer_size, cb_pad,
                                  cb_per_bb, read_bq_cutoff, flush_path, ncpu, boi):
    boi = "A"
    spacer_list = [x.replace("U", "T") for x in spacer_list]
    ed_dict_list = get_ed_kmers(spacer_size, spacer_list)
    pos_idx_list = get_central_pos(spacer_size, cb_pad, max_read_length, cb_per_bb)
    last_spacer = spacer_list[-1]
    record_list = []

    with pysam.AlignmentFile(input, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm(total=input_bam.mapped, desc="Reading BAM") as pbar:
            for idx, record in enumerate(input_bam):
                qscore = mean_phred(np.array(record.query_qualities, dtype=int))
                if qscore >= read_bq_cutoff :
                    read_length = record.query_length
                    if read_length <= max_read_length and read_length >= min_read_length:
                        record_list.append((record.query_name, str(record.query_sequence), read_length))
                pbar.update(1)

    record_list.sort(key=lambda x: x[2], reverse=True)
    record_split_dict = {i: [] for i in range(ncpu)}
    for i, record in enumerate(record_list):
        group = int(np.abs((i % (2 * ncpu)) - ncpu + 0.5) - 0.5)
        record_split_dict[group].append(record)
    del record_list
    gc.collect()

    proc_list = []

    for pid in range(ncpu):
        save_path = os.path.join(flush_path, f"{pid}.pkl")
        proc = mp.Process(target=find_block_candidates,
                          args = (record_split_dict[pid], pos_idx_list, ed_dict_list, spacer_size, cb_pad, save_path, last_spacer, boi))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    result_df = []
    for save_path in tqdm(glob.glob(f"{flush_path}/*.pkl"), desc="Merging"):
        result_df.append(pd.read_pickle(save_path))

    result_df = pd.concat(result_df)
    result_df.to_pickle(os.path.join(output, "block_df.pkl"))
    print(result_df)
    printmessage(f"Saved context blocks to {output}.")

    del result_df
    gc.collect()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--input", dest="input", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str, required=True)
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))
    parser.add_argument("--boi", dest="boi", type=str, default="A")
    parser.add_argument("--sp", dest="spacer_list", type=str, nargs="+",
                        default=["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"])
    parser.add_argument("--ss", dest="spacer_size", type=int, default=6)
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cb", dest="cb_per_bb", type=int, default=3)
    parser.add_argument("--rbq", dest="read_bq_cutoff", type=int, default=7)
    parser.add_argument("--max", dest="max_read_length", type=int, default=1000)
    parser.add_argument("--min", dest="min_read_length", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"ERROR! {args.input} does not exist.")

    flush_path = os.path.join(args.output, f"block_flush_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(flush_path, exist_ok=True)

    args_dict = vars(args)
    extract_blocks_from_read_list(**args_dict, flush_path=flush_path)

    return None


if __name__ == "__main__":
    main()
