import argparse
import atexit
import gc
import itertools as it
import json
import multiprocessing as mp
import os
import glob
import time
from collections import defaultdict
import psutil

import networkx as nx
import numpy as np
import pandas as pd
import polyleven as pl
import pysam
from tqdm import tqdm

from utils.utils import mean_phred, printmessage, oom_killer


## Step 1: Index all k-mers from the read.
## Step 2: Connect the spacers using the k-mer index.
## Step 3: Build a DAG of spacers.
## Step 4: Find the longest path in the DAG.
## Step 5: Extract the sequence from the longest path.

## Tolerances (Scoring parameters):
## 1. Mismatch tolerance per spacer
## 2. Indel tolerance per CB
## 3. Mismatch tolerance per anchor

## This code is not really built for heavy lifting. Well how about several million reads? lol godspeed.


def get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size):
    min_ideal_displacement_dict = {}
    big_step_size = cb_size + spacer_size
    small_step_size = spacer_size

    for from_idx in range(cb_per_bb + 1):
        for to_idx in range(cb_per_bb + 1):
            if from_idx < to_idx:
                small_steps = 0
                big_steps = to_idx - from_idx
                displacement = big_step_size * big_steps
            else:
                small_steps = 1
                big_steps = cb_per_bb - from_idx + to_idx
                displacement = big_step_size * big_steps + small_step_size
            min_ideal_displacement_dict[(from_idx, to_idx)] = (displacement, small_steps, big_steps)

    return min_ideal_displacement_dict


def get_ideal_displacement(from_spacer_idx, to_spacer_idx, displacement, min_ideal_displacement_dict, cb_per_bb,
                           bb_size, front_spacer_size_offset):


    min_ideal_displacement, min_small_steps, min_big_steps = min_ideal_displacement_dict[
        (from_spacer_idx, to_spacer_idx)]

    min_ideal_displacement += front_spacer_size_offset

    if displacement <= min_ideal_displacement:
        ideal_displacement = min_ideal_displacement
        small_steps = min_small_steps
        big_steps = min_big_steps
    else:
        periodicity = round((displacement - min_ideal_displacement) / bb_size)
        ideal_displacement = min_ideal_displacement + periodicity * bb_size
        small_steps = min_small_steps + periodicity
        big_steps = min_big_steps + cb_per_bb * periodicity


    return ideal_displacement, small_steps, big_steps


def get_integer_partition(indel_tolerance, cb_size_tolerance):
    indel_dict = {}
    for spacing_error in range(-cb_size_tolerance, cb_size_tolerance + 1):
        indel_list = []
        for front_error in range(-indel_tolerance, indel_tolerance + 1):
            back_error = spacing_error - front_error
            if np.abs(front_error) + np.abs(back_error) <= indel_tolerance:
                indel_list.append((front_error, back_error))
        indel_list.sort(key=lambda x: np.abs(x[0]) + np.abs(x[1]))
        indel_dict[spacing_error] = indel_list
    return indel_dict


def get_kmer_dict(read, k_min, k_max, bq_cutoff, phred):
    kmer_dict = defaultdict(list)
    for k in range(k_min, k_max + 1):
        for i in range(len(read) - k + 1):
            kmer = read[i:i + k]
            if bq_cutoff:
                bq = np.mean(phred[i:i + k])
                if bq < bq_cutoff:
                    continue
            kmer_dict[kmer].append(i)
    return kmer_dict


def get_ed_kmers(kmer, spacer_edit_tolerance):
    kmer_size = len(kmer)
    nucs = "ACGU"
    kmer_ed_dict = defaultdict(list)
    spacer_size_min = max(0, kmer_size - spacer_edit_tolerance)
    spacer_size_max = kmer_size + spacer_edit_tolerance
    for mutant_size in range(spacer_size_min, spacer_size_max + 1):
        possible_nucs = ["".join(x) for x in it.product(nucs, repeat=mutant_size)]
        for possible_kmer in possible_nucs:
            ed = pl.levenshtein(kmer, possible_kmer, spacer_edit_tolerance)
            if ed <= spacer_edit_tolerance:
                kmer_ed_dict[ed].append(possible_kmer)
    return kmer_ed_dict


def validate_anchor(read, from_pos, to_pos, possible_indel_list, spacer_size, cb_pad, single_anchor,
                    indel_penalty, anchor_mismatch_penalty, displacement_error, front_spacer_size_offset):
    query = read[from_pos + spacer_size + front_spacer_size_offset:to_pos]
    anchor_candidate_list = [
        (displacement_error * indel_penalty + anchor_mismatch_penalty, 1, None, displacement_error)]
    ## penalty, missing_anchor, anchor_pos, total_indel
    for front_indel, back_indel in possible_indel_list:
        anchor_query = query[cb_pad + front_indel]
        if anchor_query == single_anchor:
            anchor_pos = from_pos + spacer_size + cb_pad + front_indel + front_spacer_size_offset
            total_indel = np.abs(front_indel) + np.abs(back_indel)
            anchor_candidate_list.append((total_indel * indel_penalty, 0, anchor_pos, total_indel))
    anchor_candidate_list.sort(key=lambda x: (x[0], x[1]))
    return anchor_candidate_list[0][1:]


def get_kmer_tuple(spacer_edit_tolerance, from_spacer_kmer_ed_dict, to_spacer_kmer_ed_dict):
    kmer_tuple_list = []
    for front_edit in range(spacer_edit_tolerance + 1):
        for back_edit in range(spacer_edit_tolerance + 1):
            total_edit = front_edit + back_edit
            for from_kmer, to_kmer in it.product(from_spacer_kmer_ed_dict[front_edit],
                                                 to_spacer_kmer_ed_dict[back_edit]):
                kmer_tuple_list.append((from_kmer, to_kmer, total_edit))
    return kmer_tuple_list


def find_block_candidates(seq, phred, cb_bq_cutoff, spacer_kmer_ed_dict, skip_size_tolerance, cb_pad,
                          cb_per_bb, indel_penalty, anchor_mismatch_penalty, spacer_edit_penalty,
                          spacer_size, spacer_list, indel_dict, min_ideal_displacement_dict, anchor_list,
                          score_converting_func, cb_size_tolerance, spacer_edit_tolerance,
                          bb_size):

    spacer_size_min = max(0, spacer_size - spacer_edit_tolerance)
    spacer_size_max = spacer_size + spacer_edit_tolerance

    kmer_pos_dict = get_kmer_dict(seq, spacer_size_min, spacer_size_max, cb_bq_cutoff, phred)
    cb_info_dict = defaultdict(list)

    for from_spacer_idx in range(len(spacer_list)):

        if from_spacer_idx == cb_per_bb:
            single_anchor = None
        else:
            single_anchor = anchor_list[from_spacer_idx]

        from_spacer_kmer_ed_dict = spacer_kmer_ed_dict[from_spacer_idx]

        for to_spacer_idx in range(len(spacer_list)):

            to_spacer_kmer_ed_dict = spacer_kmer_ed_dict[to_spacer_idx]
            kmer_tuple_list = get_kmer_tuple(spacer_edit_tolerance, from_spacer_kmer_ed_dict,
                                             to_spacer_kmer_ed_dict)

            for from_kmer, to_kmer, kmer_edit in kmer_tuple_list:
                for from_pos, to_pos in it.product(kmer_pos_dict[from_kmer], kmer_pos_dict[to_kmer]):

                    front_spacer_size_offset = spacer_size - len(from_kmer)
                    displacement = to_pos - from_pos
                    if displacement < 0:
                        continue

                    ideal_displacement, small_steps, big_steps = get_ideal_displacement(from_spacer_idx, to_spacer_idx,
                                                                                        displacement,
                                                                                        min_ideal_displacement_dict,
                                                                                        cb_per_bb, bb_size,
                                                                                        front_spacer_size_offset)
                    displacement_tolerance_skip = big_steps * skip_size_tolerance

                    displacement_error = displacement - ideal_displacement
                    displacement_error_abs = np.abs(displacement_error)

                    if displacement_error_abs > displacement_tolerance_skip:
                        continue

                    anchor_pos = None
                    is_cb = False
                    missing_anchor = 1

                    if big_steps == 1 and small_steps == 0 and displacement_error_abs <= cb_size_tolerance and single_anchor is not None:
                        possible_indel_list = indel_dict[displacement_error]
                        missing_anchor, anchor_pos, total_indel = validate_anchor(seq, from_pos, to_pos,
                                                                                  possible_indel_list,
                                                                                  spacer_size, cb_pad, single_anchor,
                                                                                  indel_penalty,
                                                                                  anchor_mismatch_penalty,
                                                                                  displacement_error_abs,
                                                                                  front_spacer_size_offset)
                        if missing_anchor == 0:
                            is_cb = True
                            displacement_error_abs = total_indel

                    elif big_steps == 0 and small_steps == 1 and displacement_error_abs <= spacer_edit_tolerance:
                        missing_anchor = 0

                    penalty = spacer_edit_penalty * kmer_edit + indel_penalty * displacement_error_abs + anchor_mismatch_penalty * missing_anchor
                    score = score_converting_func(penalty)

                    from_pos_id = (from_spacer_idx, from_pos)
                    to_pos_id = (to_spacer_idx, to_pos)

                    cb_info_dict[(from_pos_id, to_pos_id)].append([is_cb, from_spacer_idx, from_pos, to_pos, anchor_pos, penalty, score])

    filtered_cb_info_dict = {}
    dag_list = []  ## Format: [from_pos, to_pos, score]
    dag_dict = {}  ## Format: {(from_pos, to_pos): score}

    for key, items in cb_info_dict.items():
        # ## Check if any has is_cb = True
        # items_that_is_cb = [x for x in items if x[0]]
        # if len(items_that_is_cb) > 0:
        #     items = items_that_is_cb
        arg_max_score = np.argmax([x[6] for x in items])
        max_score_item = items[arg_max_score]
        if max_score_item[0]:
            filtered_cb_info_dict[key] = max_score_item[1:]
        dag_list.append([key[0], key[1], max_score_item[6]])
        dag_dict[(key[0], key[1])] = max_score_item[6]

    return filtered_cb_info_dict, dag_list, dag_dict


def dag_longest_path(edge_list):
    node_list = list(set([x[0] for x in edge_list] + [x[1] for x in edge_list]))

    dag = nx.DiGraph()
    dag.add_nodes_from(node_list)
    dag.add_weighted_edges_from(edge_list)
    longest_path = nx.dag_longest_path(dag, weight='weight')

    return longest_path


def extract_blocks_from_read_list_mp_worker(record_list, indel_penalty, cb_size_tolerance,
                                            skip_size_tolerance, anchor_mismatch_penalty, spacer_edit_tolerance,
                                            spacer_edit_penalty,
                                            cb_pad, cb_per_bb, cb_bq_cutoff, indel_dict, spacer_kmer_ed_dict,
                                            anchor_list, spacer_list, spacer_size, bb_size, flush_path, pid,
                                            flush_interval,
                                            score_converting_func, cb_size, min_ideal_displacement_dict, resume):
    len_record = len(record_list)
    block_df_list = []
    flush_file_list = []
    last_flush_idx = 0

    if resume is not None:
        ## search for last flush file
        flush_file_list = glob.glob(f"{resume}/df_{pid}_*.pkl")
        if len(flush_file_list) > 0:
            flush_idx = [int(x.split("_")[-1].split(".")[0]) for x in flush_file_list]
            last_flush_idx = max(flush_idx)
            record_list = record_list[last_flush_idx:]
            gc.collect()
            printmessage(f"[Process-{pid}] Resuming from {last_flush_idx}th read. {len(record_list)} reads remaining.")
        else:
            printmessage(f"[Process-{pid}] No flush file found. Starting from the beginning.")

    for read_idx, record in tqdm(enumerate(record_list), total=len(record_list)):
        oom_killer()
        read_idx += last_flush_idx
        read_id = record[0]
        seq = record[1].replace("T", "U")
        phred = record[2]

        cb_info_dict, dag_list, dag_dict = find_block_candidates(seq, phred, cb_bq_cutoff, spacer_kmer_ed_dict,
                                                                 skip_size_tolerance, cb_pad,
                                                                 cb_per_bb, indel_penalty, anchor_mismatch_penalty,
                                                                 spacer_edit_penalty,
                                                                 spacer_size, spacer_list, indel_dict,
                                                                 min_ideal_displacement_dict, anchor_list,
                                                                 score_converting_func, cb_size_tolerance,
                                                                 spacer_edit_tolerance, bb_size)

        if len(cb_info_dict) > 0:
            longest_path = dag_longest_path(dag_list)
            selected_cb = []
            total_score = 0
            for x, y in zip(longest_path[:-1], longest_path[1:]):
                if (x, y) in cb_info_dict:
                    selected_cb.append(cb_info_dict[(x, y)])
                total_score += dag_dict[(x, y)]

            selected_cb_df = pd.DataFrame(selected_cb,
                                          columns=["cb_idx", "start_pos", "end_pos", "pos_RM", "penalty", "score"])

            selected_cb_df["read_id"] = read_id
            selected_cb_df["total_score"] = total_score

            spacer_pos = np.unique(selected_cb_df[["start_pos", "end_pos"]].values.flatten())
            spacer_phred = [phred[x:x + spacer_size] for x in spacer_pos]

            if len(spacer_phred) > 0:
                spacer_phred = np.concatenate(spacer_phred)
                mean_spacer_phred = np.mean(spacer_phred)

                selected_cb_df["mean_spacer_phred"] = mean_spacer_phred
                selected_cb_df["start_pos"] = selected_cb_df["pos_RM"] - cb_pad
                selected_cb_df["end_pos"] = selected_cb_df["pos_RM"] + cb_pad + 1
                selected_cb_df["motif"] = selected_cb_df.apply(lambda x: seq[x["start_pos"]:x["end_pos"]], axis=1)
                selected_cb_df["bq"] = selected_cb_df.apply(lambda x: phred[x["start_pos"]:x["end_pos"]], axis=1)
                selected_cb_df = selected_cb_df[selected_cb_df["end_pos"] <= len(seq)]
                block_df_list.append(selected_cb_df)
            ## END IF
        ## END IF

        ## Periodic flush to reduce memory usage
        if (read_idx % flush_interval == 0 and read_idx != 0) or (read_idx == len_record - 1):
            if len(block_df_list) > 0:
                block_df_flush = pd.concat(block_df_list, axis=0).reset_index(drop=True)
                block_df_flush["bq_len"] = block_df_flush["bq"].apply(len)
                block_df_flush["motif_len"] = block_df_flush["motif"].apply(len)
                block_df_flush = block_df_flush[(block_df_flush["start_pos"] >= 0) &
                                                (block_df_flush["bq_len"] == cb_size) &
                                                (block_df_flush["motif_len"] == cb_size) &
                                                (block_df_flush["mean_spacer_phred"] >= cb_bq_cutoff)]

                flush_file = f"{flush_path}df_{pid}_{read_idx}.pkl"
                block_df_flush.to_pickle(flush_file)
                flush_file_list.append(flush_file)
                block_df_list = []
                del block_df_flush
                gc.collect()

    gc.collect()
    block_df_list = []
    if len(flush_file_list) > 0:
        for flush_file in flush_file_list:
            block_df = pd.read_pickle(flush_file)
            block_df_list.append(block_df)
        gc.collect()
        block_df = pd.concat(block_df_list, axis=0).reset_index(drop=True)
        del block_df_list
        block_df.to_pickle(f"{flush_path}df_{pid}.pkl")
    gc.collect()

    return None


def extract_blocks_from_read_list(input, output, indel_tolerance, indel_penalty, cb_size_tolerance,
                                  skip_size_tolerance, anchor_mismatch_penalty, spacer_edit_tolerance,
                                  max_read_length,
                                  spacer_edit_penalty, anchor_list, spacer_list, spacer_size, cb_pad,
                                  cb_per_bb, read_bq_cutoff, cb_bq_cutoff, flush_path, flush_interval, ncpu,
                                  resume, sample, **kwargs):
    spacer_list = [x.replace("T", "U") for x in spacer_list]
    anchor_list = [x.replace("T", "U") for x in anchor_list]
    indel_dict = get_integer_partition(indel_tolerance, cb_size_tolerance)
    spacer_kmer_ed_dict = {i: get_ed_kmers(kmer, spacer_edit_tolerance) for i, kmer in enumerate(spacer_list)}
    assert indel_tolerance >= cb_size_tolerance

    max_cb_penalty = anchor_mismatch_penalty + spacer_edit_penalty * spacer_edit_tolerance * 2 + indel_penalty * indel_tolerance
    score_converting_func = lambda x: 1 - (x / (2 * max_cb_penalty))
    cb_size = 2 * cb_pad + 1
    bb_size = cb_size * cb_per_bb + spacer_size
    min_ideal_displacement_dict = get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size)


    record_list = []
    with pysam.AlignmentFile(input, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm(total=input_bam.mapped) as pbar:
            for idx, record in enumerate(input_bam):
                qscore = mean_phred(np.array(record.query_qualities, dtype=int))
                if qscore >= read_bq_cutoff :
                    read_length = record.query_length
                    if read_length <= max_read_length and read_length >= kwargs["min_read_length"]:
                        record_tuple = (str(record.query_name), str(record.query_sequence),
                                        np.array(record.query_qualities), int(read_length))
                    record_list.append(record_tuple)
                pbar.update(1)

    if sample is not None:
        sample_idx = np.random.choice(len(record_list), sample, replace=False)
        record_list = [record_list[i] for i in sample_idx]

    record_list.sort(key=lambda x: x[3], reverse=True)
    record_cnt = len(record_list)

    record_split_dict = {i: [] for i in range(ncpu)}
    for i, fastq in enumerate(record_list):
        group = int(np.abs((i % (2 * ncpu)) - ncpu + 0.5) - 0.5)
        record_split_dict[group].append(fastq)
    del record_list
    gc.collect()

    proc_list = []

    for pid in range(ncpu):
        proc = mp.Process(target=extract_blocks_from_read_list_mp_worker,
                          args=(record_split_dict[pid], indel_penalty, cb_size_tolerance,
                                skip_size_tolerance, anchor_mismatch_penalty,
                                spacer_edit_tolerance, spacer_edit_penalty,
                                cb_pad, cb_per_bb, cb_bq_cutoff, indel_dict, spacer_kmer_ed_dict,
                                anchor_list, spacer_list, spacer_size, bb_size,
                                flush_path, pid, flush_interval, score_converting_func, cb_size,
                                min_ideal_displacement_dict, resume))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    block_df_list = []
    for pid in range(ncpu):
        try:
            block_df = pd.read_pickle(f"{flush_path}df_{pid}.pkl")
            block_df_list.append(block_df)
        except:
            printmessage(f"ERROR! PID {pid} did not return any result.")
    block_df = pd.concat(block_df_list, axis=0).reset_index(drop=True)
    del block_df_list
    gc.collect()

    print(block_df)

    ## print stats
    print("=============================================")
    printmessage(f"Total number of passed reads: {record_cnt:,}")
    printmessage(f"Total number of context blocks: {len(block_df):,}")
    printmessage(f"Context blocks per read: {len(block_df) / record_cnt:.2f}")
    print(block_df["score"].describe())
    print(block_df["penalty"].describe())
    print("=============================================")

    block_df.to_pickle(output)

    printmessage(f"Saved context blocks to {output}.")

    del block_df
    gc.collect()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--input", dest="input", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str, required=True)
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    parser.add_argument("--it", dest="indel_tolerance", type=int, default=1)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=3)
    parser.add_argument("--cst", dest="cb_size_tolerance", type=int, default=1)
    parser.add_argument("--kst", dest="skip_size_tolerance", type=int, default=2)
    parser.add_argument("--amp", dest="anchor_mismatch_penalty", type=int, default=6)
    parser.add_argument("--smt", dest="spacer_edit_tolerance", type=int, default=1)
    parser.add_argument("--smp", dest="spacer_edit_penalty", type=int, default=2)
    parser.add_argument("--ac", dest="anchor_list", type=str, nargs="+", default=["A", "A", "A"])
    parser.add_argument("--sp", dest="spacer_list", type=str, nargs="+",
                        default=["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"])
    parser.add_argument("--ss", dest="spacer_size", type=int, default=6)
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cb", dest="cb_per_bb", type=int, default=3)
    parser.add_argument("--rbq", dest="read_bq_cutoff", type=int, default=7)
    parser.add_argument("--cbq", dest="cb_bq_cutoff", type=int, default=0)
    parser.add_argument("--fi", dest="flush_interval", type=int, default=1000) # smaller->faster, larger->less memory
    parser.add_argument("--max", dest="max_read_length", type=int, default=1000)
    parser.add_argument("--min", dest="min_read_length", type=int, default=0)
    parser.add_argument("--sample", dest="sample", type=int, default=None)

    parser.add_argument("--cfg", dest="config", type=str, default=None)
    parser.add_argument("--resume", dest="resume", type=str, default=None, help="Continue from previous run. Provide the path to the previous output.")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as config_file:
            config_dict = json.load(config_file)
            for key, value in config_dict.items():
                setattr(args, key, value)
            printmessage(f"Loaded configuration from: {args.config}")

    assert len(args.anchor_list) == args.cb_per_bb
    assert len(args.spacer_list) == args.cb_per_bb + 1
    assert args.skip_size_tolerance >= args.cb_size_tolerance

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"ERROR! {args.resume} does not exist.")

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"ERROR! {args.input} does not exist.")
    if os.path.exists(args.output):
        raise FileExistsError(f"ERROR! {args.output} already exists.")

    base_path = os.path.dirname(args.output)
    if args.resume is not None:
        flush_path = args.resume

    else:
        flush_path = f"{base_path}/block_flush_{time.strftime('%Y%m%d%H%M%S')}/"
        os.makedirs(flush_path, exist_ok=True)
        # atexit.register(os.system, f"rm -r {flush_path}")

    args_dict = vars(args)
    extract_blocks_from_read_list(**args_dict, flush_path=flush_path)

    os.system(f"rm -r {flush_path}")
    return None


if __name__ == "__main__":
    main()
