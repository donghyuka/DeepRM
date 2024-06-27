import argparse
import gc
import glob
import json
import multiprocessing as mp
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import pysam
from Bio import Align
from tqdm import tqdm
import itertools

from utils.utils import mean_phred, printmessage, oom_killer
from sklearn.metrics import precision_recall_curve, auc
import random
import signal
import functools


def timeout(seconds=10):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator

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


def get_ideal_displacement(displacement, cb_size, bb_size):
    displacement_error = displacement - cb_size
    period = round(displacement_error / bb_size)
    period_non_negative = max(0, period)
    ideal_displacement = cb_size + period_non_negative * bb_size
    return ideal_displacement, period + 1


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


def validate_anchor(read, from_pos, to_pos, possible_indel_list, cb_pad, single_anchor,
                    indel_penalty, anchor_mismatch_penalty, displacement_error):
    query = read[from_pos:to_pos]
    anchor_candidate_list = [
        (displacement_error * indel_penalty + anchor_mismatch_penalty, 1, None, displacement_error)]
    ## penalty, missing_anchor, anchor_pos, total_indel

    for front_indel, back_indel in possible_indel_list:
        try:
            anchor_query = query[cb_pad + front_indel]
            if anchor_query == single_anchor:
                anchor_pos = from_pos + cb_pad + front_indel
                total_indel = np.abs(front_indel) + np.abs(back_indel)
                anchor_candidate_list.append((total_indel * indel_penalty, 0, anchor_pos, total_indel))
        except IndexError:
            continue

    anchor_candidate_list.sort(key=lambda x: (x[0], x[1]))
    return anchor_candidate_list[0]


def get_align_set(aligner, seq, spacer_dimer_list, spacer_dimer_size, min_score, max_iter = 10):
    align_dict = {}
    seq_len = len(seq)

    for spacer_dimer_idx, spacer_dimer in enumerate(spacer_dimer_list):
        seq_masked = seq
        for it in range(max_iter):
            alignments = aligner.align(spacer_dimer, seq_masked)
            if len(alignments) == 0:
                break
            for alignment in alignments:
                score = alignment.score
                if score > min_score:
                    aligned = alignment.aligned
                    align_start_pos_read = aligned[1][0][0] - aligned[0][0][0]
                    align_end_pos_read = aligned[1][-1][1] + spacer_dimer_size - aligned[0][-1][1]
                    if align_start_pos_read < 0 or align_end_pos_read > seq_len:
                        continue
                    if align_start_pos_read in align_dict:
                        prev_best_score = align_dict[align_start_pos_read][2]
                        if score > prev_best_score:
                            align_dict[align_start_pos_read] = (
                                align_start_pos_read, align_end_pos_read, score, spacer_dimer_idx)
                    else:
                        align_dict[align_start_pos_read] = (
                            align_start_pos_read, align_end_pos_read, score, spacer_dimer_idx)
                else:
                    break

            ## mask the aligned region with N
            for key in align_dict:
                start, end, _, _ = align_dict[key]
                seq_masked = seq_masked[:start] + "N" * (end - start) + seq_masked[end:]

    align_set = set(align_dict.values())

    return align_set


def get_aligner(spacer_match_score, spacer_mismatch_penalty, spacer_gap_open_penalty, spacer_gap_extend_penalty):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = spacer_match_score
    aligner.mismatch_score = -spacer_mismatch_penalty
    aligner.open_gap_score = -spacer_gap_open_penalty
    aligner.extend_gap_score = -spacer_gap_extend_penalty
    return aligner

@timeout(seconds=10)
def find_block_candidates(aligner, seq, cb_pad, indel_penalty, anchor_mismatch_penalty, spacer_dimer_size, spacer_dimer_list,
                          indel_dict, anchor_list, cb_size_tolerance, bb_size, cb_size, min_score):
    cb_info_dict = {}
    dag_list = []
    dag_dict = {}

    align_set = get_align_set(aligner, seq, spacer_dimer_list, spacer_dimer_size, min_score)

    if len(align_set) < 2:
        return cb_info_dict, dag_list, dag_dict

    for aligned_1 in align_set:
        start_1, end_1, score_1, spacer_dimer_idx_1 = aligned_1
        for aligned_2 in align_set:
            start_2, end_2, score_2, spacer_dimer_idx_2 = aligned_2
            if start_1 >= start_2:
                continue
            displacement = start_2 - end_1
            ideal_displacement, steps = get_ideal_displacement(displacement, cb_size, bb_size)
            displacement_error = np.abs(displacement - ideal_displacement)

            if displacement_error > cb_size_tolerance * steps:
                continue

            if steps == 1:
                penalty, missing_anchor, anchor_pos, total_indel = validate_anchor(seq, end_1, start_2,
                                                                                   indel_dict[displacement_error],
                                                                                   cb_pad,
                                                                                   anchor_list[spacer_dimer_idx_1 + 1],
                                                                                   indel_penalty,
                                                                                   anchor_mismatch_penalty,
                                                                                   displacement_error)
                total_score = max(1,score_1 + score_2 - penalty)

                if missing_anchor == 0:
                    cb_info_dict[(start_1, start_2)] = (start_1, end_2, anchor_pos, total_score, score_1, score_2)

                dag_list.append((start_1, start_2, total_score))
                dag_dict[(start_1, start_2)] = total_score


            else:
                penalty = displacement_error * indel_penalty + anchor_mismatch_penalty
                total_score = max(1,score_1 + score_2 - penalty)

            dag_list.append((start_1, start_2, total_score))
            dag_dict[(start_1, start_2)] = total_score

    return cb_info_dict, dag_list, dag_dict


def find_edge_blocks(aligner, seq, selected_cb_df, cb_size, cb_size_tolerance, indel_penalty, anchor_mismatch_penalty,
                     front_spacer, end_spacer, spacer_edit_tolerance, min_score, spacer_size, indel_dict, cb_pad,
                     anchor_list):

    selected_cb_df = selected_cb_df.sort_values("start_pos")
    chain_start = selected_cb_df["start_pos"].values[0]
    chain_end = selected_cb_df["end_pos"].values[-1]
    front_spacer_dimer_score = selected_cb_df["front_spacer_dimer_score"].values[0]
    end_spacer_dimer_score = selected_cb_df["end_spacer_dimer_score"].values[-1]
    seq_len = len(seq)

    front_search_start = max(0,chain_start - cb_size - spacer_size - cb_size_tolerance - spacer_edit_tolerance)
    front_search_end = max(0,chain_start - cb_size + cb_size_tolerance + spacer_edit_tolerance)
    end_search_start = min(seq_len,chain_end + cb_size - cb_size_tolerance - spacer_edit_tolerance)
    end_search_end = min(seq_len,chain_end + cb_size + spacer_size + cb_size_tolerance + spacer_edit_tolerance)

    front_search_query = seq[front_search_start:front_search_end]
    end_search_query = seq[end_search_start:end_search_end]

    cb_info_dict = {}
    cb_info_dict_front = {}
    cb_info_dict_end = {}

    if len(front_search_query) > 0:
        front_alignments = aligner.align(front_spacer, front_search_query)
        for alignment in front_alignments:
            score = alignment.score
            if score > min_score:
                aligned = alignment.aligned
                align_start_pos_read = aligned[1][0][0] - aligned[0][0][0] + front_search_start
                align_end_pos_read = aligned[1][-1][1] + spacer_size - aligned[0][-1][1] + front_search_start
                displacement_error = np.abs(chain_start - align_end_pos_read - cb_size)


                if align_start_pos_read < 0 or align_end_pos_read > seq_len:
                    continue

                if displacement_error <= cb_size_tolerance:
                    penalty, missing_anchor, anchor_pos, total_indel = validate_anchor(seq, align_end_pos_read,
                                                                                       chain_start,
                                                                                       indel_dict[displacement_error],
                                                                                       cb_pad, anchor_list[0],
                                                                                       indel_penalty, anchor_mismatch_penalty,
                                                                                       displacement_error)
                    if missing_anchor == 0:
                        total_score = max(1,score * 2 + front_spacer_dimer_score - penalty)
                        cb_info_dict_front[(align_start_pos_read, chain_start)] = (
                            align_start_pos_read, chain_start, anchor_pos, total_score, score, front_spacer_dimer_score)


    if len(end_search_query) > 0:
        end_alignments = aligner.align(end_spacer, end_search_query)
        for alignment in end_alignments:
            score = alignment.score
            if score > min_score:
                aligned = alignment.aligned
                align_start_pos_read = aligned[1][0][0] - aligned[0][0][0] + end_search_start
                align_end_pos_read = aligned[1][-1][1] + spacer_size - aligned[0][-1][1] + end_search_start
                displacement_error = np.abs(align_start_pos_read - chain_end - cb_size)

                if align_start_pos_read < 0 or align_end_pos_read > seq_len:
                    continue

                if displacement_error <= cb_size_tolerance:
                    penalty, missing_anchor, anchor_pos, total_indel = validate_anchor(seq, chain_end,
                                                                                       align_start_pos_read,
                                                                                       indel_dict[displacement_error],
                                                                                       cb_pad, anchor_list[-1],
                                                                                       indel_penalty, anchor_mismatch_penalty,
                                                                                       displacement_error)
                    if missing_anchor == 0:
                        total_score = max(1,score * 2 + end_spacer_dimer_score - penalty)
                        cb_info_dict_end[(chain_end, align_end_pos_read)] = (
                            chain_end, align_end_pos_read, anchor_pos, total_score, end_spacer_dimer_score, score)

    ## select the best front and end blocks
    if len(cb_info_dict_front) > 0:
        best_front = max(cb_info_dict_front.values(), key=lambda x: x[3])
        cb_info_dict[best_front[0], best_front[1]] = best_front

    if len(cb_info_dict_end) > 0:
        best_end = max(cb_info_dict_end.values(), key=lambda x: x[3])
        cb_info_dict[best_end[0], best_end[1]] = best_end

    return cb_info_dict


def dag_longest_path(edge_list):
    node_list = list(set([x[0] for x in edge_list] + [x[1] for x in edge_list]))

    dag = nx.DiGraph()
    dag.add_nodes_from(node_list)
    dag.add_weighted_edges_from(edge_list)
    longest_path = nx.dag_longest_path(dag, weight='weight')

    return longest_path


def extract_blocks_from_read_list_mp_worker(record_list, indel_penalty, cb_size_tolerance,
                                            anchor_mismatch_penalty, cb_pad, cb_bq_cutoff, indel_dict,
                                            anchor_list, spacer_list, spacer_size, bb_size, flush_path, pid,
                                            flush_interval, cb_size, resume, spacer_match_score, spacer_mismatch_penalty,
                                            spacer_gap_open_penalty, spacer_gap_extend_penalty, spacer_edit_tolerance,
                                            collect_list):
    aligner = get_aligner(spacer_match_score, spacer_mismatch_penalty, spacer_gap_open_penalty, spacer_gap_extend_penalty)
    min_score_monomer = spacer_match_score * spacer_size - max(spacer_edit_tolerance, spacer_gap_open_penalty+spacer_gap_extend_penalty) * spacer_edit_tolerance
    min_score_dimer = min_score_monomer * 2
    spacer_dimer_size = spacer_size * 2
    spacer_dimer_list = [spacer_list[2*i-1]+spacer_list[2*i] for i in range(1, len(spacer_list)//2)]

    block_df_list = []
    last_flush_idx = 0
    timeout_count = 0
    max_timeout = 3

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
    ## END IF

    for read_idx, record in enumerate(record_list):
        oom_killer()
        read_idx += last_flush_idx
        read_id = record[0]
        seq = record[1].replace("T", "U")
        phred = record[2]

        try:
            cb_info_dict, dag_list, dag_dict = find_block_candidates(aligner, seq, cb_pad, indel_penalty,
                                                                     anchor_mismatch_penalty,
                                                                     spacer_dimer_size, spacer_dimer_list, indel_dict,
                                                                     anchor_list, cb_size_tolerance, bb_size, cb_size,
                                                                     min_score_dimer)
        except TimeoutError:
            timeout_count += 1
            if timeout_count >= max_timeout:
                printmessage(f"[Process-{pid}] Timeout limit reached. Exiting.")
                break
            continue

        if len(cb_info_dict) > 0:
            longest_path = dag_longest_path(dag_list)
            selected_cb = []
            total_score = 0
            for x, y in zip(longest_path[:-1], longest_path[1:]):
                if (x, y) in cb_info_dict:
                    selected_cb.append(cb_info_dict[(x, y)])
                total_score += dag_dict[(x, y)]

            selected_cb_df = pd.DataFrame(selected_cb,
                                          columns=["start_pos", "end_pos", "pos_RM",
                                                   "score", "front_spacer_dimer_score", "end_spacer_dimer_score"])


            if len(selected_cb_df) > 0:

                edge_cb_dict = find_edge_blocks(aligner, seq, selected_cb_df, cb_size, cb_size_tolerance,
                                                indel_penalty, anchor_mismatch_penalty, spacer_list[0], spacer_list[-1],
                                                spacer_edit_tolerance, min_score_monomer, spacer_size, indel_dict, cb_pad,
                                                anchor_list)

                if len(edge_cb_dict) > 0:
                    edge_cb_df = pd.DataFrame(edge_cb_dict.values(),
                                              columns=["start_pos", "end_pos", "pos_RM",
                                                       "score", "front_spacer_dimer_score", "end_spacer_dimer_score"])

                    selected_cb_df = pd.concat([selected_cb_df, edge_cb_df], axis=0).reset_index(drop=True)

                selected_cb_df.drop(["front_spacer_dimer_score", "end_spacer_dimer_score"], axis=1, inplace=True)
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
        ## END IF

    if len(block_df_list) > 0:
        block_df = pd.concat(block_df_list, axis=0).reset_index(drop=True)
        collect_list.append(block_df)

    return None


def extract_blocks_from_read_list(record_list, input, output, indel_tolerance, indel_penalty, cb_size_tolerance,
                                  anchor_mismatch_penalty, max_read_length, anchor_list, spacer_list, spacer_size, cb_pad,
                                  cb_per_bb, read_bq_cutoff, cb_bq_cutoff, flush_path, flush_interval, ncpu,
                                  resume, sample, spacer_match_score, spacer_mismatch_penalty, spacer_gap_open_penalty,
                                  spacer_gap_extend_penalty, spacer_edit_tolerance, **kwargs):

    spacer_list = [x.replace("T", "U") for x in spacer_list]
    anchor_list = [x.replace("T", "U") for x in anchor_list]
    indel_dict = get_integer_partition(indel_tolerance, cb_size_tolerance)
    assert indel_tolerance >= cb_size_tolerance
    cb_size = 2 * cb_pad + 1
    bb_size = cb_size * cb_per_bb + spacer_size



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
    man = mp.Manager()
    collect_list = man.list()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_blocks_from_read_list_mp_worker,
                          args=(record_split_dict[pid], indel_penalty, cb_size_tolerance,
                                anchor_mismatch_penalty, cb_pad, cb_bq_cutoff, indel_dict,
                                anchor_list, spacer_list, spacer_size, bb_size, flush_path, pid,
                                flush_interval, cb_size, resume, spacer_match_score, spacer_mismatch_penalty,
                                spacer_gap_open_penalty, spacer_gap_extend_penalty, spacer_edit_tolerance,
                                collect_list))

        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()

    if len(collect_list) > 0:
        block_df = pd.concat(collect_list, axis=0).reset_index(drop=True)
        gc.collect()
    else:
        return None

    return block_df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--input", dest="input", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str, required=True)
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    parser.add_argument("--it", dest="indel_tolerance", type=int, default=15)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=30)
    parser.add_argument("--cst", dest="cb_size_tolerance", type=int, default=15)
    parser.add_argument("--amp", dest="anchor_mismatch_penalty", type=int, default=300)
    parser.add_argument("--ss", dest="spacer_size", type=int, default=18)
    parser.add_argument("--cp", dest="cb_pad", type=int, default=12)
    parser.add_argument("--cb", dest="cb_per_bb", type=int, default=1)
    parser.add_argument("--rbq", dest="read_bq_cutoff", type=int, default=7)
    parser.add_argument("--cbq", dest="cb_bq_cutoff", type=int, default=0)
    parser.add_argument("--fi", dest="flush_interval", type=int, default=1000)  # smaller->faster, larger->less memory
    parser.add_argument("--max", dest="max_read_length", type=int, default=1000)
    parser.add_argument("--min", dest="min_read_length", type=int, default=0)
    parser.add_argument("--sample", dest="sample", type=int, default=None)
    parser.add_argument("--sms", dest="spacer_match_score", type=int, default=3)
    parser.add_argument("--smm", dest="spacer_mismatch_penalty", type=int, default=2)
    parser.add_argument("--sgo", dest="spacer_gap_open_penalty", type=int, default=3)
    parser.add_argument("--sge", dest="spacer_gap_extend_penalty", type=int, default=2)
    parser.add_argument("--set", dest="spacer_edit_tolerance", type=int, default=5)
    parser.add_argument("--cfg", dest="config", type=str, default=None)

    parser.add_argument("--resume", dest="resume", type=str, default=None,
                        help="Continue from previous run. Provide the path to the previous output.")

    parser.add_argument("--ac", dest="anchor_list", type=str, nargs="+", default=[
        "G",
        "A",
        "G",
        "C",
        "G",
        "A",

    ])
    parser.add_argument("--sp", dest="spacer_list", type=str, nargs="+",
                        default=[
                            "GUUUCCGGAAGAACCUGC", "GGAACAGUGGCUGCCAUU",
                            "GCAGACUGUGCACCCUUU", "GCCAGAGUAGUCCGACAU",
                            "CCCCAGGACCACUGCUGU", "GUCUGAAGCGUAGGCUAG",
                            "CGAACGUCGAUCUGGUAG", "GCCTCCAUUGGCTCTGCA",
                            "AGAAGTGAGUGTAAAGCG", "GUCCGACAUGTGGACACC",
                            "CATGGCCCTCTCCCAUUG", "GUCAGGGATGTGACTGAG",
                        ])

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as config_file:
            config_dict = json.load(config_file)
            for key, value in config_dict.items():
                setattr(args, key, value)
            printmessage(f"Loaded configuration from: {args.config}")

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"ERROR! {args.resume} does not exist.")

    return args



def read_bam(bam_path, args):
    read_dict = {}
    align_pair_dict = {}
    with pysam.AlignmentFile(bam_path, "rb", threads=32) as bam:
        for record in tqdm(bam, total=bam.mapped):
            qscore = mean_phred(np.array(record.query_qualities, dtype=int))
            if qscore >= args["read_bq_cutoff"]:
                read_length = record.query_length
                if read_length <= args["max_read_length"] and read_length >= args["min_read_length"]:
                    read_dict[record.query_name] = record
    for id, read in tqdm(read_dict.items()):
        if read is not None:
            aligned_pair = read.get_aligned_pairs()
            aligned_pair_as_dict = {x[0]: x[1] for x in aligned_pair}
            for i in range(read.query_length):
                if i not in aligned_pair_as_dict:
                    aligned_pair_as_dict[i] = None
            align_pair_dict[id] = aligned_pair_as_dict
        else:
            align_pair_dict[id] = None
    return align_pair_dict

def check_if_correct_position(block_df):
    CORRECT_POSITION = [61+61*n+30 for n in range(6)]
    block_df["ref_anchor"] = block_df.apply(lambda x: x["align_pairs"][x["pos_RM"]], axis = 1)
    block_df["correct"] = block_df["ref_anchor"].apply(lambda x: x in CORRECT_POSITION)
    return None


def main():

    indel_tolerance_list = [5,10,15,20]
    indel_penalty_list = [3,10,30,100,300]
    cb_size_tolerance_list = [5,10,15,20]
    anchor_mismatch_penalty_list = [3,10,30,100,300,1000]
    spacer_match_score_list = [1,2,3,5,7]
    spacer_mismatch_penalty_list = [1,2,3,5,7]
    spacer_gap_open_penalty_list = [1,2,3,5,7]
    spacer_gap_extend_penalty_list = [1,2,3,5,7]
    spacer_edit_tolerance_list = [5,10,15,20,25]

    hyperparam_combination = itertools.product(indel_tolerance_list, indel_penalty_list, cb_size_tolerance_list,
                                               anchor_mismatch_penalty_list, spacer_match_score_list,
                                               spacer_mismatch_penalty_list, spacer_gap_open_penalty_list,
                                               spacer_gap_extend_penalty_list, spacer_edit_tolerance_list)

    ## Randomly shuffle the hyperparam_combination
    hyperparam_combination = list(hyperparam_combination)
    random.shuffle(hyperparam_combination)

    args = parse_args()

    args_dict = vars(args)

    record_list = []
    with pysam.AlignmentFile(args_dict["input"], "rb", check_sq=False, threads=args_dict["ncpu"]) as input_bam:
        for idx, record in enumerate(input_bam):
            qscore = mean_phred(np.array(record.query_qualities, dtype=int))
            if qscore >= args_dict["read_bq_cutoff"]:
                read_length = record.query_length
                if read_length <= args_dict["max_read_length"] and read_length >= args_dict["min_read_length"]:
                    record_tuple = (str(record.query_name), str(record.query_sequence),
                                    np.array(record.query_qualities), int(read_length))
                record_list.append(record_tuple)

    align_pair_dict = read_bam(args_dict["input"], args_dict)
    args_dict["record_list"] = record_list
    total = len(indel_tolerance_list) * len(indel_penalty_list) * len(cb_size_tolerance_list) * len(anchor_mismatch_penalty_list) * len(spacer_match_score_list) * len(spacer_mismatch_penalty_list) * len(spacer_gap_open_penalty_list) * len(spacer_gap_extend_penalty_list) * len(spacer_edit_tolerance_list)

    write_file = open(args_dict["output"], "a")
    header = ["id_tol", "id_pen", "cs_tol", "am_pen", "sm_scr", "sm_pen", "sgo_pen", "sge_pen", "se_tol", "pr_auc", "max_rec", "max_prec"]
    write_file.write("\t".join(header) + "\n")
    write_file.flush()

    flush_interval = 10

    for index, hyperparam in tqdm(enumerate(hyperparam_combination), total=total):

        if index % flush_interval == 0:
            write_file.flush()
            gc.collect()

        args_dict["indel_tolerance"] = hyperparam[0]
        args_dict["indel_penalty"] = hyperparam[1]
        args_dict["cb_size_tolerance"] = hyperparam[2]
        args_dict["anchor_mismatch_penalty"] = hyperparam[3]
        args_dict["spacer_match_score"] = hyperparam[4]
        args_dict["spacer_mismatch_penalty"] = hyperparam[5]
        args_dict["spacer_gap_open_penalty"] = hyperparam[6]
        args_dict["spacer_gap_extend_penalty"] = hyperparam[7]
        args_dict["spacer_edit_tolerance"] = hyperparam[8]

        if args_dict["cb_size_tolerance"] > args_dict["indel_tolerance"]:
            continue

        try:
            block_df = extract_blocks_from_read_list(**args_dict, flush_path="")

            if block_df is None:
                continue

            block_df["align_pairs"] = block_df["read_id"].map(align_pair_dict)
            block_df = block_df.dropna()
            check_if_correct_position(block_df)
            x = block_df["score"].values
            y = block_df["correct"].values
            precision, recall, _ = precision_recall_curve(y, x)
            recall = recall * (np.sum(y) / (6*15475))
            pr_auc = auc(recall, precision)
            max_recall = recall[0]
            max_precision = precision[-2]
            write_file.write("\t".join([str(x) for x in hyperparam] + [f"{pr_auc:.5f}", f"{max_recall:.5f}", f"{max_precision:.5f}"]) + "\n")

        except:
            continue

    return None



if __name__ == "__main__":
    main()