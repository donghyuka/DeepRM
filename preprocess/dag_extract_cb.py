import argparse
import atexit
import gc
import os
import pysam
import time

import itertools as it
import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd
import polyleven as pl

from tqdm import tqdm
from collections import defaultdict


## Step 1: Extract all possible blocks from the read.
## Step 2: Build a DAG from the blocks.
## Step 3: Find the longest path in the DAG.
## Step 4: Extract the sequence from the longest path.

## Tolerances (Scoring parameters):
## 1. The number of mismatches in ligation linker.
## 2. The discrepancy between expected and observed length of the block. (+ indel constraint)
## 3. The discrepancy between expected and observed number of anchors in the block.
## 4. The spacing between the blocks.

## This code is not really built for heavy lifting. Well how about several million reads? lol godspeed.

def get_integer_partition(indel_tolerance, spacing_tolerance):
    indel_list = list(it.product(range(indel_tolerance + 1), range(indel_tolerance + 1)))
    indel_dict = defaultdict(list)
    for insertion, deletion in indel_list:
        offset = insertion - deletion
        indel = insertion + deletion
        if np.abs(insertion - deletion) <= spacing_tolerance and indel <= indel_tolerance:
            indel_dict[offset].append((insertion, deletion))
    return indel_dict


def get_offset_list(insertion, deletion, cb_count, cb_pad=5):
    offset_blocks = range(cb_count + 1)
    insertion_cwr = list(it.combinations_with_replacement(offset_blocks, insertion))
    deletion_cwr = list(it.combinations_with_replacement(offset_blocks, deletion))
    indel_cwr_product = list(it.product(insertion_cwr, deletion_cwr))
    offset_list = []

    for insertion_list, deletion_list in indel_cwr_product:
        deletion_counts = [deletion_list.count(x) for x in range(cb_count + 1)]
        zeros = np.zeros(cb_count + 1, dtype=int)

        if np.max(deletion_counts) > cb_pad * 2 + 1:
            continue
        if deletion_counts[0] > cb_pad:
            continue
        if deletion_counts[-1] > cb_pad:
            continue

        for insertion in insertion_list:
            zeros[insertion:] += 1
        for deletion in deletion_list:
            zeros[deletion:] -= 1

        offset_list.append(zeros[:-1])

    offset_list_unique = list(set([tuple(x) for x in offset_list]))
    offset_list_unique = [np.array(x) for x in offset_list_unique]
    return offset_list_unique


def get_offset_dict(indel_dict, cb_count):
    offset_dict = {}
    for _, indel_list in indel_dict.items():
        for insertion, deletion in indel_list:
            offset_list = get_offset_list(insertion, deletion, cb_count)
            offset_dict[(insertion, deletion)] = offset_list
    return offset_dict


def get_ed_kmers(kmer, linker_mismatch_tolerance):
    nucs = "ACGU"
    possible_nucs = ["".join(x) for x in it.product(nucs, repeat=len(kmer))]
    kmer_ed_dict = defaultdict(list)
    for possible_kmer in possible_nucs:
        ed = pl.levenshtein(kmer, possible_kmer, linker_mismatch_tolerance)
        kmer_ed_dict[ed].append(possible_kmer)
    kmer_ed_dict[linker_mismatch_tolerance + 1] = []
    return kmer_ed_dict


def get_kmer_dict(read, k):
    kmer_dict = defaultdict(list)
    for i in range(len(read) - k + 1):
        kmer = read[i:i + k]
        kmer_dict[kmer].append(i)
    return kmer_dict


def get_kmer_tuple(linker_mismatch_tolerance, linker_kmer_ed_dict):
    kmer_tuple_list = []
    for total_mismatch in range(linker_mismatch_tolerance + 1):
        for front_mismatch in range(total_mismatch + 1):
            back_mismatch = total_mismatch - front_mismatch
            for from_kmer, to_kmer in it.product(linker_kmer_ed_dict[front_mismatch],
                                                 linker_kmer_ed_dict[back_mismatch]):
                kmer_tuple_list.append((from_kmer, to_kmer, front_mismatch, back_mismatch))
    return kmer_tuple_list


def find_block_candidates(read_idx, seq, phred, bq_cutoff, dimer_ed_dict, linker_front_ed_dict, linker_back_ed_dict,
                          skip_size_tolerance, cb_pad, ideal_anchor_pos,
                          cb_count, indel_penalty, anchor_mismatch_penalty, linker_mismatch_penalty,
                          linker_front, linker_back, indel_dict, offset_dict, anchor_list,
                          score_converting_func, bb_size_tolerance, linker_mismatch_tolerance,
                          edge_linker_mismatch_tolerance):
    dimer_size = len(linker_front) + len(linker_back)
    bb_size = (2 * cb_pad + 1) * cb_count + dimer_size
    kmer_pos_dict = get_kmer_dict(seq, dimer_size)
    len_linker_front = len(linker_front)
    len_linker_back = len(linker_back)
    front_linker_kmer_pos_dict = get_kmer_dict(seq, len_linker_front)

    if len_linker_front == len_linker_back:
        back_linker_kmer_pos_dict = front_linker_kmer_pos_dict
    else:
        back_linker_kmer_pos_dict = get_kmer_dict(seq, len_linker_back)

    dag_list = []  ## Format: [from_pos, to_pos, score]
    dag_dict = {}  ## Format: {(from_pos, to_pos): score}
    bb_info_dict = {}  ## Format: {(from_pos, to_pos): [from_pos,to_pos,anchor_pos,score]}
    node_dict = {}  ## Format: {pos: mismatch}

    kmer_tuple_list = get_kmer_tuple(linker_mismatch_tolerance, dimer_ed_dict)

    for from_kmer, to_kmer, front_mismatch, back_mismatch in kmer_tuple_list:
        kmer_mismatch = front_mismatch + back_mismatch
        for from_pos, to_pos in it.product(kmer_pos_dict[from_kmer], kmer_pos_dict[to_kmer]):
            if from_pos >= to_pos:
                continue

            displacement = to_pos - from_pos
            bb_between = int(np.round(displacement / bb_size))

            if bb_between < 1:
                continue

            ideal_displacement = bb_between * bb_size
            displacement_tolerance = skip_size_tolerance * bb_between
            displacement_error = displacement - ideal_displacement
            displacement_error_abs = np.abs(displacement_error)

            if displacement_error_abs > displacement_tolerance:
                continue

            linker_penalty = linker_mismatch_penalty * kmer_mismatch

            if bb_between == 1 and displacement_error_abs <= bb_size_tolerance:
                bb_seq = seq[from_pos + dimer_size:to_pos]
                possible_indel_list = indel_dict[displacement_error]

                anchor_indel_penalty, _, anchor_pos_list = validate_anchor(bb_seq, possible_indel_list, offset_dict,
                                                                           ideal_anchor_pos,
                                                                           cb_count, anchor_list, indel_penalty,
                                                                           anchor_mismatch_penalty,
                                                                           displacement_error_abs)


            else:
                anchor_pos_list = []
                anchor_indel_penalty = indel_penalty * displacement_error_abs + anchor_mismatch_penalty * cb_count

            penalty = linker_penalty + anchor_indel_penalty
            score = score_converting_func(penalty)

            if len(anchor_pos_list) > 0:
                bb_info_dict[(from_pos, to_pos)] = [read_idx, from_pos, to_pos, anchor_pos_list, penalty, score]

            dag_list.append((from_pos, to_pos, score))
            dag_dict[(from_pos, to_pos)] = score
            node_dict[from_pos] = front_mismatch
            node_dict[to_pos] = back_mismatch

    front_linker_pos_ed_dict = {x: [] for x in range(edge_linker_mismatch_tolerance + 1)}
    back_linker_pos_ed_dict = {x: [] for x in range(edge_linker_mismatch_tolerance + 1)}

    for i in range(edge_linker_mismatch_tolerance + 1):
        front_linker_kmer_list = linker_front_ed_dict[i]
        back_linker_kmer_list = linker_back_ed_dict[i]
        front_linker_pos_list = np.concatenate(
            [front_linker_kmer_pos_dict[kmer] for kmer in front_linker_kmer_list]).astype(int)
        back_linker_pos_list = np.concatenate(
            [back_linker_kmer_pos_dict[kmer] for kmer in back_linker_kmer_list]).astype(int)
        front_linker_pos_ed_dict[i] = front_linker_pos_list
        back_linker_pos_ed_dict[i] = back_linker_pos_list

    for pos, mismatch in node_dict.items():
        max_allowed_mismatch = min(linker_mismatch_tolerance - mismatch, edge_linker_mismatch_tolerance)
        for edge_mismatch in range(max_allowed_mismatch + 1):
            front_linker_pos_list = front_linker_pos_ed_dict[edge_mismatch]
            back_linker_pos_list = back_linker_pos_ed_dict[edge_mismatch]
            front_linker_pos_list = front_linker_pos_list[front_linker_pos_list < pos]
            back_linker_pos_list = back_linker_pos_list[back_linker_pos_list > pos]

            for front_linker_pos in front_linker_pos_list:
                displacement = pos - front_linker_pos + len_linker_back
                displacement_error = displacement - bb_size
                displacement_error_abs = np.abs(displacement_error)

                if displacement_error_abs > bb_size_tolerance:
                    continue

                from_pos = front_linker_pos - len_linker_back
                to_pos = pos + len_linker_back

                if to_pos >= len(seq):
                    continue

                bb_seq = seq[from_pos + dimer_size:to_pos]
                possible_indel_list = indel_dict[displacement_error]
                linker_penalty = linker_mismatch_penalty * (mismatch + edge_mismatch)

                anchor_indel_penalty, _, anchor_pos_list = validate_anchor(bb_seq, possible_indel_list, offset_dict,
                                                                           ideal_anchor_pos,
                                                                           cb_count, anchor_list, indel_penalty,
                                                                           anchor_mismatch_penalty,
                                                                           displacement_error_abs)

                if len(anchor_pos_list) == 0:
                    continue

                penalty = linker_penalty + anchor_indel_penalty
                score = score_converting_func(penalty)
                bb_info_dict[(from_pos, to_pos)] = [read_idx, from_pos, to_pos, anchor_pos_list, penalty, score]

                dag_list.append((from_pos, to_pos, score))
                dag_dict[(from_pos, to_pos)] = score

            for back_linker_pos in back_linker_pos_list:
                displacement = back_linker_pos - pos
                displacement_error = displacement - bb_size
                displacement_error_abs = np.abs(displacement_error)

                if displacement_error > bb_size_tolerance:
                    continue

                from_pos = pos
                to_pos = back_linker_pos
                bb_seq = seq[from_pos + dimer_size:to_pos]
                possible_indel_list = indel_dict[displacement_error]
                linker_penalty = linker_mismatch_penalty * (mismatch + edge_mismatch)

                anchor_indel_penalty, _, anchor_pos_list = validate_anchor(bb_seq, possible_indel_list, offset_dict,
                                                                           ideal_anchor_pos,
                                                                           cb_count, anchor_list, indel_penalty,
                                                                           anchor_mismatch_penalty,
                                                                           displacement_error_abs)

                if len(anchor_pos_list) == 0:
                    continue

                penalty = linker_penalty + anchor_indel_penalty
                score = score_converting_func(penalty)
                bb_info_dict[(from_pos, to_pos)] = [read_idx, from_pos, to_pos, anchor_pos_list, penalty, score]

                dag_list.append((from_pos, to_pos, score))
                dag_dict[(from_pos, to_pos)] = score

    return bb_info_dict, dag_list, dag_dict


def validate_anchor(bb_seq, possible_indel_list, offset_dict, ideal_anchor_pos, cb_count, anchor_list, indel_penalty,
                    anchor_mismatch_penalty, displacement_error_abs):
    default_indel_penalty = indel_penalty * displacement_error_abs
    default_anchor_penalty = anchor_mismatch_penalty * cb_count

    anchor_candidate_list = [(default_indel_penalty + default_anchor_penalty, default_anchor_penalty, [])]

    for indel in possible_indel_list:
        possible_offset_list = offset_dict[indel]
        indel_penalty = indel_penalty * sum(indel)
        for offset_arr in possible_offset_list:
            actual_anchor_pos = ideal_anchor_pos + offset_arr
            valid_anchor_pos = []
            for i, pos in enumerate(actual_anchor_pos):
                if len(bb_seq) <= pos:
                    print(len(bb_seq), displacement_error_abs, possible_indel_list, actual_anchor_pos, offset_arr)
                    continue
                if bb_seq[pos] == anchor_list[i]:
                    valid_anchor_pos.append(pos)
            valid_anchor_pos = np.array(valid_anchor_pos)
            anchor_mismatch = len(anchor_list) - len(valid_anchor_pos)
            anchor_penalty = anchor_mismatch_penalty * anchor_mismatch
            anchor_candidate_list.append((indel_penalty + anchor_penalty, anchor_penalty, valid_anchor_pos))

    anchor_candidate_list.sort(key=lambda x: (x[0], x[1]), reverse=False)
    # best_penalty, best_anchor_penalty, best_anchor_pos
    return anchor_candidate_list[0]


def dag_longest_path(edge_list):
    node_list = []

    for u, v, score in edge_list:
        node_list.append(u)
        node_list.append(v)
    node_list = list(set(node_list))

    dag = nx.DiGraph()
    dag.add_nodes_from(node_list)
    dag.add_weighted_edges_from(edge_list)
    longest_path = nx.dag_longest_path(dag, weight='weight')
    longest_path_score = sum([dag[u][v]['weight'] for u, v in zip(longest_path[:-1], longest_path[1:])])

    ## Free up memory
    del dag

    return longest_path, longest_path_score


def get_linker_bq(start_pos, end_pos, front_len, back_len, phred):
    front_bq = phred[start_pos:start_pos + front_len]
    back_bq = phred[end_pos - back_len:end_pos]
    mean_bq = np.mean(np.concatenate([front_bq, back_bq]))
    return mean_bq

def extract_blocks_from_read_list_mp_worker(fastq_list, linker_front, linker_back, linker_mismatch_tolerance,
                                            edge_linker_mismatch_tolerance, skip_size_tolerance,
                                            cb_pad, cb_count, bq_cutoff,
                                            indel_penalty, anchor_mismatch_penalty,
                                            linker_mismatch_penalty, indel_tolerance, bb_size_tolerance,
                                            anchor_list, return_list, flush_path, pid, flush_interval=1000):
    # write_fastq(fastq_list, f"{flush_path}/fastq_{pid}.fastq")
    # fastq_list = SeqIO.index(f"{flush_path}/fastq_{pid}.fastq", "fastq")
    # fastq_list = [x for x in fastq_list.values()]
    # gc.collect()

    indel_dict = get_integer_partition(indel_tolerance, bb_size_tolerance)
    offset_dict = get_offset_dict(indel_dict, cb_count)
    ideal_anchor_pos = np.array([cb_pad + (2 * cb_pad + 1) * i for i in range(cb_count)])
    max_cb_penalty = anchor_mismatch_penalty * cb_count + linker_mismatch_tolerance * linker_mismatch_penalty + indel_penalty * indel_tolerance
    score_converting_func = lambda x: 1 - (x / (2 * max_cb_penalty))

    dimer_ed_dict = get_ed_kmers(linker_back + linker_front, linker_mismatch_tolerance)
    linker_front_ed_dict = get_ed_kmers(linker_front, edge_linker_mismatch_tolerance)
    linker_back_ed_dict = get_ed_kmers(linker_back, edge_linker_mismatch_tolerance)

    flush_file_list = []
    len_fastq = len(fastq_list)

    seq_dict = {}
    readid_dict = {}
    phred_dict = {}
    bb_info_dict_selected = {}
    cb_len = 2 * cb_pad + 1

    for read_idx, fastq_record in tqdm(enumerate(fastq_list), total=len_fastq):

        # seq = fastq_record.seq
        # phred = np.array(fastq_record.letter_annotations["phred_quality"])
        # read_id = fastq_record.id
        # seq = str(seq.replace("T", "U"))

        read_id = fastq_record[0]
        seq = fastq_record[1]
        phred = np.array(fastq_record[2])
        seq = str(seq.replace("T", "U"))

        bb_info_dict, dag_list, dag_dict = find_block_candidates(read_idx, seq, phred, bq_cutoff, dimer_ed_dict,
                                                                 linker_front_ed_dict, linker_back_ed_dict,
                                                                 skip_size_tolerance, cb_pad, ideal_anchor_pos,
                                                                 cb_count, indel_penalty, anchor_mismatch_penalty,
                                                                 linker_mismatch_penalty,
                                                                 linker_front, linker_back, indel_dict, offset_dict,
                                                                 anchor_list,
                                                                 score_converting_func, bb_size_tolerance,
                                                                 linker_mismatch_tolerance,
                                                                 edge_linker_mismatch_tolerance)

        if len(bb_info_dict) == 0:
            continue

        longest_path, longest_path_score = dag_longest_path(dag_list)

        selected_cb = []
        for x, y in zip(longest_path[:-1], longest_path[1:]):
            if (x, y) in bb_info_dict:
                selected_cb.append((x, y))

        if len(selected_cb) == 0:
            continue

        for key in selected_cb:
            new_key = (read_idx,) + key
            bb_info_dict_selected[new_key] = bb_info_dict[key]

        seq_dict[read_idx] = seq
        readid_dict[read_idx] = read_id
        phred_dict[read_idx] = phred

        if (read_idx % flush_interval == 0 and read_idx != 0) or (read_idx == len_fastq - 1):
            if len(bb_info_dict_selected) == 0:
                continue

            block_df_flush = pd.DataFrame.from_dict(bb_info_dict_selected, orient="index",
                                                    columns=["read_idx", "start_pos", "end_pos", "pos_RM", "penalty",
                                                             "score"])
            block_df_flush["start_pos"] += len(linker_back)
            block_df_flush["pos_RM"] += (len(linker_front) + block_df_flush["start_pos"])
            block_df_flush["anchor_count"] = block_df_flush["pos_RM"].apply(len)

            block_df_flush["id"] = block_df_flush["read_idx"].apply(lambda x: readid_dict[x])
            block_df_flush["motif"] = block_df_flush["read_idx"].apply(lambda x: seq_dict[x])
            block_df_flush["bq"] = block_df_flush["read_idx"].apply(lambda x: phred_dict[x])
            block_df_flush["seq_len"] = block_df_flush["motif"].apply(len)
            block_df_flush = block_df_flush.explode("pos_RM")

            block_df_flush["start_pos"] = block_df_flush["pos_RM"] - cb_pad
            block_df_flush["end_pos"] = block_df_flush["pos_RM"] + cb_pad + 1

            block_df_flush = block_df_flush[block_df_flush["end_pos"] <= block_df_flush["seq_len"]]
            block_df_flush["motif"] = block_df_flush.apply(lambda x: x["motif"][x["start_pos"]:x["end_pos"]], axis=1)
            block_df_flush["bq"] = block_df_flush.apply(lambda x: x["bq"][x["start_pos"]:x["end_pos"]], axis=1)

            block_df_flush["seq_len"] = block_df_flush["motif"].apply(len)
            block_df_flush["phred_len"] = block_df_flush["phred"].apply(len)
            block_df_flush = block_df_flush[
                (block_df_flush["seq_len"] == cb_len) & (block_df_flush["phred_len"] == cb_len)].copy().reset_index(
                drop=True)

            block_df_flush.drop(columns=["read_idx", "seq_len", "phred_len", "phred"], inplace=True)

            flush_file = f"{flush_path}df_{pid}_{read_idx}.pkl"
            block_df_flush.to_pickle(flush_file)
            flush_file_list.append(flush_file)

            del block_df_flush
            seq_dict = {}
            readid_dict = {}
            phred_dict = {}
            bb_info_dict_selected = {}
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
        return_list[pid] = block_df
    gc.collect()

    return None


def extract_blocks_from_read_list(inputpath, outpath, linker_front, linker_back, linker_mismatch_tolerance,
                                  bb_size_tolerance, cb_pad, cb_count, bq_cutoff,
                                  indel_penalty, anchor_mismatch_penalty, linker_mismatch_penalty,
                                  edge_linker_mismatch_tolerance, skip_size_tolerance, indel_tolerance, anchor_list,
                                  flushpath, ncpu):
    record_list = []
    with pysam.AlignmentFile(inputpath, "rb", check_sq=False) as input_bam:
        for record in tqdm(input_bam, total=input_bam.count()):
            if record.get_tag("qs") > 7:
                record_tuple = (record.query_name, record.query_sequence, record.query_qualities, record.query_length)
                record_list.append(record_tuple)
    record_list.sort(key=lambda x: x[3], reverse=True)

    fastq_split_dict = {i: [] for i in range(ncpu)}
    for i, fastq in enumerate(record_list):
        group = int(np.abs((i % (2 * ncpu)) - ncpu + 0.5) - 0.5)
        fastq_split_dict[group].append(fastq)
    del record_list
    gc.collect()

    proc_list = []
    man = mp.Manager()
    return_dict = man.dict()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_blocks_from_read_list_mp_worker,
                          args=(fastq_split_dict[pid], linker_front, linker_back, linker_mismatch_tolerance,
                                edge_linker_mismatch_tolerance, skip_size_tolerance,
                                cb_pad, cb_count, bq_cutoff,
                                indel_penalty, anchor_mismatch_penalty,
                                linker_mismatch_penalty, indel_tolerance, bb_size_tolerance,
                                anchor_list, return_dict, flushpath, pid))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return_dict = dict(return_dict)
    man.shutdown()
    block_df = []
    for pid in range(ncpu):
        try:
            block_df.append(return_dict[pid])
        except:
            print(f"ERROR! PID {pid} did not return any result.")
    block_df = pd.concat(block_df, axis=0).reset_index(drop=True)
    print(block_df)
    block_df.to_pickle(outpath)
    gc.collect()

    return None


def parse_args_lenient():
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--exp", dest="exp", type=str, required=True, nargs="+")
    parser.add_argument("--fl", dest="linker_front", type=str, default="CGAC")
    parser.add_argument("--bl", dest="linker_back", type=str, default="AGUC")
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cc", dest="cb_count", type=int, default=3)
    parser.add_argument("--lt", dest="linker_mismatch_tolerance", type=int, default=4)
    parser.add_argument("--et", dest="edge_linker_mismatch_tolerance", type=int, default=1)
    parser.add_argument("--st", dest="skip_size_tolerance", type=int, default=10)
    parser.add_argument("--bt", dest="bb_size_tolerance", type=int, default=8)
    parser.add_argument("--it", dest="indel_tolerance", type=int, default=8)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=1)
    parser.add_argument("--ap", dest="anchor_mismatch_penalty", type=int, default=5)
    parser.add_argument("--lp", dest="linker_mismatch_penalty", type=int, default=3)
    parser.add_argument("--bq", dest="bq_cutoff", type=int, default=None)
    parser.add_argument("--ac", dest="anchor_list", type=str, default="AAA")
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    args = parser.parse_args()
    return args


def parse_args_strict():
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--input", dest="input", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str, required=True)
    parser.add_argument("--fl", dest="linker_front", type=str, default="CGAC")
    parser.add_argument("--bl", dest="linker_back", type=str, default="AGUC")
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cc", dest="cb_count", type=int, default=3)
    parser.add_argument("--lt", dest="linker_mismatch_tolerance", type=int, default=2)
    parser.add_argument("--et", dest="edge_linker_mismatch_tolerance", type=int, default=0)
    parser.add_argument("--st", dest="skip_size_tolerance", type=int, default=8)
    parser.add_argument("--bt", dest="bb_size_tolerance", type=int, default=6)
    parser.add_argument("--it", dest="indel_tolerance", type=int, default=6)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=2)
    parser.add_argument("--ap", dest="anchor_mismatch_penalty", type=int, default=6)
    parser.add_argument("--lp", dest="linker_mismatch_penalty", type=int, default=3)
    parser.add_argument("--bq", dest="bq_cutoff", type=int, default=None)
    parser.add_argument("--ac", dest="anchor_list", type=str, default="AAA")
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    args = parser.parse_args()
    return args


def main():
    args = parse_args_strict()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"ERROR! {args.input} does not exist.")
    if os.path.exists(args.output):
        raise FileExistsError(f"ERROR! {args.output} already exists.")

    basepath = os.path.dirname(args.output)
    flushpath = f"{basepath}/block_flush_{time.strftime('%Y%m%d%H%M%S')}/"
    os.makedirs(flushpath, exist_ok=True)
    atexit.register(os.system, f"rm -r {flushpath}")

    args_dict = vars(args)

    extract_blocks_from_read_list(**args_dict, flushpath=flushpath)

    os.system(f"rm -r {flushpath}")

    return None


if __name__ == "__main__":
    main()
