import pickle
import sys
import pod5
import tqdm
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
import networkx as nx
import numpy as np
import pandas as pd
import polyleven as pl
import pysam
from tqdm import tqdm
import shutil
from utils.utils import mean_phred, printmessage, oom_killer

## Step 1: Index all k-mers from the read.
## Step 2: Connect the spacers using the k-mer index.
## Step 3: Build a DAG of spacers.
## Step 4: Find the longest path in the DAG.
## Step 5: Extract the sequence from the longest path.

def get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size):
    """
    Generates a dictionary of minimum ideal displacements for given parameters.

    Args:
        cb_per_bb (int): Number of context blocks per base block.
        spacer_size (int): Size of the spacer.
        cb_size (int): Size of the context block.

    Returns:
        dict: Dictionary with keys as tuples of (from_idx, to_idx) and values as tuples of (displacement, small_steps, big_steps).
    """
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
                           bb_size):
    """
    Calculates the ideal displacement and steps between spacers.

    Args:
        from_spacer_idx (int): Index of the starting spacer.
        to_spacer_idx (int): Index of the ending spacer.
        displacement (int): Actual displacement between spacers.
        min_ideal_displacement_dict (dict): Dictionary of minimum ideal displacements.
        cb_per_bb (int): Number of context blocks per base block.
        bb_size (int): Size of the base block.

    Returns:
        tuple: Ideal displacement, small steps, and big steps.
    """
    min_ideal_displacement, min_small_steps, min_big_steps = min_ideal_displacement_dict[
        (from_spacer_idx, to_spacer_idx)]
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
    """
    Generates a dictionary of integer partitions for indel tolerance.

    Args:
        indel_tolerance (int): Indel tolerance.
        cb_size_tolerance (int): Context block size tolerance.

    Returns:
        dict: Dictionary with keys as spacing errors and values as lists of tuples of (front_error, back_error).
    """
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


def get_kmer_dict(read, k, bq_cutoff, phred):
    """
    Generates a dictionary of k-mers from a read.

    Args:
        read (str): The read sequence.
        k (int): Length of the k-mer.
        bq_cutoff (float): Base quality cutoff.
        phred (list): List of Phred quality scores.

    Returns:
        defaultdict: Dictionary with k-mers as keys and positions as values.
    """
    kmer_dict = defaultdict(list)
    for i in range(len(read) - k + 1):
        kmer = read[i:i + k]
        if bq_cutoff:
            bq = np.mean(phred[i:i + k])
            if bq < bq_cutoff:
                continue
        kmer_dict[kmer].append(i)
    return kmer_dict


def get_ed_kmers(kmer, spacer_mismatch_tolerance):
    """
    Generates a dictionary of k-mers with edit distances.

    Args:
        kmer (str): The k-mer sequence.
        spacer_mismatch_tolerance (int): Tolerance for mismatches in spacers.

    Returns:
        defaultdict: Dictionary with edit distances as keys and lists of k-mers as values.
    """
    nucs = "ACGU"
    possible_nucs = ["".join(x) for x in it.product(nucs, repeat=len(kmer))]
    kmer_ed_dict = defaultdict(list)
    for possible_kmer in possible_nucs:
        ed = pl.levenshtein(kmer, possible_kmer, spacer_mismatch_tolerance)
        kmer_ed_dict[ed].append(possible_kmer)
    kmer_ed_dict[spacer_mismatch_tolerance + 1] = []
    return kmer_ed_dict


def validate_anchor(read, from_pos, to_pos, possible_indel_list, spacer_size, cb_pad, single_anchor,
                    indel_penalty, anchor_mismatch_penalty, displacement_error):
    """
    Validates the anchor in the read sequence.

    Args:
        read (str): The read sequence.
        from_pos (int): Starting position.
        to_pos (int): Ending position.
        possible_indel_list (list): List of possible indels.
        spacer_size (int): Size of the spacer.
        cb_pad (int): Context block padding.
        single_anchor (str): Single anchor sequence.
        indel_penalty (int): Penalty for indels.
        anchor_mismatch_penalty (int): Penalty for anchor mismatches.
        displacement_error (int): Displacement error.

    Returns:
        tuple: Missing anchor, anchor position, and total indel.
    """
    query = read[from_pos + spacer_size:to_pos]
    anchor_candidate_list = [
        (displacement_error * indel_penalty + anchor_mismatch_penalty, 1, None, displacement_error)]
    ## penalty, missing_anchor, anchor_pos, total_indel
    for front_indel, back_indel in possible_indel_list:
        anchor_query = query[cb_pad + front_indel]
        if anchor_query == single_anchor:
            anchor_pos = from_pos + spacer_size + cb_pad + front_indel
            total_indel = np.abs(front_indel) + np.abs(back_indel)
            anchor_candidate_list.append((total_indel * indel_penalty, 0, anchor_pos, total_indel))
    anchor_candidate_list.sort(key=lambda x: (x[0], x[1]))
    return anchor_candidate_list[0][1:]


def get_kmer_tuple(spacer_mismatch_tolerance, from_spacer_kmer_ed_dict, to_spacer_kmer_ed_dict):
    """
    Generates a list of k-mer tuples with mismatches.

    Args:
        spacer_mismatch_tolerance (int): Tolerance for mismatches in spacers.
        from_spacer_kmer_ed_dict (dict): Dictionary of k-mers with edit distances for the starting spacer.
        to_spacer_kmer_ed_dict (dict): Dictionary of k-mers with edit distances for the ending spacer.

    Returns:
        list: List of tuples of (from_kmer, to_kmer, total_mismatch).
    """
    kmer_tuple_list = []
    for total_mismatch in range(spacer_mismatch_tolerance + 1):
        for front_mismatch in range(total_mismatch + 1):
            back_mismatch = total_mismatch - front_mismatch
            for from_kmer, to_kmer in it.product(from_spacer_kmer_ed_dict[front_mismatch],
                                                 to_spacer_kmer_ed_dict[back_mismatch]):
                kmer_tuple_list.append((from_kmer, to_kmer, total_mismatch))
    return kmer_tuple_list


def find_block_candidates(seq, phred, cb_bq_cutoff, spacer_kmer_ed_dict, skip_size_tolerance, cb_pad,
                          cb_per_bb, indel_penalty, anchor_mismatch_penalty, spacer_mismatch_penalty,
                          spacer_size, spacer_list, indel_dict, min_ideal_displacement_dict, anchor_list,
                          score_converting_func, cb_size_tolerance, spacer_mismatch_tolerance, spacer_size_tolerance,
                          bb_size):
    """
    Finds block candidates in the read sequence.

    Args:
        seq (str): The read sequence.
        phred (list): List of Phred quality scores.
        cb_bq_cutoff (float): Base quality cutoff for context blocks.
        spacer_kmer_ed_dict (dict): Dictionary of k-mers with edit distances for spacers.
        skip_size_tolerance (int): Tolerance for skip size.
        cb_pad (int): Context block padding.
        cb_per_bb (int): Number of context blocks per base block.
        indel_penalty (int): Penalty for indels.
        anchor_mismatch_penalty (int): Penalty for anchor mismatches.
        spacer_mismatch_penalty (int): Penalty for spacer mismatches.
        spacer_size (int): Size of the spacer.
        spacer_list (list): List of spacers.
        indel_dict (dict): Dictionary of integer partitions for indel tolerance.
        min_ideal_displacement_dict (dict): Dictionary of minimum ideal displacements.
        anchor_list (list): List of anchors.
        score_converting_func (function): Function to convert penalty to score.
        cb_size_tolerance (int): Context block size tolerance.
        spacer_mismatch_tolerance (int): Tolerance for mismatches in spacers.
        spacer_size_tolerance (int): Tolerance for spacer size.
        bb_size (int): Size of the base block.

    Returns:
        tuple: Dictionary of context block information, list of DAG edges, and dictionary of DAG edges with scores.
    """
    kmer_pos_dict = get_kmer_dict(seq, spacer_size, cb_bq_cutoff, phred)
    dag_list = []  ## Format: [from_pos, to_pos, score]
    dag_dict = {}  ## Format: {(from_pos, to_pos): score}
    cb_info_dict = {}  ## Format: {(from_pos, to_pos): [cb_idx,from_pos,to_pos,anchor_pos,score]}

    for from_spacer_idx in range(len(spacer_list)):

        if from_spacer_idx == cb_per_bb:
            single_anchor = None
        else:
            single_anchor = anchor_list[from_spacer_idx]

        from_spacer_kmer_ed_dict = spacer_kmer_ed_dict[from_spacer_idx]

        for to_spacer_idx in range(len(spacer_list)):

            to_spacer_kmer_ed_dict = spacer_kmer_ed_dict[to_spacer_idx]
            kmer_tuple_list = get_kmer_tuple(spacer_mismatch_tolerance, from_spacer_kmer_ed_dict,
                                             to_spacer_kmer_ed_dict)

            for from_kmer, to_kmer, kmer_mismatch in kmer_tuple_list:
                for from_pos, to_pos in it.product(kmer_pos_dict[from_kmer], kmer_pos_dict[to_kmer]):

                    displacement = to_pos - from_pos
                    if displacement < 0:
                        continue

                    ideal_displacement, small_steps, big_steps = get_ideal_displacement(from_spacer_idx, to_spacer_idx,
                                                                                        displacement,
                                                                                        min_ideal_displacement_dict,
                                                                                        cb_per_bb, bb_size)
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
                                                                                  displacement_error_abs)
                        if missing_anchor == 0:
                            is_cb = True
                            displacement_error_abs = total_indel

                    elif big_steps == 0 and small_steps == 1 and displacement_error_abs <= spacer_size_tolerance:
                        missing_anchor = 0

                    penalty = spacer_mismatch_penalty * kmer_mismatch + indel_penalty * displacement_error_abs + anchor_mismatch_penalty * missing_anchor
                    score = score_converting_func(penalty)

                    from_pos_id = (from_spacer_idx, from_pos)
                    to_pos_id = (to_spacer_idx, to_pos)

                    if is_cb:
                        cb_info_dict[(from_pos_id, to_pos_id)] = [from_spacer_idx, from_pos, to_pos, anchor_pos,
                                                                  penalty, score]

                    dag_list.append((from_pos_id, to_pos_id, score))
                    dag_dict[(from_pos_id, to_pos_id)] = score

    return cb_info_dict, dag_list, dag_dict


def dag_longest_path(edge_list):
    """
    Finds the longest path in a directed acyclic graph (DAG).

    Args:
        edge_list (list): List of edges in the DAG.

    Returns:
        list: Longest path in the DAG.
    """
    node_list = list(set([x[0] for x in edge_list] + [x[1] for x in edge_list]))

    dag = nx.DiGraph()
    dag.add_nodes_from(node_list)
    dag.add_weighted_edges_from(edge_list)
    longest_path = nx.dag_longest_path(dag, weight='weight')

    return longest_path


def extract_blocks_from_read_list_mp_worker(record_list, indel_penalty, cb_size_tolerance,
                                            skip_size_tolerance, anchor_mismatch_penalty, spacer_size_tolerance,
                                            spacer_mismatch_tolerance, spacer_mismatch_penalty,
                                            cb_pad, cb_per_bb, cb_bq_cutoff, indel_dict, spacer_kmer_ed_dict,
                                            anchor_list, spacer_list, spacer_size, bb_size, flush_path, pid,
                                            flush_interval,
                                            score_converting_func, cb_size, min_ideal_displacement_dict, resume):
    """
    Worker function to extract blocks from a list of reads using multiprocessing.

    Args:
        record_list (list): List of read records.
        indel_penalty (int): Penalty for indels.
        cb_size_tolerance (int): Context block size tolerance.
        skip_size_tolerance (int): Tolerance for skip size.
        anchor_mismatch_penalty (int): Penalty for anchor mismatches.
        spacer_size_tolerance (int): Tolerance for spacer size.
        spacer_mismatch_tolerance (int): Tolerance for mismatches in spacers.
        spacer_mismatch_penalty (int): Penalty for spacer mismatches.
        cb_pad (int): Context block padding.
        cb_per_bb (int): Number of context blocks per base block.
        cb_bq_cutoff (float): Base quality cutoff for context blocks.
        indel_dict (dict): Dictionary of integer partitions for indel tolerance.
        spacer_kmer_ed_dict (dict): Dictionary of k-mers with edit distances for spacers.
        anchor_list (list): List of anchors.
        spacer_list (list): List of spacers.
        spacer_size (int): Size of the spacer.
        bb_size (int): Size of the base block.
        flush_path (str): Path to save intermediate flush files.
        pid (int): Process ID.
        flush_interval (int): Interval for flushing data to disk.
        score_converting_func (function): Function to convert penalty to score.
        cb_size (int): Size of the context block.
        min_ideal_displacement_dict (dict): Dictionary of minimum ideal displacements.
        resume (str): Path to resume from previous run.

    Returns:
        None
    """

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
                                                                 spacer_mismatch_penalty,
                                                                 spacer_size, spacer_list, indel_dict,
                                                                 min_ideal_displacement_dict, anchor_list,
                                                                 score_converting_func, cb_size_tolerance,
                                                                 spacer_mismatch_tolerance,
                                                                 spacer_size_tolerance, bb_size)

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
                                  skip_size_tolerance, anchor_mismatch_penalty, spacer_size_tolerance,
                                  spacer_mismatch_tolerance, max_read_length,
                                  spacer_mismatch_penalty, anchor_list, spacer_list, spacer_size, cb_pad,
                                  cb_per_bb, read_bq_cutoff, cb_bq_cutoff, flush_path, flush_interval, ncpu,
                                  resume, sample, **kwargs):
    """
    Extracts context blocks from a list of reads using multiprocessing.

    Args:
        input (str): Path to the input BAM file.
        output (str): Path to save the output pickle file.
        indel_tolerance (int): Indel tolerance.
        indel_penalty (int): Penalty for indels.
        cb_size_tolerance (int): Context block size tolerance.
        skip_size_tolerance (int): Tolerance for skip size.
        anchor_mismatch_penalty (int): Penalty for anchor mismatches.
        spacer_size_tolerance (int): Tolerance for spacer size.
        spacer_mismatch_tolerance (int): Tolerance for mismatches in spacers.
        max_read_length (int): Maximum read length.
        spacer_mismatch_penalty (int): Penalty for spacer mismatches.
        anchor_list (list): List of anchors.
        spacer_list (list): List of spacers.
        spacer_size (int): Size of the spacer.
        cb_pad (int): Context block padding.
        cb_per_bb (int): Number of context blocks per base block.
        read_bq_cutoff (float): Base quality cutoff for reads.
        cb_bq_cutoff (float): Base quality cutoff for context blocks.
        flush_path (str): Path to save intermediate flush files.
        flush_interval (int): Interval for flushing data to disk.
        ncpu (int): Number of CPU threads to use.
        resume (str): Path to resume from previous run.
        sample (int): Number of reads to sample.
        **kwargs: Additional arguments.

    Returns:
        None
    """
    spacer_list = [x.replace("T", "U") for x in spacer_list]
    anchor_list = [x.replace("T", "U") for x in anchor_list]
    indel_dict = get_integer_partition(indel_tolerance, cb_size_tolerance)
    spacer_kmer_ed_dict = {i: get_ed_kmers(kmer, spacer_mismatch_tolerance) for i, kmer in enumerate(spacer_list)}
    assert indel_tolerance >= cb_size_tolerance

    max_cb_penalty = anchor_mismatch_penalty + spacer_mismatch_penalty * spacer_mismatch_tolerance + indel_penalty * indel_tolerance
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
                                skip_size_tolerance, anchor_mismatch_penalty, spacer_size_tolerance,
                                spacer_mismatch_tolerance, spacer_mismatch_penalty,
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

    block_df.to_pickle(f"{output}/block.pkl")

    log = []
    log.append(f"Total number of passed reads: {record_cnt:,}")
    log.append(f"Total number of context blocks: {len(block_df):,}")
    log.append(f"Context blocks per read: {len(block_df) / record_cnt:.2f}")
    log.append(block_df["score"].describe())
    log.append(block_df["penalty"].describe())

    log_path = f"{output}/dag_log.txt"
    with open(log_path, "w") as log_file:
        for line in log:
            log_file.write(f"{line}\n")

    print("=============================================")
    for line in log:
        printmessage(line)
    print("=============================================")

    printmessage(f"Saved context blocks to {output}/block.pkl.")

    return block_df


def extract_move(bam_path, ncpu, signal_path_dict, signal_path_arr, intermediate_path):
    """
    Extracts the 'mv' tag from a BAM file and saves it to separate files.

    Args:
        bam_path (str): Path to the BAM file.
        ncpu (int): Number of CPU threads to use.
        signal_path_dict (dict): Dictionary mapping read IDs to signal paths.
        signal_path_arr (list): List of signal paths.
        intermediate_path (str): Path to save intermediate files.

    Returns:
        None
    """
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": []} for x in signal_path_arr}
    count = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                else:
                    read_id = str(read.query_name)

                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except:
                    continue

                if read.has_tag("mv"):
                    mv = read.get_tag("mv")
                else:
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

                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["mv"].append(mv)
                data["sp"].append(sp)
                count += 1

                pbar.update(1)

    printmessage(f"Valid read count: {count}", msg_type="info")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None


def preprocess_pod5(pod5_path, save_path, ncpu, chunk, max_mb, min_mb):
    """
    Exports POD5 files to DataFrame format.

    Args:
        pod5_path (str): Path to the POD5 files.
        save_path (str): Path to save the DataFrame files.
        ncpu (int): Number of CPU threads to use.
        chunk (int): Chunk size for processing.
        max_mb (int): Maximum size of the DataFrame in MB.
        min_mb (int): Minimum size of the DataFrame in MB.

    Returns:
        dict: Dictionary mapping file paths to read IDs.
    """
    pod5_path_list = glob.glob(pod5_path + "/*.pod5")
    proc_list = []
    np.random.shuffle(pod5_path_list)
    pod5_path_list_split = np.array_split(pod5_path_list, ncpu)

    man = mp.Manager()
    index_list = man.list()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_signal_proc, args=(pod5_path_list_split[pid], save_path, pid, index_list,
                                                            chunk, max_mb, min_mb))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    index_dict = {}
    for local_index_dict in index_list:
        index_dict.update(local_index_dict)

    man.shutdown()
    gc.collect()

    return index_dict


def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_list, chunk, max_mb, min_mb):
    """
    Extracts signal data from POD5 files and processes it.

    Args:
        pod5_path_list (list): List of POD5 file paths.
        signal_df_path (str): Path to save the signal data.
        pid (int): Process ID.
        index_list (list): List to store the index data.
        chunk (int): Chunk size for processing.
        max_mb (int): Maximum size of the dataframe in MB.
        min_mb (int): Minimum size of the dataframe in MB.

    Returns:
        None
    """
    index_dict_local = {}
    chunk_buffer = []
    pod5_idx = 0

    for pod5_idx, pod5_path in tqdm.tqdm(enumerate(pod5_path_list), total=len(pod5_path_list), desc=f"Parsing POD5 Files"):
        oom_killer()
        signal_list = []
        offset_list = []
        scale_list = []
        id_list = []
        try:
            with pod5.Reader(pod5_path) as reader:
                skipped = 0
                for record in reader:
                    try:
                        signal_arr = record.signal
                        offset = record.calibration.offset
                        scale = record.calibration.scale
                        id = str(record.read_id)
                    except:
                        skipped += 1
                        continue

                    offset_list.append(offset)
                    scale_list.append(scale)
                    signal_list.append(signal_arr)
                    id_list.append(id)

            if skipped > 0:
                printmessage(f"Skipped {skipped} faulty records in: {pod5_path}", msg_type="warning")

        except:
            ## Pod5 file is corrupted
            printmessage(f"Corrupted POD5 file: {pod5_path} - Skipping", msg_type="warning")
            continue

        df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
        del signal_list, offset_list, scale_list, id_list
        gc.collect()

        save_idx = 0

        for chunk_idx in range(0, len(df) // chunk + 1):
            signal_df = df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(df))].copy()
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if len(signal_df) == chunk and df_size > min_mb:
                save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer.append(signal_df)
            gc.collect()

        if len(chunk_buffer) > 0:
            signal_df = pd.concat(chunk_buffer, ignore_index=True)
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if df_size > min_mb:
                chunk_buffer = []
                write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer = [signal_df]
        gc.collect()

    if len(chunk_buffer) > 0:
        signal_df = pd.concat(chunk_buffer, ignore_index=True)
        save_idx = 0
        pod5_idx += 1
        write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
        gc.collect()

    index_list.append(index_dict_local)

    return None


def write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb):
    """
    Writes the signal dataframe to a file.

    Args:
        signal_df (pd.DataFrame): Dataframe containing the signal data.
        signal_df_path (str): Path to save the signal data.
        pid (int): Process ID.
        pod5_idx (int): POD5 file index.
        save_idx (int): Save index.
        index_dict (dict): Dictionary to store the index data.
        max_mb (int): Maximum size of the dataframe in MB.

    Returns:
        int: Updated save index.
    """
    save_idx += 1
    df_size = sys.getsizeof(signal_df) / (1024 ** 2)
    if df_size > max_mb and len(signal_df) > 1:
        ## Split the dataframe
        split_num = min(int(df_size // max_mb) + 1 ,len(signal_df))
        split_size = len(signal_df) // split_num
        for split_idx in range(split_num):
            split_df = signal_df.iloc[split_idx * split_size:min((split_idx + 1) * split_size, len(signal_df))].copy()
            save_idx = write_df(split_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb)
    else:
        save_path = f"{signal_df_path}/{pid}-{pod5_idx}-{save_idx}.pkl"
        if os.path.exists(save_path):
            raise FileExistsError(f"File {save_path} already exists")
        signal_df.to_pickle(save_path)
        id_list = signal_df["read_id"].tolist()
        index_dict[save_path] = id_list
    return save_idx


def sequence_to_kmer_token(seq, kmer):
    """
    Converts a DNA sequence to k-mer tokens.

    Args:
        seq (str): DNA sequence.
        kmer (int): Length of the k-mer.

    Returns:
        np.ndarray: Array of k-mer tokens.
    """
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i:kmer+i] for i in range(len(seq)-kmer+1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4**np.arange(kmer).reshape(-1,1)
    seq = np.sum(seq * quaternary, axis=0) + 1 ## 0 is reserved for padding
    seq = seq.astype(np.int16)
    return seq


def create_segment_len_arr(segment_arr, sampling):
    """
    Creates an array of segment lengths.

    Args:
        segment_arr (list): List of segments.
        sampling (int): Sampling rate.

    Returns:
        np.ndarray: Array of segment lengths.
    """
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int) // sampling
    return segment_len_arr


def expand_token_to_segment(token_arr, segment_len_arr):
    """
    Expands tokens to segments.

    Args:
        token_arr (np.ndarray): Array of tokens.
        segment_len_arr (np.ndarray): Array of segment lengths.

    Returns:
        np.ndarray: Expanded array of tokens.
    """
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_move_token(segment_len_arr):
    """
    Creates move tokens.

    Args:
        segment_len_arr (np.ndarray): Array of segment lengths.

    Returns:
        np.ndarray: Array of move tokens.
    """
    token = np.arange(1, len(segment_len_arr)+1, dtype=np.uint8)
    token = np.repeat(token, segment_len_arr)
    return token


def create_target_mask(segment_len_arr, lr_pad):
    """
    Creates a target mask.

    Args:
        segment_len_arr (np.ndarray): Array of segment lengths.
        lr_pad (int): Left-right padding.

    Returns:
        np.ndarray: Target mask.
    """
    binary_mask = np.zeros(2*lr_pad+1, dtype=np.uint8)
    binary_mask[lr_pad] = 1
    binary_mask = np.repeat(binary_mask, segment_len_arr)
    return binary_mask


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    """
    Segments and pads the signal.

    Args:
        signal_segmented (np.ndarray): Segmented signal.
        segment_len_arr (np.ndarray): Array of segment lengths.
        kmer (int): Length of the k-mer.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        pad_to (int): Padding size.

    Returns:
        np.ndarray: Padded signal.
    """
    try:
        kmer_pad = (kmer-1)//2
        lr_pad = (sig_window-1)//2
        l_skip = (np.sum(segment_len_arr[:kmer_pad])-lr_pad)*sampling
        r_skip = (np.sum(segment_len_arr[-kmer_pad:])-lr_pad)*sampling
        assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
        assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]
        # signal_segmented = np.lib.stride_tricks.sliding_window_view(signal_segmented, sig_window * sampling)[::sampling]

        padding = (pad_to+kmer-1) * sampling - len(signal_segmented)
        if padding > 0:
            signal_segmented = np.pad(signal_segmented, (0, padding), mode="constant", constant_values=0)

    except:
        return None
    return signal_segmented


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Converts move data to dwell time.

    Args:
        move (np.ndarray): Move data.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        np.ndarray: Dwell time data.
    """
    sampling = move[0]
    move = np.flip(move[1:]) * np.arange(1, len(move))
    move = move[move > 0]
    move = np.concatenate([np.zeros(1, dtype=int), move])
    move = move[1:] - move[:-1]
    move = move * sampling
    move = np.log10(move.astype(np.float32))
    quantile_a_value = np.quantile(move, quantile_a)
    quantile_b_value = np.quantile(move, quantile_b)
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    move = (move - q_shift) / q_scale
    return move


def trim_scale_segment_signal(signal,move,sp,ts,ns, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Trims and scales the signal.

    Args:
        signal (np.ndarray): Signal data.
        move (np.ndarray): Move data.
        sp (int): Start position.
        ts (int): Timestamp.
        ns (int): Number of samples.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        np.ndarray: Trimmed and scaled signal.
    """
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

    q_shift = shift_mult * (quantile_a_value + quantile_b_value)
    q_scale = scale_mult * (quantile_b_value - quantile_a_value)
    signal = (signal - q_shift) / q_scale
    signal = signal.astype(np.float32)
    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def segment_normalize_signal(seg_df_path, postfix, signal_path_arr, norm_factor, kmer = 5, cb_len = 21, sampling = 6,
                             sig_window = 5, max_penalty = 10, chunk_size = 1000, max_token_len = 200, dwell_shift = 10):
    """
    Segments and normalizes the signal data.

    Args:
        seg_df_path (str): Path to the segmented dataframe.
        postfix (str): Postfix for the output files.
        signal_path_arr (list): List of signal paths.
        norm_factor (dict): Normalization factors.
        kmer (int, optional): Length of the k-mer. Defaults to 5.
        cb_len (int, optional): Length of the codebook. Defaults to 21.
        sampling (int, optional): Sampling rate. Defaults to 6.
        sig_window (int, optional): Signal window size. Defaults to 5.
        max_penalty (int, optional): Maximum penalty. Defaults to 10.
        chunk_size (int, optional): Chunk size for processing. Defaults to 1000.
        max_token_len (int, optional): Maximum token length. Defaults to 200.
        dwell_shift (int, optional): Dwell shift. Defaults to 10.

    Returns:
        None
    """

    trim = kmer//2

    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]
    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]

    for signal_path in tqdm.tqdm(signal_path_arr, total=len(signal_path_arr), desc="Segmenting and Tokenizing Signals"):
        oom_killer()
        file_id = signal_path.split('/')[-1]

        if not os.path.exists(signal_path):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/move_df_split/{file_id}"):
            continue
        if not os.path.exists(f"{seg_df_path}/intermediates/block_df_split/{file_id}"):
            continue

        out_path = f"{seg_df_path}/{postfix}/{file_id}"
        if os.path.exists(out_path):
            continue

        signal_df = pd.read_pickle(signal_path)
        move_df = pd.read_pickle(f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}")
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["dwell_token"] = signal_df["mv"].apply(lambda x: move_to_dwell(x, 0.2, 0.8, 0.5, 1.5))
        signal_df["signal"] = signal_df.apply(lambda x: trim_scale_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                  quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        signal_df = signal_df[signal_df["signal"].notnull()][["read_id", "signal", "dwell_token"]].copy()

        block_df = pd.read_pickle(f"{seg_df_path}/intermediates/block_df_split/{signal_path.split('/')[-1]}")
        # block_df = block_df[block_df["penalty"]==0]
        if len(block_df) == 0:
            continue

        signal_df = block_df.merge(signal_df, on="read_id", how="inner")
        del block_df
        gc.collect()

        signal_df["signal_length"] = signal_df["signal"].apply(lambda x: len(x))
        signal_df = signal_df[signal_df["end_pos"] + dwell_shift - trim < signal_df["signal_length"]]

        if len(signal_df) == 0:
            continue

        signal_df["block_score"] = signal_df["penalty"].apply(lambda x: 1-(x/max_penalty))
        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["dwell_motor_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"]+dwell_shift+trim):(x["end_pos"]+dwell_shift-trim)], axis=1)
        signal_df["dwell_pore_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"]+trim):(x["end_pos"]-trim)], axis=1)
        signal_df["bq"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))

        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x)==cb_len)) &
                              (signal_df["token_len"] <= max_token_len) &
                              (signal_df["token_len"] > 0)]
        try:
            signal_df["signal"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"],
                                                                                      kmer, sampling, sig_window, max_token_len), axis=1)
        except:
            print(f"Signal Tokenization Error in: {signal_path} - Skipping")
            continue

        signal_df = signal_df[signal_df["signal"].notnull()]

        if len(signal_df) == 0:
            continue

        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])

        signal_df = signal_df[["block_score", "segment_len_arr", "signal", "motif", "dwell_motor_token", "dwell_pore_token", "bq"]].copy()
        signal_df.rename(columns={"motif": "kmer_token", "signal": "signal_token", "bq": "bq_token"}, inplace=True)
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x.astype(np.uint8))
        signal_df["signal_token"] = signal_df["signal_token"].apply(lambda x: x.astype(np.float32))
        signal_df["kmer_token"] = signal_df["kmer_token"].apply(lambda x: np.array(list(x)).view(np.int32).astype(np.uint8))
        signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: np.clip(x,0,60).astype(np.uint8))

        for chunk_idx in range(0, len(signal_df) // chunk_size + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk_size:min((chunk_idx + 1) * chunk_size, len(signal_df))].copy()
            save_path = f"{out_path.replace('.pkl','')}-{chunk_idx}.npz"
            save_npz(save_path, chunk_df)

        del signal_df, chunk_df
        gc.collect()

    return None
def save_npz(save_path, df):
    """
    Saves the dataframe to a compressed NPZ file.

    Args:
        save_path (str): Path to save the NPZ file.
        df (pd.DataFrame): Dataframe containing the data to be saved.

    Returns:
        None
    """
    if len(df) > 0:
        segment_len_arr = np.stack(df["segment_len_arr"].values)
        signal_token = np.stack(df["signal_token"].values)
        kmer_token = np.stack(df["kmer_token"].values)
        dwell_motor_token = np.stack(df["dwell_motor_token"].values)
        dwell_pore_token = np.stack(df["dwell_pore_token"].values)
        bq_token = np.stack(df["bq_token"].values)
        block_score = df["block_score"].values
        np.savez_compressed(save_path,
                            segment_len_arr=segment_len_arr,
                            signal_token=signal_token,
                            kmer_token=kmer_token,
                            dwell_motor_token=dwell_motor_token,
                            dwell_pore_token=dwell_pore_token,
                            bq_token=bq_token,
                            block_score=block_score)

    return None


def assign_block_id(block_df):
    """
    Assigns block IDs to the dataframe.

    Args:
        block_df (pd.DataFrame): Dataframe containing block data.

    Returns:
        pd.DataFrame: Dataframe with assigned block IDs.
    """
    index = 0
    read_id_prev = ""
    block_id = []
    for read_id in block_df["read_id"]:
        if read_id != read_id_prev:
            index = 0
        else:
            index += 1
        block_id.append(index)
        read_id_prev = read_id
    block_df["block_id"] = block_id
    return block_df


def split_block_df(signal_path_dict, signal_path_arr, intermediate_path, block_df):
    """
    Splits the block dataframe into smaller dataframes based on signal paths.

    Args:
        signal_path_dict (dict): Dictionary mapping read IDs to signal paths.
        signal_path_arr (list): List of signal paths.
        intermediate_path (str): Path to save intermediate files.
        block_df (pd.DataFrame): Dataframe containing block data.

    Returns:
        None
    """
    printmessage("Reading Block Dataframe. It may take a while.", msg_type="info")
    block_df = assign_block_id(block_df)
    block_df["signal_path"] = block_df["read_id"].map(signal_path_dict)

    ## Groupby read_id and make dict
    block_df_groupby = block_df.groupby("signal_path")

    del block_df
    gc.collect()

    for signal_path, group_df in tqdm.tqdm(block_df_groupby, total=len(signal_path_arr), desc="Splitting Block Dataframe"):
        group_df.to_pickle(f"{intermediate_path}/block_df_split/{signal_path}")

    del block_df_groupby
    gc.collect()

    return None


def get_norm_factor():
    """
    Returns the default normalization factors.

    Returns:
        dict: Dictionary containing default normalization factors.
    """
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59

    return norm_factor_default


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract context blocks from basecalled BAM file using DAG.")
    num_cpu = os.cpu_count()
    parser.add_argument("--input", dest="input BAM file", type=str, required=True)
    parser.add_argument("--output", dest="output directory", type=str, required=True)
    parser.add_argument("--cpu", dest="ncpu", type=int, default=int(num_cpu * 0.9))

    ## DAG extraction parameters
    parser.add_argument("--it", dest="indel_tolerance", type=int, default=3)
    parser.add_argument("--ip", dest="indel_penalty", type=int, default=3)
    parser.add_argument("--cst", dest="cb_size_tolerance", type=int, default=3)
    parser.add_argument("--kst", dest="skip_size_tolerance", type=int, default=4)
    parser.add_argument("--amp", dest="anchor_mismatch_penalty", type=int, default=6)
    parser.add_argument("--smt", dest="spacer_mismatch_tolerance", type=int, default=3)
    parser.add_argument("--smp", dest="spacer_mismatch_penalty", type=int, default=2)
    parser.add_argument("--sst", dest="spacer_size_tolerance", type=int, default=1)
    parser.add_argument("--ac", dest="anchor_list", type=str, nargs="+", default=["A", "A", "A"])
    parser.add_argument("--sp", dest="spacer_list", type=str, nargs="+",
                        default=["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"])
    parser.add_argument("--ss", dest="spacer_size", type=int, default=6)
    parser.add_argument("--cp", dest="cb_pad", type=int, default=10)
    parser.add_argument("--cb", dest="cb_per_bb", type=int, default=3)
    parser.add_argument("--rbq", dest="read_bq_cutoff", type=int, default=7)
    parser.add_argument("--cbq", dest="cb_bq_cutoff", type=int, default=0)
    parser.add_argument("--fi", dest="flush_interval", type=int, default=1000)
    parser.add_argument("--max", dest="max_read_length", type=int, default=1000)
    parser.add_argument("--min", dest="min_read_length", type=int, default=0)
    parser.add_argument("--sample", dest="sample", type=int, default=None)
    parser.add_argument("--keep", dest="keep_intermediate", type=bool, default=False)
    parser.add_argument("--cfg", dest="config", type=str, default=None)
    parser.add_argument("--resume", dest="resume", type=str, default=None, help="Continue from previous run. Provide the path to the previous output.")

    ## Signal preprocessing parameters
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--chunk", "-n", type=int, default=500, help="POD5 Chunk size")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5 dataframe size in MB")
    parser.add_argument("--keep_intermediate", "-ki", action="store_true", help="Keep intermediate files", default=True)
    parser.add_argument("--postfix", "-x", type=str, default="training_dataset", help="Output file postfix")

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

    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input POD5 directory {args.pod5} does not exist")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input BAM file {args.input} does not exist")
    if not os.path.exists(args.block):
        raise FileNotFoundError(f"Context Block file {args.block} does not exist")
    if os.path.exists(args.output):
        raise FileExistsError(f"Output directory {args.output} already exists. Please choose a different output directory or remove the existing one.")
    os.makedirs(args.output, exist_ok=True)

    return args


def main():
    """
    Main function to extract context blocks from a basecalled BAM file using a directed acyclic graph (DAG).

    Returns:
        None
    """
    args = parse_args()

    norm_factor = get_norm_factor()

    token_output_path = f"{args.output}/{args.postfix}/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    os.makedirs(f"{intermediate_path}/block_df_split", exist_ok=True)

    if not args.keep_intermediate:
        atexit.register(lambda: os.system(f"rm -r {intermediate_path}"))

    index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.ncpu, args.chunk, args.max_size, args.min_size)
    signal_path_arr = list(index_dict.keys())
    signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
    gc.collect()

    if len(signal_path_arr) == 0:
        printmessage("No valid signal files found. Exiting.", msg_type="error")
        return None

    with open(signal_index_path, "wb") as outfile:
        pickle.dump(index_dict, outfile)

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path.split('/')[-1]

    del index_dict
    gc.collect()

    if args.resume is not None:
        flush_path = args.resume

    else:
        flush_path = f"{args.output}/block_flush_{time.strftime('%Y%m%d%H%M%S')}/"
        os.makedirs(flush_path, exist_ok=True)
        if not args.keep_intermediate:
            atexit.register(os.system, f"rm -r {flush_path}")

    args_dict = vars(args)
    block_df = extract_blocks_from_read_list(**args_dict, flush_path=flush_path)

    shutil.rmtree(flush_path, ignore_errors=True)

    split_block_df(signal_path_dict, signal_name_arr, intermediate_path, block_df)
    del block_df
    gc.collect()

    extract_move(args.input, args.ncpu, signal_path_dict, signal_name_arr, intermediate_path)

    del signal_path_dict, signal_name_arr
    gc.collect()

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.ncpu))

    proc_list = []
    for signal_paths in signal_path_arr_split:
        proc = mp.Process(target=segment_normalize_signal,
                          args=(args.output, args.postfix, signal_paths, norm_factor))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    printmessage("Signal Segmentation and Tokenization Complete", msg_type="success")
    printmessage("Saved to: " + args.output, msg_type="success")
    return None


if __name__ == "__main__":
    main()