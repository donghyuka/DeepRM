"""
Extract context blocks from reads.

Key steps to extract context blocks from reads:
    1. Index all k-mers from the read.
    2. Connect the spacers using the k-mer index.
    3. Build a DAG of spacers.
    4. Find the longest path in the DAG.
    5. Extract the sequence from the longest path.
"""

import gc
import glob
import itertools as it
import multiprocessing as mp
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import polyleven as pl
import pysam
from tqdm import tqdm

from deeprm.utils.logging import get_logger
from deeprm.utils.memory import start_mem_watchdog
from deeprm.utils.utils import mean_phred

log = get_logger(__name__)


def get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size):
    """
    Generates a dictionary of minimum ideal displacements for given parameters.

    Args:
        cb_per_bb (int): Number of context blocks per base block.
        spacer_size (int): Size of the spacer.
        cb_size (int): Size of the context block.

    Returns:
        dict: Dictionary with keys as tuples of (from_idx, to_idx) and
            values as tuples of (displacement, small_steps, big_steps).
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


def get_ideal_displacement(
    from_spacer_idx, to_spacer_idx, displacement, min_ideal_displacement_dict, cb_per_bb, bb_size
):
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
        (from_spacer_idx, to_spacer_idx)
    ]
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
        collections.defaultdict: Dictionary with k-mers as keys and positions as values.
    """
    kmer_dict = defaultdict(list)
    for i in range(len(read) - k + 1):
        kmer = read[i : i + k]
        if bq_cutoff:
            bq = np.mean(phred[i : i + k])
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
        collections.defaultdict: Dictionary with edit distances as keys and lists of k-mers as values.
    """
    nucs = "ACGU"
    possible_nucs = ["".join(x) for x in it.product(nucs, repeat=len(kmer))]
    kmer_ed_dict = defaultdict(list)
    for possible_kmer in possible_nucs:
        ed = pl.levenshtein(kmer, possible_kmer, spacer_mismatch_tolerance)
        kmer_ed_dict[ed].append(possible_kmer)
    kmer_ed_dict[spacer_mismatch_tolerance + 1] = []
    return kmer_ed_dict


def validate_anchor(
    read,
    from_pos,
    to_pos,
    possible_indel_list,
    spacer_size,
    cb_pad,
    single_anchor,
    indel_penalty,
    anchor_mismatch_penalty,
    displacement_error,
):
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
    query = read[from_pos + spacer_size : to_pos]
    anchor_candidate_list = [
        (displacement_error * indel_penalty + anchor_mismatch_penalty, 1, None, displacement_error)
    ]
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
            for from_kmer, to_kmer in it.product(
                from_spacer_kmer_ed_dict[front_mismatch], to_spacer_kmer_ed_dict[back_mismatch]
            ):
                kmer_tuple_list.append((from_kmer, to_kmer, total_mismatch))
    return kmer_tuple_list


def find_block_candidates(
    seq,
    phred,
    cb_bq_cutoff,
    spacer_kmer_ed_dict,
    skip_size_tolerance,
    cb_pad,
    cb_per_bb,
    indel_penalty,
    anchor_mismatch_penalty,
    spacer_mismatch_penalty,
    spacer_size,
    spacer_list,
    indel_dict,
    min_ideal_displacement_dict,
    anchor_list,
    score_converting_func,
    cb_size_tolerance,
    spacer_mismatch_tolerance,
    spacer_size_tolerance,
    bb_size,
):
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
        score_converting_func (typing.Callable): Function to convert penalty to score.
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
            kmer_tuple_list = get_kmer_tuple(
                spacer_mismatch_tolerance, from_spacer_kmer_ed_dict, to_spacer_kmer_ed_dict
            )

            for from_kmer, to_kmer, kmer_mismatch in kmer_tuple_list:
                for from_pos, to_pos in it.product(kmer_pos_dict[from_kmer], kmer_pos_dict[to_kmer]):

                    displacement = to_pos - from_pos
                    if displacement < 0:
                        continue

                    ideal_displacement, small_steps, big_steps = get_ideal_displacement(
                        from_spacer_idx, to_spacer_idx, displacement, min_ideal_displacement_dict, cb_per_bb, bb_size
                    )
                    displacement_tolerance_skip = big_steps * skip_size_tolerance

                    displacement_error = displacement - ideal_displacement
                    displacement_error_abs = np.abs(displacement_error)

                    if displacement_error_abs > displacement_tolerance_skip:
                        continue

                    anchor_pos = None
                    is_cb = False
                    missing_anchor = 1

                    if (
                        big_steps == 1
                        and small_steps == 0
                        and displacement_error_abs <= cb_size_tolerance
                        and single_anchor is not None
                    ):
                        possible_indel_list = indel_dict[displacement_error]
                        missing_anchor, anchor_pos, total_indel = validate_anchor(
                            seq,
                            from_pos,
                            to_pos,
                            possible_indel_list,
                            spacer_size,
                            cb_pad,
                            single_anchor,
                            indel_penalty,
                            anchor_mismatch_penalty,
                            displacement_error_abs,
                        )
                        if missing_anchor == 0:
                            is_cb = True
                            displacement_error_abs = total_indel

                    elif big_steps == 0 and small_steps == 1 and displacement_error_abs <= spacer_size_tolerance:
                        missing_anchor = 0

                    penalty = (
                        spacer_mismatch_penalty * kmer_mismatch
                        + indel_penalty * displacement_error_abs
                        + anchor_mismatch_penalty * missing_anchor
                    )
                    score = score_converting_func(penalty)

                    from_pos_id = (from_spacer_idx, from_pos)
                    to_pos_id = (to_spacer_idx, to_pos)

                    if is_cb:
                        cb_info_dict[(from_pos_id, to_pos_id)] = [
                            from_spacer_idx,
                            from_pos,
                            to_pos,
                            anchor_pos,
                            penalty,
                            score,
                        ]

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
    longest_path = nx.dag_longest_path(dag, weight="weight")

    return longest_path


def extract_blocks_from_read_list_mp_worker(
    record_list,
    indel_penalty,
    cb_size_tolerance,
    skip_size_tolerance,
    anchor_mismatch_penalty,
    spacer_size_tolerance,
    spacer_mismatch_tolerance,
    spacer_mismatch_penalty,
    cb_pad,
    cb_per_bb,
    cb_bq_cutoff,
    indel_dict,
    spacer_kmer_ed_dict,
    anchor_list,
    spacer_list,
    spacer_size,
    bb_size,
    flush_path,
    pid,
    flush_interval,
    score_converting_func,
    cb_size,
    min_ideal_displacement_dict,
    resume,
):
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
        score_converting_func (typing.Callable): Function to convert penalty to score.
        cb_size (int): Size of the context block.
        min_ideal_displacement_dict (dict): Dictionary of minimum ideal displacements.
        resume (str): Path to resume from previous run.

    Returns:
        None
    """

    start_mem_watchdog()

    len_record = len(record_list)
    block_df_list = []
    flush_file_list = []
    last_flush_idx = 0

    if resume is not None:
        ## search for last flush file
        flush_file_list = glob.glob(os.path.join(resume, f"df_{pid}_*.pkl"))
        if len(flush_file_list) > 0:
            flush_idx = [int(x.split("_")[-1].split(".")[0]) for x in flush_file_list]
            last_flush_idx = max(flush_idx)
            record_list = record_list[last_flush_idx:]
            gc.collect()
            log.info(f"[Process-{pid}] Resuming from {last_flush_idx}th read. {len(record_list)} reads remaining.")
        else:
            log.info(f"[Process-{pid}] No flush file found. Starting from the beginning.")

    for read_idx, record in tqdm(enumerate(record_list), total=len(record_list)):
        read_idx += last_flush_idx
        read_id = record[0]
        seq = record[1].replace("T", "U")
        phred = record[2]

        cb_info_dict, dag_list, dag_dict = find_block_candidates(
            seq,
            phred,
            cb_bq_cutoff,
            spacer_kmer_ed_dict,
            skip_size_tolerance,
            cb_pad,
            cb_per_bb,
            indel_penalty,
            anchor_mismatch_penalty,
            spacer_mismatch_penalty,
            spacer_size,
            spacer_list,
            indel_dict,
            min_ideal_displacement_dict,
            anchor_list,
            score_converting_func,
            cb_size_tolerance,
            spacer_mismatch_tolerance,
            spacer_size_tolerance,
            bb_size,
        )

        if len(cb_info_dict) > 0:
            longest_path = dag_longest_path(dag_list)
            selected_cb = []
            total_score = 0
            for x, y in zip(longest_path[:-1], longest_path[1:]):
                if (x, y) in cb_info_dict:
                    selected_cb.append(cb_info_dict[(x, y)])
                total_score += dag_dict[(x, y)]

            selected_cb_df = pd.DataFrame(
                selected_cb, columns=["cb_idx", "start_pos", "end_pos", "pos_RM", "penalty", "score"]
            )
            selected_cb_df["read_id"] = read_id
            selected_cb_df["total_score"] = total_score

            spacer_pos = np.unique(selected_cb_df[["start_pos", "end_pos"]].values.flatten())
            spacer_phred = [phred[x : x + spacer_size] for x in spacer_pos]

            if len(spacer_phred) > 0:
                spacer_phred = np.concatenate(spacer_phred)
                mean_spacer_phred = np.mean(spacer_phred)

                selected_cb_df["mean_spacer_phred"] = mean_spacer_phred
                selected_cb_df["start_pos"] = selected_cb_df["pos_RM"] - cb_pad
                selected_cb_df["end_pos"] = selected_cb_df["pos_RM"] + cb_pad + 1
                selected_cb_df["motif"] = selected_cb_df.apply(lambda x: seq[x["start_pos"] : x["end_pos"]], axis=1)
                selected_cb_df["bq"] = selected_cb_df.apply(lambda x: phred[x["start_pos"] : x["end_pos"]], axis=1)
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
                block_df_flush = block_df_flush[
                    (block_df_flush["start_pos"] >= 0)
                    & (block_df_flush["bq_len"] == cb_size)
                    & (block_df_flush["motif_len"] == cb_size)
                    & (block_df_flush["mean_spacer_phred"] >= cb_bq_cutoff)
                ]

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


def extract_block(
    input,
    output,
    indel_tolerance,
    indel_penalty,
    cb_size_tolerance,
    skip_size_tolerance,
    anchor_mismatch_penalty,
    spacer_size_tolerance,
    spacer_mismatch_tolerance,
    max_read_length,
    spacer_mismatch_penalty,
    anchor_list,
    spacer_list,
    spacer_size,
    cb_pad,
    cb_per_bb,
    read_bq_cutoff,
    cb_bq_cutoff,
    flush_path,
    flush_interval,
    ncpu,
    resume,
    sample,
    **kwargs,
):
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

    def score_converting_func(x):
        return 1 - (x / (2 * max_cb_penalty))

    spacer_list = [x.replace("T", "U") for x in spacer_list]
    anchor_list = [x.replace("T", "U") for x in anchor_list]
    indel_dict = get_integer_partition(indel_tolerance, cb_size_tolerance)
    spacer_kmer_ed_dict = {i: get_ed_kmers(kmer, spacer_mismatch_tolerance) for i, kmer in enumerate(spacer_list)}
    assert indel_tolerance >= cb_size_tolerance

    max_cb_penalty = (
        anchor_mismatch_penalty + spacer_mismatch_penalty * spacer_mismatch_tolerance + indel_penalty * indel_tolerance
    )
    cb_size = 2 * cb_pad + 1
    bb_size = cb_size * cb_per_bb + spacer_size
    min_ideal_displacement_dict = get_min_ideal_displacement_dict(cb_per_bb, spacer_size, cb_size)

    record_list = []
    with pysam.AlignmentFile(input, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm(total=input_bam.mapped) as pbar:
            for idx, record in enumerate(input_bam):
                qscore = mean_phred(np.array(record.query_qualities, dtype=int))
                if qscore >= read_bq_cutoff:
                    read_length = record.query_length
                    if read_length <= max_read_length and read_length >= kwargs["min_read_length"]:
                        record_tuple = (
                            str(record.query_name),
                            str(record.query_sequence),
                            np.array(record.query_qualities),
                            int(read_length),
                        )
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
        proc = mp.Process(
            target=extract_blocks_from_read_list_mp_worker,
            args=(
                record_split_dict[pid],
                indel_penalty,
                cb_size_tolerance,
                skip_size_tolerance,
                anchor_mismatch_penalty,
                spacer_size_tolerance,
                spacer_mismatch_tolerance,
                spacer_mismatch_penalty,
                cb_pad,
                cb_per_bb,
                cb_bq_cutoff,
                indel_dict,
                spacer_kmer_ed_dict,
                anchor_list,
                spacer_list,
                spacer_size,
                bb_size,
                flush_path,
                pid,
                flush_interval,
                score_converting_func,
                cb_size,
                min_ideal_displacement_dict,
                resume,
            ),
        )
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    block_df_list = []
    for pid in range(ncpu):
        try:
            block_df = pd.read_pickle(f"{flush_path}df_{pid}.pkl")
            block_df_list.append(block_df)
        except Exception as e:
            log.warning(f"{e}")
            log.warning(f"PID {pid} did not return any result.")
    block_df = pd.concat(block_df_list, axis=0).reset_index(drop=True)
    del block_df_list
    gc.collect()

    block_df.to_pickle(f"{output}/block.pkl")

    dag_log = []
    dag_log.append(f"Total number of passed reads: {record_cnt:,}")
    dag_log.append(f"Total number of context blocks: {len(block_df):,}")
    dag_log.append(f"Context blocks per read: {len(block_df) / record_cnt:.2f}")
    dag_log.append(block_df["score"].describe())
    dag_log.append(block_df["penalty"].describe())

    log_path = f"{output}/dag_log.txt"
    with open(log_path, "w") as log_file:
        for line in dag_log:
            log_file.write(f"{line}\n")

    print("=============================================")
    for line in dag_log:
        log.info(line)
    print("=============================================")

    log.info(f"Saved context blocks to {output}/block.pkl.")
    return block_df
