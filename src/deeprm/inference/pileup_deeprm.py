"""
DeepRM Pileup (Post-Processing) Module

This script performs post-processing on DeepRM prediction files to generate a pileup.
It reads .npz prediction arrays, groups statistics by label IDs, and computes metrics.

The two metrics calculated are:
1. modscore: A score reflecting the site-level modification probability. (arbitrary units)
2. stoichiometry: Estimated modification stoichiometry of the site. (0-1 range)

Finally, it writes a .npz file containing the results.
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os
import shutil
import uuid

import numpy as np
import pandas as pd
import pysam
import tqdm

from deeprm.inference.pileup_genomic import pileup_genomic

mp.set_start_method("fork", force=True)


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.
    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    parser.add_argument("--input", "-i", type=str, required=True, help="Input (predictions) path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output (pileup) path")
    parser.add_argument("--thread", "-t", type=int, default=None, help="Number of threads to use")
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file path")
    parser.add_argument("--threshold", "-th", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--epsilon", "-e", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Comment")
    parser.add_argument("--slice", "-s", type=int, default=None, help="Slice index (for 2D predictions)")
    parser.add_argument("--flip", "-f", action="store_true", help="Flip label")
    parser.add_argument(
        "--label_div", "-d", type=int, default=10**9, help="Divisor for label_id to separate transcript and position"
    )
    parser.add_argument("--annot", "-a", type=str, default=None, help="Annotation file (e.g., refFlat.txt)")

    return None


def main(args: argparse.Namespace):
    """
    Main function: spawns worker processes, aggregates results, computes final metrics,
    and writes output .npz file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None: Results are saved to a .npz file in the specified output directory.
    """
    import time

    start = time.time()
    if args.thread is None:
        args.thread = max(1, int(0.95 * mp.cpu_count()))

    os.makedirs(args.output, exist_ok=True)

    ## Define keys for shared data storage
    keys = ["logsum_1_p_pos", "kl_div_neg", "kl_div_pos", "count_all", "count_pos", "label_id"]
    modbam_keys = ["read_id_high", "read_id_low", "ref_id", "pos", "pred"]

    ## Gather all prediction files and split them for multiprocessing
    file_paths = glob.glob(os.path.join(args.input, "*.npz"))
    file_paths_split = np.array_split(file_paths, min(args.thread, len(file_paths)))

    ## Create a shared dictionary to store results from all processes
    manager = mp.Manager()
    shared_dict = manager.dict()
    for key in keys:
        shared_dict[key] = manager.dict()
    shared_dict["modbam_data"] = manager.dict()

    ## Start worker processes to process each chunk of files
    proc_list = []
    for pid, file_paths in enumerate(file_paths_split):
        proc = mp.Process(
            target=worker,
            args=(
                pid,
                file_paths,
                keys,
                shared_dict,
                args.label_div,
                args.slice,
                args.threshold,
                args.epsilon,
                args.flip,
            ),
        )
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    gc.collect()
    modbam_data = (
        pd.concat([shared_dict["modbam_data"][pid] for pid in range(len(file_paths_split))], axis=0)
        .groupby(["ref_id", "read_id_high", "read_id_low"])
        .agg({"pos": "sum", "pred": "sum"})
    )
    modbam_out_path = os.path.join(args.output, "modbam_" + os.path.basename(args.bam))
    write_modbam(args.bam, modbam_out_path, modbam_data, args.thread)

    ## find unique label across all chunks
    all_ids = np.concatenate([shared_dict["label_id"][pid] for pid in range(len(file_paths_split))])
    global_ids = np.unique(all_ids)
    n_label_id = len(global_ids)

    ## pre-allocate accumulators
    final_count_all = np.zeros(n_label_id, dtype=np.int32)
    final_count_pos = np.zeros(n_label_id, dtype=np.int32)
    final_logsum = np.zeros(n_label_id, dtype=np.float32)
    final_kl_neg = np.zeros(n_label_id, dtype=np.float32)
    final_kl_pos = np.zeros(n_label_id, dtype=np.float32)

    ## vectorized accumulation (because the label_id is already unique for each chunk)
    for pid in tqdm.tqdm(range(len(file_paths_split)), desc="Accumulating data", leave=False):
        label_idx = np.searchsorted(global_ids, shared_dict["label_id"][pid])
        final_count_all[label_idx] += shared_dict["count_all"][pid]
        final_count_pos[label_idx] += shared_dict["count_pos"][pid]
        final_logsum[label_idx] += shared_dict["logsum_1_p_pos"][pid]
        final_kl_neg[label_idx] += shared_dict["kl_div_neg"][pid]
        final_kl_pos[label_idx] += shared_dict["kl_div_pos"][pid]

    ## extract only the IDs that were seen
    unique_id = np.nonzero(final_count_all > 0)[0]

    ## slice to compact arrays
    label_id = np.ascontiguousarray(global_ids[unique_id])
    count_all = np.ascontiguousarray(final_count_all[unique_id])
    count_pos = np.ascontiguousarray(final_count_pos[unique_id])
    logsum_1_p_pos = np.ascontiguousarray(final_logsum[unique_id])
    kl_div_neg = np.ascontiguousarray(final_kl_neg[unique_id])
    kl_div_pos = np.ascontiguousarray(final_kl_pos[unique_id])

    ## Calculate modscore and stoichiometry metrics
    stoichiometry = kl_div_pos / (kl_div_neg + kl_div_pos + args.epsilon)
    modscore = -(2 - stoichiometry) * logsum_1_p_pos / count_all + (
        (1 - stoichiometry) * np.log10(np.clip(1 - stoichiometry, 1e-30, 1))
        + stoichiometry * np.log10(np.clip(stoichiometry, 1e-30, 1))
    ) * (count_pos / count_all)

    ## Read BAM Header to get reference names
    input_bam = pysam.AlignmentFile(args.bam, "rb", check_sq=False, threads=args.thread)
    ref_arr = np.array(input_bam.references)
    input_bam.close()

    ## Convert label_id to ref_names, ref_pos, and ref_strand
    ref_strand = np.sign(label_id)
    label_id_abs = np.abs(label_id) - 1  ## 1 was added during preprocessing to avoid zero label_id
    transcript_id = label_id_abs // args.label_div
    ref_pos = label_id_abs % args.label_div
    ref_names = ref_arr[transcript_id]  ## Map transcript_id to reference names with vectorized operation

    ## Format results into a BED-like structure
    path = f"{args.output}/pileup.bed"
    bed_formatter(
        ref_names=ref_names,
        ref_pos=ref_pos,
        ref_strand=ref_strand,
        modscore=modscore,
        stoichiometry=stoichiometry,
        count_all=count_all,
        count_pos=count_pos,
        output_path=path,
    )

    ## Save results to compressed .npz
    path = f"{args.output}/pileup.npz"
    np.savez_compressed(
        path,
        ref_names=ref_names,
        ref_pos=ref_pos,
        ref_strand=ref_strand,
        modscore=modscore,
        stoichiometry=stoichiometry,
        count_all=count_all,
        count_pos=count_pos,
    )

    if args.annot:
        ## Generate genomic pileup if annotation is provided
        input_df = {
            "ref_names": ref_names,
            "ref_pos": ref_pos,
            "count_all": count_all,
            "count_pos": count_pos,
            "kl_div_pos": kl_div_pos,
            "kl_div_neg": kl_div_neg,
            "logsum_1_p_pos": logsum_1_p_pos,
        }
        input_df = pd.DataFrame(input_df)
        genomic_df = pileup_genomic(args, input_df)

        path = f"{args.output}/genomic_pileup{args.postfix}.bed"
        bed_formatter(
            ref_names=genomic_df["chrom"].values,
            ref_pos=genomic_df["pos"].values,
            ref_strand=genomic_df["strand"].values,
            modscore=genomic_df["modscore"].values,
            stoichiometry=genomic_df["stoichiometry"].values,
            count_all=genomic_df["count_all"].values,
            count_pos=genomic_df["count_pos"].values,
            output_path=path,
        )

        path = f"{args.output}/genomic_pileup{args.postfix}.npz"
        np.savez_compressed(
            path,
            ref_names=genomic_df["chrom"].values,
            ref_pos=genomic_df["pos"].values,
            ref_strand=genomic_df["strand"].values,
            modscore=genomic_df["modscore"].values,
            stoichiometry=genomic_df["stoichiometry"].values,
            count_all=genomic_df["count_all"].values,
            count_pos=genomic_df["count_pos"].values,
        )
    ##############
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds.")
    return None


def grouped_sum(n_unique, idx, vals):
    """
    Sum values in 'vals' according to group indices 'idx'.

    Args:
        n_unique (int): Number of unique groups.
        idx (numpy.ndarray): Integer indices mapping each element in 'vals' to a group.
        vals (numpy.ndarray): Values to sum per group.

    Returns:
        numpy.ndarray: Array of summed values of length n_unique.
    """
    group_sums = np.zeros((n_unique,), dtype=vals.dtype)
    np.add.at(group_sums, idx, vals)
    return group_sums


def worker(pid, file_paths, keys, shared_dict, label_div, slice=None, threshold_pos=0.98, epsilon=1e-30, flip=False):
    """
    Worker function to process a subset of prediction files.
    Computes per-label statistics and stores results in a shared dictionary.

    Args:
        pid (int): Process ID for indexing results.
        file_paths (list): List of .npz input file paths.
        keys (list): List of data keys to compute/store.
        shared_dict (dict): Shared structure for results.
        slice (int): Column for 2D predictions. Defaults to None. (optional)
        threshold_pos (float): Threshold to count positive predictions.
        epsilon (float): Small constant for log and division safety.
        flip (bool): Whether to invert probabilities (1 - p).

    Returns:
        None: Results are stored in shared_dict.
    """

    ## Initialize container dictionary for this process
    data_dict = {k: [] for k in keys}
    modbam_data = []

    ## Iterate over prediction files
    for idx, path in enumerate(tqdm.tqdm(file_paths, desc="Reading input files", leave=False)):
        modbam_chunk = {}
        with np.load(path) as data:
            pred = data["pred"]  # prediction probabilities
            label_id = data["label_id"]  # integer labels for each prediction
            read_id = data["read_id"]
        label_id_abs = np.abs(label_id) - 1
        modbam_chunk["read_id_high"] = read_id[:, 0]
        modbam_chunk["read_id_low"] = read_id[:, 1]
        modbam_chunk["ref_id"] = label_id_abs // label_div
        modbam_chunk["pos"] = label_id_abs % label_div
        modbam_chunk["pred"] = (pred * 256).astype(np.uint8).clip(0, 255)
        modbam_chunk = pd.DataFrame(modbam_chunk)
        modbam_chunk = modbam_chunk.groupby(["ref_id", "read_id_high", "read_id_low"]).agg({"pos": list, "pred": list})
        modbam_data.append(modbam_chunk)

        ## Ensure correct dtypes
        assert pred.dtype == np.float32, f"Expected pred to be int32, but got {pred.dtype} in {path}"
        assert (
            label_id.dtype == np.int64 or label_id.dtype == np.uint64
        ), f"Expected label_id to be int64, but got {label_id.dtype} in {path}"
        assert len(pred) == len(label_id), f"Length of pred and label_id do not match in {path}"

        ## Filter out any NaN or infinite values
        valid_idx = np.isfinite(pred) & np.isfinite(label_id)
        pred = pred[valid_idx]
        label_id = label_id[valid_idx]

        ## Slice 2D predictions if requested
        if slice is not None:
            assert pred.ndim == 2, f"Expected pred to be 2D as slice was given, but got {pred.ndim} in {path}"
            pred = pred[:, slice]
        else:
            assert pred.ndim == 1, f"Expected pred to be 1D as slice was not given, but got {pred.ndim} in {path}"

        ## Optionally invert probabilities
        if flip:
            pred = 1 - pred

        ## Sanity-check range of predictions
        pred_min = np.min(pred)
        pred_max = np.max(pred)
        assert pred_min >= 0.0, f"Minimum value of pred is {pred_min} in {path}"
        assert pred_max <= 1.0, f"Maximum value of pred is {pred_max} in {path}"

        ## Calculate count of positive predictions
        count_pos = (pred >= threshold_pos).astype(np.int32)

        ## Calculate sum(log10(1 - p)) of positive predictions
        logsum_1_p_pos = np.log10(np.clip(1 - pred, epsilon, 1.0)) * count_pos

        ## Calculate KL divergence
        kl_div = pred * np.log2(2 * pred + epsilon) + (1 - pred) * np.log2(2 * (1 - pred) + epsilon)
        kl_div_neg = kl_div * (pred <= 0.5)
        kl_div_pos = kl_div * (pred > 0.5)

        ## Aggregate data by label_id
        unique_id, id_idx, count_all = np.unique(label_id, return_inverse=True, return_counts=True)
        n_unique = len(unique_id)
        data_dict["label_id"].append(unique_id)
        data_dict["count_all"].append(count_all.astype(np.int32))
        data_dict["count_pos"].append(grouped_sum(n_unique, id_idx, count_pos))
        data_dict["logsum_1_p_pos"].append(grouped_sum(n_unique, id_idx, logsum_1_p_pos))
        data_dict["kl_div_neg"].append(grouped_sum(n_unique, id_idx, kl_div_neg))
        data_dict["kl_div_pos"].append(grouped_sum(n_unique, id_idx, kl_div_pos))

    modbam_data = (
        pd.concat(modbam_data, axis=0)
        .groupby(["ref_id", "read_id_high", "read_id_low"])
        .agg({"pos": "sum", "pred": "sum"})
    )

    shared_dict["modbam_data"][pid] = modbam_data

    del modbam_data
    gc.collect()

    ## Combine chunked results for this process
    all_ids = np.concatenate(data_dict["label_id"])
    global_ids = np.unique(all_ids)
    n_label_id = len(global_ids)

    ## pre-allocate accumulators
    final_count_all = np.zeros(n_label_id, dtype=np.int32)
    final_count_pos = np.zeros(n_label_id, dtype=np.int32)
    final_logsum = np.zeros(n_label_id, dtype=np.float32)
    final_kl_neg = np.zeros(n_label_id, dtype=np.float32)
    final_kl_pos = np.zeros(n_label_id, dtype=np.float32)

    ## vectorized accumulation (because the label_id is already unique for each chunk)
    for chunk_idx in tqdm.tqdm(range(len(data_dict["label_id"])), desc="Accumulating data", leave=False):
        label_idx = np.searchsorted(global_ids, data_dict["label_id"][chunk_idx])
        final_count_all[label_idx] += data_dict["count_all"][chunk_idx]
        final_count_pos[label_idx] += data_dict["count_pos"][chunk_idx]
        final_logsum[label_idx] += data_dict["logsum_1_p_pos"][chunk_idx]
        final_kl_neg[label_idx] += data_dict["kl_div_neg"][chunk_idx]
        final_kl_pos[label_idx] += data_dict["kl_div_pos"][chunk_idx]

    ## extract only the IDs that were seen
    unique_id = np.nonzero(final_count_all > 0)[0]

    ## slice to compact arrays
    label_id = np.ascontiguousarray(global_ids[unique_id])
    count_all = np.ascontiguousarray(final_count_all[unique_id])
    count_pos = np.ascontiguousarray(final_count_pos[unique_id])
    logsum_1_p_pos = np.ascontiguousarray(final_logsum[unique_id])
    kl_div_neg = np.ascontiguousarray(final_kl_neg[unique_id])
    kl_div_pos = np.ascontiguousarray(final_kl_pos[unique_id])

    ## Store in shared dictionary
    shared_dict["label_id"][pid] = label_id
    shared_dict["count_all"][pid] = count_all
    shared_dict["count_pos"][pid] = count_pos
    shared_dict["logsum_1_p_pos"][pid] = logsum_1_p_pos
    shared_dict["kl_div_neg"][pid] = kl_div_neg
    shared_dict["kl_div_pos"][pid] = kl_div_pos

    return None


def bed_formatter(
    ref_names,
    ref_pos,
    ref_strand,
    modscore,
    stoichiometry,
    count_all,
    count_pos,
    output_path,
):
    """
    Formats the results into a BED-like structure.

    Args:
        ref_names (numpy.ndarray): Array of reference names.
        ref_pos (numpy.ndarray): Array of reference positions.
        ref_strand (numpy.ndarray): Array of reference strands.
        modscore (numpy.ndarray): modscore scores.
        stoichiometry (numpy.ndarray): stoichiometry scores.
        count_all (numpy.ndarray): Total counts.
        count_pos (numpy.ndarray): Positive counts.

    Returns:
        list: List of formatted strings for each entry.
    """

    col1 = ref_names
    col2 = ref_pos
    col3 = ref_pos + 1
    col4 = ["a"] * len(ref_names)
    col5 = np.clip((modscore * 100).astype(int), 0, 1000)
    col6 = ref_strand
    col7 = ref_pos
    col8 = ref_pos + 1
    col9 = ["255,0,0"] * len(ref_names)
    col10 = count_all
    col11 = stoichiometry * 100
    col12 = (count_all * stoichiometry).astype(int)
    col13 = count_all - stoichiometry
    col14 = np.zeros(len(ref_names))
    col15 = np.zeros(len(ref_names))
    col16 = np.zeros(len(ref_names))
    col17 = np.zeros(len(ref_names))
    col18 = np.zeros(len(ref_names))

    df = pd.DataFrame(
        {
            "col1": col1,
            "col2": col2,
            "col3": col3,
            "col4": col4,
            "col5": col5,
            "col6": col6,
            "col7": col7,
            "col8": col8,
            "col9": col9,
            "col10": col10,
            "col11": col11,
            "col12": col12,
            "col13": col13,
            "col14": col14,
            "col15": col15,
            "col16": col16,
            "col17": col17,
            "col18": col18,
        }
    )

    df.to_csv(output_path, sep="\t", header=False, index=False, float_format="%.2f")
    return None


def get_mm_tag(q_pos, preds, seq, base="A", mod="a"):
    pred_dict = dict(zip(q_pos, preds))
    base_positions = np.where(np.array(list(seq)) == base)[0]
    pred_values = np.array([pred_dict.get(pos, 0) for pos in base_positions])
    run_lengths = np.ediff1d(np.concatenate(([True], pred_values > 0, [True])).nonzero()[0]) - 1
    mm_tag = f"{base}+{mod},{','.join(map(str, run_lengths))}"
    preds = np.array(preds)
    ml_tag = preds[preds > 0].tolist()
    if not ml_tag:
        ml_tag = [0]
    return mm_tag, ml_tag


def write_modbam(in_path, out_path, data, threads):
    intermediate_dir = out_path + ".shard"
    os.makedirs(intermediate_dir, exist_ok=True)

    proc_list = []
    for i, sub_data in enumerate(np.array_split(data, threads)):
        out_path_proc = os.path.join(intermediate_dir, f"{i}.bam")
        proc = mp.Process(target=write_modbam_worker, args=(in_path, out_path_proc, sub_data))
        proc_list.append(proc)
    for proc in proc_list:
        proc.start()
    for proc in proc_list:
        proc.join()

    unsorted_path = out_path + ".unsorted.bam"
    pysam.merge(f"-@ {threads} -f", unsorted_path, *glob.glob(os.path.join(intermediate_dir, "*.bam")))
    pysam.sort(f"-@ {threads}", "-m 4G", "-o", out_path, unsorted_path)
    pysam.index(out_path)

    shutil.rmtree(intermediate_dir)
    os.remove(unsorted_path)

    return None


def write_modbam_worker(in_path, out_path, data):
    in_bam = pysam.AlignmentFile(in_path, "rb")
    out_bam = pysam.AlignmentFile(out_path, "wb", template=in_bam)
    for read in tqdm.tqdm(in_bam, total=in_bam.mapped + in_bam.unmapped):
        read_id = str(read.get_tag("pi")) if read.has_tag("pi") else str(read.query_name)
        read_id_high, read_id_low = np.frombuffer(uuid.UUID(read_id).bytes, dtype=np.int64)
        ref_id = read.reference_id
        try:
            data_read = data.loc[ref_id, read_id_high, read_id_low]
        except KeyError:
            continue
        mapping = read.get_aligned_pairs()
        mapping = {i[1]: i[0] for i in mapping if i[1] is not None}
        q_pos = [mapping.get(i, None) for i in data_read["pos"]]
        mm_tag, ml_tag = get_mm_tag(q_pos, data_read["pred"], str(read.query_sequence))
        read.set_tag("MM", mm_tag, "Z")
        read.set_tag("ML", ml_tag)
        out_bam.write(read)
    in_bam.close()
    out_bam.close()
    return None
