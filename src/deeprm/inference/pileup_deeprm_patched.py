"""
DeepRM Pileup (Post-Processing) Module.
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
    """Adds command-line arguments."""
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
    parser.add_argument("--skip-modbam", "-sm", action="store_true", help="Skip modBAM writing and only output BED")
    return None


def _validate_args(args: argparse.Namespace) -> None:
    if args.thread is not None and args.thread <= 0:
        raise ValueError("--thread must be a positive integer.")
    if not (0.0 <= args.threshold < 1.0):
        raise ValueError("--threshold must satisfy 0 <= threshold < 1.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.label_div <= 0:
        raise ValueError("--label_div must be positive.")
    if args.slice is not None and args.slice < 0:
        raise ValueError("--slice must be >= 0.")


def main(args: argparse.Namespace):
    """Main pileup function."""
    import time

    start = time.time()
    _validate_args(args)
    if args.thread is None:
        args.thread = max(1, int(0.95 * mp.cpu_count()))

    os.makedirs(args.output, exist_ok=True)

    keys = ["logsum_1_p_pos", "kl_div_neg", "kl_div_pos", "count_all", "count_pos", "label_id"]

    file_paths = sorted(glob.glob(os.path.join(args.input, "*.npz")))
    if len(file_paths) == 0:
        raise ValueError(f"No inference .npz files found in: {args.input}")
    file_paths_split = np.array_split(file_paths, min(args.thread, len(file_paths)))

    manager = mp.Manager()
    shared_dict = manager.dict()
    for key in keys:
        shared_dict[key] = manager.dict()
    if not args.skip_modbam:
        shared_dict["modbam_data"] = manager.dict()

    proc_list = []
    for pid, file_subset in enumerate(file_paths_split):
        proc = mp.Process(
            target=worker,
            args=(
                pid,
                list(file_subset),
                keys,
                shared_dict,
                args.label_div,
                args.slice,
                args.threshold,
                args.epsilon,
                args.flip,
                not args.skip_modbam,
            ),
        )
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Pileup worker exited with code {proc.exitcode}.")
    gc.collect()

    if not args.skip_modbam:
        modbam_frames = [shared_dict["modbam_data"].get(pid) for pid in range(len(file_paths_split))]
        modbam_frames = [df for df in modbam_frames if df is not None and len(df) > 0]
        if modbam_frames:
            modbam_data = (
                pd.concat(modbam_frames, axis=0)
                .groupby(["ref_id", "read_id_high", "read_id_low"])
                .agg({"pos": "sum", "pred": "sum"})
            )
            modbam_out_path = os.path.join(args.output, "modbam_" + os.path.basename(args.bam))
            write_modbam(args.bam, modbam_out_path, modbam_data, args.thread)
        else:
            print("Warning: no valid modBAM entries were produced; skipping modBAM writing.")

    all_ids = [shared_dict["label_id"].get(pid) for pid in range(len(file_paths_split))]
    all_ids = [arr for arr in all_ids if arr is not None and len(arr) > 0]
    if not all_ids:
        raise ValueError("No valid pileup entries were produced from the inference outputs.")
    global_ids = np.unique(np.concatenate(all_ids))
    n_label_id = len(global_ids)

    final_count_all = np.zeros(n_label_id, dtype=np.int64)
    final_count_pos = np.zeros(n_label_id, dtype=np.int64)
    final_logsum = np.zeros(n_label_id, dtype=np.float64)
    final_kl_neg = np.zeros(n_label_id, dtype=np.float64)
    final_kl_pos = np.zeros(n_label_id, dtype=np.float64)

    for pid in tqdm.tqdm(range(len(file_paths_split)), desc="Accumulating data", leave=False):
        label_id = shared_dict["label_id"].get(pid)
        if label_id is None or len(label_id) == 0:
            continue
        label_idx = np.searchsorted(global_ids, label_id)
        final_count_all[label_idx] += shared_dict["count_all"][pid]
        final_count_pos[label_idx] += shared_dict["count_pos"][pid]
        final_logsum[label_idx] += shared_dict["logsum_1_p_pos"][pid]
        final_kl_neg[label_idx] += shared_dict["kl_div_neg"][pid]
        final_kl_pos[label_idx] += shared_dict["kl_div_pos"][pid]

    unique_id = np.nonzero(final_count_all > 0)[0]
    label_id = np.ascontiguousarray(global_ids[unique_id])
    count_all = np.ascontiguousarray(final_count_all[unique_id])
    count_pos = np.ascontiguousarray(final_count_pos[unique_id])
    logsum_1_p_pos = np.ascontiguousarray(final_logsum[unique_id])
    kl_div_neg = np.ascontiguousarray(final_kl_neg[unique_id])
    kl_div_pos = np.ascontiguousarray(final_kl_pos[unique_id])

    digitization = 1000
    stoichiometry = kl_div_pos / (kl_div_neg + kl_div_pos + args.epsilon)
    modscore = 1 - np.power(10, logsum_1_p_pos / count_all * (1 + np.power(10, 2 * (stoichiometry - 1))))
    modscore = np.digitize(modscore, np.linspace(0, 1, digitization + 1), right=True) / digitization
    stoichiometry = stoichiometry * ((np.log10(1 - args.threshold) * stoichiometry) > (logsum_1_p_pos / count_all))

    input_bam = pysam.AlignmentFile(args.bam, "rb", check_sq=False, threads=args.thread)
    ref_arr = np.array(input_bam.references)
    input_bam.close()

    ref_strand = np.where(label_id > 0, "+", "-")
    label_id_abs = np.abs(label_id) - 1
    transcript_id = label_id_abs // args.label_div
    if transcript_id.size and np.max(transcript_id) >= len(ref_arr):
        raise ValueError("label_id contains transcript IDs outside the BAM reference range.")
    ref_pos = label_id_abs % args.label_div
    ref_names = ref_arr[transcript_id]

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
        input_df = pd.DataFrame(
            {
                "ref_names": ref_names,
                "ref_pos": ref_pos,
                "count_all": count_all,
                "count_pos": count_pos,
                "kl_div_pos": kl_div_pos,
                "kl_div_neg": kl_div_neg,
                "logsum_1_p_pos": logsum_1_p_pos,
            }
        )
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

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds.")
    return None


def grouped_sum(n_unique, idx, vals):
    """Sum values in vals according to group indices idx."""
    group_sums = np.zeros((n_unique,), dtype=vals.dtype)
    np.add.at(group_sums, idx, vals)
    return group_sums


def _empty_worker_payload(shared_dict, pid, make_modbam):
    shared_dict["label_id"][pid] = np.empty(0, dtype=np.int64)
    shared_dict["count_all"][pid] = np.empty(0, dtype=np.int64)
    shared_dict["count_pos"][pid] = np.empty(0, dtype=np.int64)
    shared_dict["logsum_1_p_pos"][pid] = np.empty(0, dtype=np.float64)
    shared_dict["kl_div_neg"][pid] = np.empty(0, dtype=np.float64)
    shared_dict["kl_div_pos"][pid] = np.empty(0, dtype=np.float64)
    if make_modbam:
        shared_dict["modbam_data"][pid] = pd.DataFrame(columns=["pos", "pred"])


def worker(
    pid,
    file_paths,
    keys,
    shared_dict,
    label_div,
    slice_idx=None,
    threshold_pos=0.98,
    epsilon=1e-30,
    flip=False,
    make_modbam=True,
):
    """Worker function to process a subset of prediction files."""
    if len(file_paths) == 0:
        _empty_worker_payload(shared_dict, pid, make_modbam)
        return None

    data_dict = {k: [] for k in keys}
    modbam_data = []

    for path in tqdm.tqdm(file_paths, desc="Reading input files", leave=False):
        with np.load(path) as data:
            pred = np.asarray(data["pred"])
            label_id = np.asarray(data["label_id"])
            read_id = np.asarray(data["read_id"])

        assert pred.dtype == np.float32, f"Expected pred to be float32, but got {pred.dtype} in {path}"
        assert label_id.dtype in (
            np.int64,
            np.uint64,
        ), f"Expected label_id to be int64/uint64, but got {label_id.dtype} in {path}"
        assert pred.shape[0] == label_id.shape[0], f"Length of pred and label_id do not match in {path}"
        assert read_id.shape[0] == label_id.shape[0], f"Length of read_id and label_id do not match in {path}"

        if slice_idx is not None:
            assert pred.ndim == 2, f"Expected pred to be 2D as --slice was given, but got {pred.ndim} in {path}"
            if slice_idx >= pred.shape[1]:
                raise IndexError(f"Slice index {slice_idx} is out of bounds for pred with shape {pred.shape} in {path}")
            valid_idx = np.isfinite(label_id) & np.all(np.isfinite(pred), axis=1)
            pred = pred[valid_idx, slice_idx]
        else:
            assert pred.ndim == 1, f"Expected pred to be 1D as --slice was not given, but got {pred.ndim} in {path}"
            valid_idx = np.isfinite(label_id) & np.isfinite(pred)
            pred = pred[valid_idx]

        label_id = label_id[valid_idx]
        read_id = read_id[valid_idx]

        if flip:
            pred = 1 - pred

        if pred.size == 0:
            continue

        pred_min = float(np.min(pred))
        pred_max = float(np.max(pred))
        assert pred_min >= 0.0, f"Minimum value of pred is {pred_min} in {path}"
        assert pred_max <= 1.0, f"Maximum value of pred is {pred_max} in {path}"

        if make_modbam:
            label_id_abs = np.abs(label_id).astype(np.int64) - 1
            modbam_chunk = pd.DataFrame(
                {
                    "read_id_high": read_id[:, 0],
                    "read_id_low": read_id[:, 1],
                    "ref_id": label_id_abs // label_div,
                    "pos": label_id_abs % label_div,
                    "pred": np.clip(np.rint(pred * 255), 0, 255).astype(np.uint8),
                }
            )
            if len(modbam_chunk) > 0:
                modbam_chunk = modbam_chunk.groupby(["ref_id", "read_id_high", "read_id_low"]).agg(
                    {"pos": list, "pred": list}
                )
                modbam_data.append(modbam_chunk)

        count_pos = (pred >= threshold_pos).astype(np.int64)
        logsum_1_p_pos = np.log10(np.clip(1 - pred, epsilon, 1.0)) * count_pos
        kl_div = pred * np.log2(2 * pred + epsilon) + (1 - pred) * np.log2(2 * (1 - pred) + epsilon)
        kl_div_neg = kl_div * (pred <= 0.5)
        kl_div_pos = kl_div * (pred > 0.5)

        unique_id, id_idx, count_all = np.unique(label_id, return_inverse=True, return_counts=True)
        n_unique = len(unique_id)
        data_dict["label_id"].append(unique_id.astype(np.int64, copy=False))
        data_dict["count_all"].append(count_all.astype(np.int64, copy=False))
        data_dict["count_pos"].append(grouped_sum(n_unique, id_idx, count_pos))
        data_dict["logsum_1_p_pos"].append(grouped_sum(n_unique, id_idx, logsum_1_p_pos))
        data_dict["kl_div_neg"].append(grouped_sum(n_unique, id_idx, kl_div_neg))
        data_dict["kl_div_pos"].append(grouped_sum(n_unique, id_idx, kl_div_pos))

    if len(data_dict["label_id"]) == 0:
        _empty_worker_payload(shared_dict, pid, make_modbam)
        return None

    if make_modbam:
        if modbam_data:
            shared_dict["modbam_data"][pid] = (
                pd.concat(modbam_data, axis=0)
                .groupby(["ref_id", "read_id_high", "read_id_low"])
                .agg({"pos": "sum", "pred": "sum"})
            )
        else:
            shared_dict["modbam_data"][pid] = pd.DataFrame(columns=["pos", "pred"])

    all_ids = np.concatenate(data_dict["label_id"])
    global_ids = np.unique(all_ids)
    n_label_id = len(global_ids)

    final_count_all = np.zeros(n_label_id, dtype=np.int64)
    final_count_pos = np.zeros(n_label_id, dtype=np.int64)
    final_logsum = np.zeros(n_label_id, dtype=np.float64)
    final_kl_neg = np.zeros(n_label_id, dtype=np.float64)
    final_kl_pos = np.zeros(n_label_id, dtype=np.float64)

    for chunk_idx in tqdm.tqdm(range(len(data_dict["label_id"])), desc="Accumulating data", leave=False):
        label_idx = np.searchsorted(global_ids, data_dict["label_id"][chunk_idx])
        final_count_all[label_idx] += data_dict["count_all"][chunk_idx]
        final_count_pos[label_idx] += data_dict["count_pos"][chunk_idx]
        final_logsum[label_idx] += data_dict["logsum_1_p_pos"][chunk_idx]
        final_kl_neg[label_idx] += data_dict["kl_div_neg"][chunk_idx]
        final_kl_pos[label_idx] += data_dict["kl_div_pos"][chunk_idx]

    unique_id = np.nonzero(final_count_all > 0)[0]

    shared_dict["label_id"][pid] = np.ascontiguousarray(global_ids[unique_id])
    shared_dict["count_all"][pid] = np.ascontiguousarray(final_count_all[unique_id])
    shared_dict["count_pos"][pid] = np.ascontiguousarray(final_count_pos[unique_id])
    shared_dict["logsum_1_p_pos"][pid] = np.ascontiguousarray(final_logsum[unique_id])
    shared_dict["kl_div_neg"][pid] = np.ascontiguousarray(final_kl_neg[unique_id])
    shared_dict["kl_div_pos"][pid] = np.ascontiguousarray(final_kl_pos[unique_id])
    return None


def bed_formatter(ref_names, ref_pos, ref_strand, modscore, stoichiometry, count_all, count_pos, output_path):
    """Formats the results into a BED-like structure."""
    ref_pos = np.asarray(ref_pos, dtype=np.int64)
    count_all = np.asarray(count_all, dtype=np.int64)
    count_pos = np.asarray(count_pos, dtype=np.int64)
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    modscore = np.asarray(modscore, dtype=np.float64)

    ref_strand = np.asarray(ref_strand)
    if np.issubdtype(ref_strand.dtype, np.number):
        ref_strand = np.where(ref_strand > 0, "+", "-")
    else:
        ref_strand = ref_strand.astype(str)

    estimated_mod_count = np.clip(np.rint(count_all * stoichiometry).astype(np.int64), 0, count_all)
    estimated_unmod_count = count_all - estimated_mod_count

    df = pd.DataFrame(
        {
            "col1": ref_names,
            "col2": ref_pos,
            "col3": ref_pos + 1,
            "col4": ["a"] * len(ref_names),
            "col5": np.clip(np.rint(modscore * 1000).astype(int), 0, 1000),
            "col6": ref_strand,
            "col7": ref_pos,
            "col8": ref_pos + 1,
            "col9": ["255,0,0"] * len(ref_names),
            "col10": count_all,
            "col11": stoichiometry * 100,
            "col12": estimated_mod_count,
            "col13": estimated_unmod_count,
            "col14": np.zeros(len(ref_names), dtype=np.int64),
            "col15": np.zeros(len(ref_names), dtype=np.int64),
            "col16": np.zeros(len(ref_names), dtype=np.int64),
            "col17": np.zeros(len(ref_names), dtype=np.int64),
            "col18": np.zeros(len(ref_names), dtype=np.int64),
        }
    )
    df.to_csv(output_path, sep="\t", header=False, index=False, float_format="%.2f")
    return None


def get_mm_tag(q_pos, preds, seq, base="A", mod="a"):
    q_pos = np.asarray(q_pos, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.uint8)

    if q_pos.size == 0:
        return f"{base}+{mod}?,;", []

    order = np.argsort(q_pos, kind="stable")
    q_pos = q_pos[order]
    preds = preds[order]

    base_positions = np.fromiter((i for i, b in enumerate(seq) if b == base), dtype=int)
    if base_positions.size == 0:
        return f"{base}+{mod}?,;", []

    keep = np.isin(q_pos, base_positions)
    q_pos = q_pos[keep]
    preds = preds[keep]
    if q_pos.size == 0:
        return f"{base}+{mod}?,;", []

    idx = np.searchsorted(base_positions, q_pos)
    run_lengths = np.empty_like(idx)
    run_lengths[0] = idx[0]
    if len(idx) > 1:
        run_lengths[1:] = np.diff(idx) - 1
    mm_tag = f"{base}+{mod}?,{','.join(map(str, run_lengths))};"
    ml_tag = preds.tolist()
    return mm_tag, ml_tag


def write_modbam(in_path, out_path, data, threads):
    """Write modBAM using multiple shard workers."""
    if data is None or len(data) == 0:
        return None

    intermediate_dir = out_path + ".shard"
    os.makedirs(intermediate_dir, exist_ok=True)

    n_proc = max(1, min(int(threads), len(data)))
    proc_list = []
    shard_paths = []
    for i, sub_data in enumerate(np.array_split(data, n_proc)):
        if len(sub_data) == 0:
            continue
        out_path_proc = os.path.join(intermediate_dir, f"{i}.bam")
        shard_paths.append(out_path_proc)
        proc = mp.Process(target=write_modbam_worker, args=(in_path, out_path_proc, sub_data))
        proc_list.append(proc)
    for proc in proc_list:
        proc.start()
    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"modBAM worker exited with code {proc.exitcode}.")

    if not shard_paths:
        shutil.rmtree(intermediate_dir)
        return None

    unsorted_path = out_path + ".unsorted.bam"
    pysam.merge("-@", str(n_proc), "-f", unsorted_path, *shard_paths)
    pysam.sort("-@", str(n_proc), "-m", "4G", "-o", out_path, unsorted_path)
    pysam.index(out_path)

    shutil.rmtree(intermediate_dir)
    os.remove(unsorted_path)
    return None


def write_modbam_worker(in_path, out_path, data):
    """Write a modBAM shard for a subset of reads."""
    in_bam = pysam.AlignmentFile(in_path, "rb")
    out_bam = pysam.AlignmentFile(out_path, "wb", template=in_bam)
    total = (in_bam.mapped or 0) + (in_bam.unmapped or 0)
    for read in tqdm.tqdm(in_bam, total=total):
        read_id = read.query_name
        if read_id is None or read.query_sequence is None:
            continue
        try:
            read_id_high, read_id_low = np.frombuffer(uuid.UUID(read_id).bytes, dtype=np.int64)
        except (ValueError, AttributeError):
            continue
        ref_id = read.reference_id
        if ref_id < 0:
            continue

        try:
            data_read = data.loc[(ref_id, read_id_high, read_id_low)]
        except KeyError:
            continue

        mapping_rpos_to_qpos = {r: q for q, r in read.get_aligned_pairs() if r is not None and q is not None}

        qpos = []
        pred = []
        for r, p in zip(data_read["pos"], data_read["pred"]):
            q = mapping_rpos_to_qpos.get(int(r))
            if q is not None:
                qpos.append(q)
                pred.append(p)

        mm_tag, ml_tag = get_mm_tag(qpos, pred, str(read.query_sequence))
        if ml_tag:
            read.set_tag("MM", mm_tag, "Z")
            read.set_tag("ML", ml_tag)
            out_bam.write(read)
    in_bam.close()
    out_bam.close()
    return None
