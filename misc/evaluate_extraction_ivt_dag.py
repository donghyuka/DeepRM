import pysam
import pandas as pd
import numpy as np
import tqdm
import gc
from utils.utils import mean_phred, printmessage
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import time
import multiprocessing as mp
import glob
import pickle
import argparse

plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 1, "legend.frameon": 1, "lines.linewidth": 2})


def is_correct_position(ref_pos):
    return (ref_pos-3)%87%27 == 16 and ref_pos < 435


def read_bam(id_list_dict, wdir, alignment_path):

    if os.path.exists(os.path.join(wdir, "align_pair_dict.pkl")):
        with open(os.path.join(wdir, "align_pair_dict.pkl"), "rb") as f:
            align_pair_dict = pickle.load(f)
        with open(os.path.join(wdir, "total_cb_count.txt"), "r") as f:
            total_cb_count = wdir(f.read())

    else:
        read_dict = {}
        with pysam.AlignmentFile(alignment_path, "rb", threads=32) as bam:
            for read in tqdm.tqdm(bam, total=bam.mapped, desc = "Reading BAM"):
                read_dict[read.query_name] = read

        align_pair_dict = {}
        total_cb_count = 0

        for id, read in tqdm.tqdm(read_dict.items(), desc = "Getting aligned pairs"):
            aligned_pair = read.get_aligned_pairs()
            aligned_pair_as_dict = {x[0]: x[1] for x in aligned_pair}
            for i in range(read.query_length):
                if i not in aligned_pair_as_dict:
                    aligned_pair_as_dict[i] = None
            align_pair_dict[id] = aligned_pair_as_dict

            for q, r in aligned_pair:
                if r is not None and q is not None:
                    if is_correct_position(r):
                        if q >= 10 and q < read.query_length - 10:
                            if read.query_sequence[q] == "A":
                                total_cb_count += 1

        with open(os.path.join(wdir, "total_cb_count.txt"), "w") as f:
            f.write(str(total_cb_count))
        with open(os.path.join(wdir, "align_pair_dict.pkl"), "wb") as f:
            pickle.dump(align_pair_dict, f)

    align_pair_dict_dict = {}
    for i, id_list in tqdm.tqdm(id_list_dict.items(), desc = "Splitting"):
        align_pair_dict_dict[i] = {id: align_pair_dict[id] for id in id_list}

    return align_pair_dict_dict, total_cb_count


def get_pos_rm(pos_RM, align_pair):
    if pos_RM is None:
        return None
    try:
        return align_pair[pos_RM]
    except:
        return None


def get_block_df(align_pair_dict, block_df, output_path):
    block_df["align_pairs"] = block_df["read_id"].map(align_pair_dict)
    block_df = block_df.dropna().copy()
    block_df["ref_anchor"] = block_df.apply(lambda x: get_pos_rm(x["pos_RM"], x["align_pairs"]), axis = 1)
    block_df = block_df.dropna().copy()
    block_df["correct"] = block_df["ref_anchor"].apply(is_correct_position)
    block_df.to_pickle(output_path)
    return None


def plot_pr_curve(block_df, total_count, wdir):
    x = 100 - block_df["penalty"]
    y = block_df["correct"]
    precision, recall, _ = precision_recall_curve(y, x)
    recall = recall * (np.sum(y) / total_count)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.savefig(os.path.join(wdir, "pr_curve.pdf"))
    return None


def main(threads = 120):

    args = parse_args()
    wdir = args.wdir
    alignment_path = args.alignment

    flush_path = os.path.join(wdir, f"block_flush_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(flush_path, exist_ok=True)

    block_df = pd.read_pickle(os.path.join(wdir, "block_df.pkl"))
    block_df_split = np.array_split(block_df, threads)
    printmessage("Loaded block_df")

    del block_df
    gc.collect()

    id_list_dict = {i: x["read_id"].unique() for i, x in enumerate(block_df_split)}
    align_pair_dict_dict, total_cb_count = read_bam(id_list_dict, wdir, alignment_path)

    printmessage("Finished reading BAM")
    printmessage(f"Total CB count: {total_cb_count}")

    proc_list = []
    for i, block_df in enumerate(block_df_split):
        proc = mp.Process(target=get_block_df, args=(align_pair_dict_dict[i], block_df, os.path.join(flush_path, f"{i}.pkl")))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    del align_pair_dict_dict, block_df_split, id_list_dict, block_df
    gc.collect()

    block_df_list = []
    for path in tqdm.tqdm(glob.glob(f"{flush_path}/*.pkl"), desc="Merging outputs"):
        block_df_list.append(pd.read_pickle(path))

    block_df = pd.concat(block_df_list)
    block_df.to_pickle(os.path.join(wdir, "block_df_with_align_pairs_correct.pkl"))
    print(block_df)
    print(block_df["correct"].value_counts())

    plot_pr_curve(block_df, total_cb_count, wdir)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate extraction")
    parser.add_argument("--wdir", "-w", dest="wdir", type=str, required=True)
    parser.add_argument("--alignment", "-a", dest="alignment", type=str, default = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/intermediates/dorado_basecalled.sorted.pentamer.subsampled.bam")
    return parser.parse_args()


if __name__ == "__main__":
    main()