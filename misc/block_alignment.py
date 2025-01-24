import numpy as np
import pandas as pd
import pysam
import os
import multiprocessing as mp


def get_alignment_accuracy(align_pairs, pos_rm, ref_seq, query_seq):
    pos_ref = np.array([align_pairs[x] for x in np.arange(pos_rm-10, pos_rm+11)])
    mismatches = np.array(list(query_seq)) != np.array([ref_seq[x] if x is not None else "X" for x in pos_ref])
    insertions = pd.isna(pos_ref)
    deletions = np.diff(pd.Series(pos_ref).ffill().bfill()) > 1
    deletions = np.concatenate([deletions[:10], [False], deletions[10:]])
    ins_del_mis = np.stack([insertions, deletions, mismatches], axis = 1)
    return ins_del_mis

def worker(block_df, ref_seq, flush_path):
    alignment_arr = block_df.apply(lambda x: get_alignment_accuracy(x["align_pairs"], x["pos_RM"], ref_seq, x["motif"]), axis = 1)
    alignment_arr = np.stack(alignment_arr.values)
    np.save(flush_path, alignment_arr)
    return None

def main(threads = 120):

    out_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/alignment"

    block_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/dag/block_df_with_align_pairs_correct.pkl")
    ref_fasta = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/reference/ON0095_ref.fasta"
    ref_fasta = pysam.FastaFile(ref_fasta)
    ref_seq = ref_fasta.fetch("pentamer")
    ref_seq = ref_seq.upper().replace("T", "U")
    ref_fasta.close()

    block_df = block_df[block_df["correct"] & (block_df["penalty"]==0)].copy()
    block_df = np.array_split(block_df, threads)

    flush_path = os.path.join(out_path, "temp")
    os.makedirs(flush_path, exist_ok=True)

    proc_list = []
    for i, block_df_split in enumerate(block_df):
        proc = mp.Process(target=worker, args=(block_df_split, ref_seq, os.path.join(flush_path, f"{i}.npy")))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    alignment_arr = np.concatenate([np.load(os.path.join(flush_path, f"{i}.npy")) for i in range(threads)])
    np.save(os.path.join(out_path, "alignment.npy"), alignment_arr)

    return None

if __name__ == "__main__":
    main()






