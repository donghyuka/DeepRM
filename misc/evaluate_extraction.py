import pysam
import pandas as pd
import numpy as np
import tqdm
import gc
from utils.utils import mean_phred
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt



BLOCK_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/pentamer_block_flexible_kmer_v0326v10.pkl"
ALIGNMENT_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/intermediates/dorado_output.aligned.filtered.sorted.pentamer.subsampled.bam"
CORRECT_POSITION = [3+87*n+i for n in range(5) for i in [16, 27+16, 27+27+16]]

def read_bam(id_list):
    read_dict = {}
    with pysam.AlignmentFile(ALIGNMENT_PATH, "rb") as bam:
        for read in tqdm.tqdm(bam, total=bam.mapped):
            read_dict[read.query_name] = read
    read_dict_selected = {}
    for id in tqdm.tqdm(id_list):
        try:
            read_dict_selected[id] = read_dict[id]
        except KeyError:
            read_dict_selected[id] = None
    align_pair_dict = {}
    for id, read in tqdm.tqdm(read_dict_selected.items()):
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

def get_block_df(align_pair_dict, block_df):
    block_df["align_pairs"] = block_df["read_id"].map(align_pair_dict)
    block_df = block_df.dropna()
    block_df.to_pickle(BLOCK_PATH+"_with_align_pairs.pkl")
    return block_df

def check_if_correct_position(block_df):
    block_df["ref_anchor"] = block_df.apply(lambda x: x["align_pairs"][x["pos_RM"]], axis = 1)
    block_df["correct"] = block_df["ref_anchor"].apply(lambda x: x in CORRECT_POSITION)
    block_df.to_pickle(BLOCK_PATH+"_with_align_pairs_correct.pkl")
    print(block_df[["read_id","start_pos","pos_RM","ref_anchor","motif","correct"]])
    print(block_df["correct"].value_counts())
    return None

def main():
    block_df = pd.read_pickle(BLOCK_PATH)
    # ## Downsample
    # id_list = np.random.choice(id_list, 100000, replace=False)
    # block_df = block_df[block_df["read_id"].isin(id_list)]
    # block_df.to_pickle(BLOCK_PATH+"_downsampled.pkl")
    # gc.collect()
    # block_df = pd.read_pickle(BLOCK_PATH+"_downsampled.pkl")
    id_list = block_df["read_id"].unique()
    align_pair_dict = read_bam(id_list)
    block_df = get_block_df(align_pair_dict, block_df)
    check_if_correct_position(block_df)
    plot_pr_curve(block_df)
    return None


def plot_pr_curve(block_df):
    x = block_df["score"]
    y = block_df["correct"]
    precision, recall, _ = precision_recall_curve(y, x)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/plot/pr_curve_flex.png", dpi=300)
    return None


if __name__ == "__main__":
    main()
