import pysam
import pandas as pd
import numpy as np
import tqdm
import gc
from utils.utils import mean_phred
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from utils.utils import max_f1_score

PLOTPATH  ="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/plot/1cb_sp15_rand10.png"
BLOCK_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/1cb_sp15_rand10.pkl"
ALIGNMENT_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/intermediates/dorado_output.aligned.filtered.sorted.pentamer.subsampled.subsampled.bam"
# CORRECT_POSITION = [123+60*n+i for n in range(5) for i in [16, 27+16]]
CORRECT_POSITION = [72+51*n+25 for n in range(6)]

def read_bam(id_list):
    read_dict = {}
    with pysam.AlignmentFile(ALIGNMENT_PATH, "rb", threads=32) as bam:
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
    recall = recall * (np.sum(y) / (6*15475))
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.savefig(PLOTPATH, dpi=300)
    return None


def main2():
    bb87_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/pentamer_block.pkl_downsampled.pkl_with_align_pairs_correct.pkl"
    bb60_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/2cb_sp6_rigid.pkl_with_align_pairs_correct.pkl"
    bb66_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/2cb_sp8_rigid.pkl_with_align_pairs_correct.pkl"
    bb45_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/1cb_sp12_rigid.pkl_with_align_pairs_correct.pkl"
    bb49_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/1cb_sp12_rand12.pkl_with_align_pairs_correct.pkl"
    bb55_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/1cb_sp15_rand12.pkl_with_align_pairs_correct.pkl"
    bb51_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/block/1cb_sp15_rand10.pkl_with_align_pairs_correct.pkl"
    bb87_df = pd.read_pickle(bb87_path)
    bb60_df = pd.read_pickle(bb60_path)
    bb66_df = pd.read_pickle(bb66_path)
    bb45_df = pd.read_pickle(bb45_path)
    # bb49_df = pd.read_pickle(bb49_path)
    # bb55_df = pd.read_pickle(bb55_path)
    bb51_df = pd.read_pickle(bb51_path)
    df_dict = {"87BB: 21CB x 3 + 6SP x 4": bb87_df,
               "60BB: 21CB x 2 + 6SP x 3": bb60_df,
               "66BB: 21CB x 2 + 8SP x 3": bb66_df,
               "45BB: 21CB x 1 + 12SP x 2": bb45_df,
               "51BB: 21CB x 1 + 15SP x 2": bb51_df,}
    total_dict = {"87BB: 21CB x 3 + 6SP x 4": 15*100000,
                  "60BB: 21CB x 2 + 6SP x 3": 10*100000,
                  "66BB: 21CB x 2 + 8SP x 3": 10*100000,
                  "45BB: 21CB x 1 + 12SP x 2": 6*15475,
                  "51BB: 21CB x 1 + 15SP x 2": 6*15475,}
    colour_list = ["royalblue", "tomato", "forestgreen", "orchid", "darkgoldenrod", "turquoise"]
    plot_pr_curve_2(df_dict, colour_list, total_dict)
    return None


def plot_pr_curve_2(df_dict, colour_list, total_dict):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    for i, (name, df) in enumerate(df_dict.items()):
        x = df["score"].to_numpy()
        y = df["correct"].to_numpy()
        total = total_dict[name]
        x = np.concatenate([x, np.zeros(total - np.sum(y), dtype=float)])
        y = np.concatenate([y, np.ones(total - np.sum(y), dtype=int)], dtype=int)
        precision, recall, _ = precision_recall_curve(y, x)
        max_f1 = max_f1_score(y, x)
        ax.plot(recall[1:], precision[1:], label=f"{name} (Max-F1 = {max_f1:.3f})", color=colour_list[i], linewidth=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/plot/pr_curve_compare_v4.png", dpi=300)
    return None


if __name__ == "__main__":
    # main()
    main2()
