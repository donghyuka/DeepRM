import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import sys
import glob
import tqdm
import gc
import multiprocessing as mp


def worker(pid, paths, fp_id, tp_id):
    k_list = ["label_id", "segment_len_arr", "signal_token", "kmer_token", "dwell_token"]
    fp_data_dict = {k:[] for k in k_list}
    tp_data_dict = {k:[] for k in k_list}

    for path in tqdm.tqdm(paths):
        with np.load(path, allow_pickle=True) as npz:
            data = {k:npz[k] for k in k_list}

        fp_idx_bool = np.isin(data["label_id"], fp_id)
        tp_idx_bool = np.isin(data["label_id"], tp_id)

        for k in k_list:
            fp_data_dict[k].append(data[k][fp_idx_bool])
            tp_data_dict[k].append(data[k][tp_idx_bool])


    fp_data_dict = {k:np.concatenate(v) for k,v in fp_data_dict.items()}
    tp_data_dict = {k:np.concatenate(v) for k,v in tp_data_dict.items()}

    np.savez_compressed(f"/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_analysis/fp_data_{pid}.npz", **fp_data_dict)
    np.savez_compressed(f"/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_analysis/tp_data_{pid}.npz", **tp_data_dict)

    return None

def main():
    label_df = pd.read_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.tsv", sep="\t")
    label_df = label_df[["id", "depth", "label", "m6A_level", "5mer", "drach"]]
    label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    label_df_filtered = label_df.copy()
    label_df_filtered[(label_df_filtered["dom_label"] == 0.0) & (label_df_filtered["label"] == 1)]["label"] = -1
    label_df_filtered[(label_df_filtered["dom_label"] > 0.0) & (label_df_filtered["label"] == -1)]["label"] = 1
    label_df_filtered = label_df_filtered[label_df["label"] >= 0]
    print(label_df_filtered["label"].value_counts())

    cell_df = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup.pkl"
    cell_df = pd.read_pickle(cell_df)
    cell_df = cell_df[cell_df["count_pm6a"] >= 20]
    cell_label_df = cell_df.merge(label_df_filtered, on="label_id", how="inner")
    cell_label_df["threshold"] = 1-((0.9 ** (1-cell_label_df["mean_pred"])) * (0.02 ** (cell_label_df["mean_pred"])))

    neg_df = cell_label_df[cell_label_df["dom_label"] == 0]
    pos_df = cell_label_df[cell_label_df["dom_label"] > 0]
    fp_id = neg_df[neg_df["pm6a"] > neg_df["threshold"]]["label_id"].values
    tp_id = pos_df[pos_df["pm6a"] > pos_df["threshold"]]["label_id"].values

    ## sample 10000
    fp_id = np.random.choice(fp_id, 10000, replace=False)
    tp_id = np.random.choice(tp_id, 10000, replace=False)

    del cell_df, cell_label_df, neg_df, pos_df
    gc.collect()

    paths = glob.glob("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/token_normalise_dwell_all_npz/*.npz")


    n_worker = 120

    path_split = np.array_split(paths, n_worker)

    proc_list = []
    for i in range(n_worker):
        proc = mp.Process(target=worker, args=(i, path_split[i], fp_id, tp_id))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    print("Complete")
    return None

if __name__ == "__main__":
    main()