import os
import time

os.environ["OMP_NUM_THREADS"] = "64" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "64" # export OPENBLAS_NUM_THREADS=4


import gc
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import glob, os
import torch
import math

from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns



def reduce_dim(data_df, column):

    ## Normalize and scale
    # signal_data_arr = np.array(data_df["signal_embedding"].tolist())
    kmer_data_arr = np.array(data_df[column].tolist())

    scaler = StandardScaler()
    # signal_data_arr = scaler.fit_transform(signal_data_arr)
    kmer_data_arr = scaler.fit_transform(kmer_data_arr)
    # data_arr = np.concatenate([signal_data_arr, kmer_data_arr], axis=0)
    data_arr = kmer_data_arr
    print(np.shape(data_arr))

    ## Run PCA to have std of 1e-4
    pca = PCA(n_components=64, svd_solver="auto")
    pca_arr = pca.fit_transform(data_arr)
    pca_arr = pca_arr / np.std(pca_arr[:,0]) * 1e-4
    print("PCA done")

    ## Run TSNE
    tsne = TSNE(n_components=2, n_jobs=60,
                learning_rate="auto", perplexity=500, n_iter=4000, n_iter_without_progress=200, verbose=1,
                init="pca")

    transformed = tsne.fit_transform(pca_arr)
    # transformed_signal = transformed[:len(data_df)]
    # transformed_kmer = transformed[len(data_df):]
    #
    # data_df["signal_axis1"] = transformed_signal[:,0]
    # data_df["signal_axis2"] = transformed_signal[:,1]
    # data_df["kmer_axis1"] = transformed_kmer[:,0]
    # data_df["kmer_axis2"] = transformed_kmer[:,1]

    data_df["axis1"] = transformed[:,0]
    data_df["axis2"] = transformed[:,1]


    return data_df


def plot_reduced(reduced_df, figpath):
    keyfunc = lambda x: x.str[2]+x.str[1]+x.str[3]+x.str[1]+x.str[4]
    reduced_df = reduced_df.sort_values("motif", key = lambda x: keyfunc(x))
    # reduced_df["label_pos"] = reduced_df["label"].astype(str)+":"+reduced_df["move"].astype(str)
    # reduced_df = reduced_df[["signal_axis1", "signal_axis2", "kmer_axis1", "kmer_axis2", "label_pos", "motif", "label"]]
    reduced_df["cent1"] = reduced_df["motif"].str[2]
    reduced_df["cent3"] = reduced_df["motif"].str[1:4]
    reduced_df = reduced_df[reduced_df["cent3"] != "UAA"]
    reduced_df = reduced_df[reduced_df["cent3"] != "UAG"]
    reduced_df = reduced_df[reduced_df["cent3"] != "CAG"]


    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(figsize=(60, 40), ncols=3, nrows=2)

    for axes_row, label in zip(axes, [0,1]):
        data_df = reduced_df[reduced_df["label"]==label]
        label_str = "cA" if label == 0 else "m6A"
        sns.scatterplot(data=data_df, x="axis1", y="axis2", hue="bq", ax=axes_row[0],
                        alpha=1.0, s=30, palette="viridis")
        sns.move_legend(
            axes_row[0], "lower center", ncols = 4,
            bbox_to_anchor=(.5, 1), title=None, frameon=False,
        )
        axes_row[0].set_title(f"{label_str} - BQ")

        sns.scatterplot(data=data_df, x="axis1", y="axis2", hue="cent3", ax=axes_row[1],
                        alpha=0.5, s=30, palette="gist_rainbow")
        sns.move_legend(
            axes_row[1], "lower center", ncols = 4,
            bbox_to_anchor=(.5, 1), title=None, frameon=False,
        )
        axes_row[1].set_title(f"{label_str} - Motif")

        sns.scatterplot(data=data_df, x="axis1", y="axis2", hue="position", ax=axes_row[2],
                        alpha=1.0, s=30, palette="viridis")
        sns.move_legend(
            axes_row[2], "lower center", ncols = 4,
            bbox_to_anchor=(.5, 1), title=None, frameon=False,
        )
        axes_row[2].set_title(f"{label_str} - Position")


        # for ax in axes_row:
        #     ax.set_xlim(-150, 150)
        #     ax.set_ylim(-100, 100)

    plt.savefig(figpath)
    return None


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Plotting")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--figdir", type=str, default=None)
    args = parser.parse_args()
    if args.figdir is None:
        args.figdir = args.data_path + "/plot"
        os.makedirs(args.figdir, exist_ok=True)
    return args


def process_data(data):
    ## explode the data
    data = data[["kmer", "bq", "move", "kmer_embedding", "signal_embedding", "label", "label_id"]].copy()
    array_of_list = np.array([list(range(200))]* len(data))
    array_of_list = array_of_list.tolist()
    data["position"] = array_of_list
    data = data.explode(["kmer", "bq", "move", "kmer_embedding", "signal_embedding", "position"])
    data = data[data["move"] > 0]
    return data


def int_to_motif(kmer_int,kmer_size = 5):
    kmer_int_quarternary = np.base_repr(kmer_int-1, base=4)
    ## zerofill to 5 digits
    kmer_int_quarternary = "0"*(kmer_size - len(kmer_int_quarternary)) + kmer_int_quarternary
    ## 0->A, 1->C, 2->G, 3->U
    convert_dict = {"0":"A", "1":"C", "2":"G", "3":"U"}
    kmer = "".join([convert_dict[i] for i in kmer_int_quarternary])
    ## flip
    kmer = kmer[::-1]
    return kmer


def add_position(data, seq_len=200, d_model = 512):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.numpy()
    data["pe"] = data.position.apply(lambda x: pe[x])
    data["sig_pe"] = data.apply(lambda x: x["signal_embedding"] + x["pe"], axis=1)
    return data



def main():
    args = parse_args()
    # data = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{args.data_path}/inference/*.pkl")])
    # data = process_data(data)
    # data = data[["bq", "label_id", "move", "label"]]
    # data = data[data["move"]==9]
    # data["label_id_move"] = data["label_id"].astype(str) + ":" + data["move"].astype(str)
    # data_bq = data[["bq", "label_id_move", "label"]].copy()
    # data_bq.to_pickle(args.data_path + "/bq.pkl")
    # print(data)
    # data.to_pickle(args.data_path + "/combined.pkl")
    # data_label = pd.read_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GP3.depth20.twm6astrict.sampled.drach.balanced.tsv", sep="\t")
    # # data = pd.read_pickle(args.data_path + "/combined.pkl")
    # data_label = data_label[["id", "m6A_level"]]
    # data_label.rename(columns={"id": "label_id"}, inplace=True)
    # data = pd.merge(data, data_label, how="left", on="label_id")
    # data = data[data["move"]==9]
    # data.to_pickle(args.data_path + "/combined_centre.pkl")
    # data = pd.read_pickle(args.data_path + "/combined_centre.pkl")
    # data_ca = data[data["label"]==0].sample(n = 10000)
    # data_m6a = data[data["m6A_level"]>=0.9].sample(n = 10000)
    # data = pd.concat([data_ca, data_m6a])
    # data["motif"] = data["kmer"].apply(int_to_motif)
    # # data.to_pickle(args.data_path + "/sampled_centre.pkl")
    # # data = pd.read_pickle(args.data_path + "/sampled_centre.pkl")
    # # data = pd.read_pickle(args.data_path + "/sampled.pkl")
    # # print(data)
    # data = add_position(data)
    # print(data)
    # data.to_pickle(args.data_path + "/sampled_centre_positioned.pkl")
    # data = pd.read_pickle(args.data_path + "/sampled_centre_positioned.pkl")
    # #
    # # os.makedirs(args.figdir, exist_ok=True)
    # # # ## remove duplicates, by motif
    # # data = data.drop_duplicates(subset=["bq"], keep="first")
    # # data = pd.read_pickle(args.data_path + "/sampled_centre.pkl")
    # data["sig_kmer"] = data.apply(lambda x: x["signal_embedding"] + x["kmer_embedding"], axis=1)
    #
    # data_red = reduce_dim(data, "signal_embedding")
    # data_red.to_pickle(args.data_path + "/reduced_sig_only.pkl")
    # data_red = reduce_dim(data, "sig_kmer")
    # data_red.to_pickle(args.data_path + "/reduced_sig_kmer.pkl")
    # print(data)
    # data_red = reduce_dim(data, "sig_pe")
    # data_red.to_pickle(args.data_path + "/reduced_sig_pe.pkl")
    # data["sig_kmer_pe"] = data.apply(lambda x: x["sig_pe"] + x["kmer_embedding"], axis=1)
    # data_red = reduce_dim(data, "sig_kmer_pe")
    # data_red.to_pickle(args.data_path + "/reduced_sig_kmer_pe.pkl")

    # #
    # # # # ## sort by motif
    # data.to_pickle(args.data_path + "/reduced_sig_bq.pkl")

    # data_bq = pd.read_pickle(args.data_path + "/bq.pkl")
    # data_bq = data_bq[["bq", "label_id_move"]].copy()
    # data_bq = data_bq.drop_duplicates(subset=["label_id_move"], keep="first")
    # print(data_bq)
    # data_bq.to_pickle(args.data_path + "/bq.pkl")

    data_bq = pd.read_pickle(args.data_path + "/bq.pkl")

    data = pd.read_pickle(args.data_path + "/reduced_sig_pe.pkl")
    data["label_id_move"] = data["label_id"].astype(str) + ":" + data["move"].astype(str)
    data = data.merge(data_bq, how="left", on="label_id_move")
    data = data.copy()
    data.to_pickle(args.data_path + "/reduced_sig_pe.pkl")
    print(data)
    plot_reduced(data, f"{args.figdir}/reduced_sig_pe.png")

    data = pd.read_pickle(args.data_path + "/reduced_sig_kmer.pkl")
    data["label_id_move"] = data["label_id"].astype(str) + ":" + data["move"].astype(str)
    data = data.merge(data_bq, how="left", on="label_id_move")
    data.to_pickle(args.data_path + "/reduced_sig_kmer.pkl")
    print(data)
    plot_reduced(data, f"{args.figdir}/reduced_sig_kmer.png")

    data = pd.read_pickle(args.data_path + "/reduced_sig_only.pkl")
    data["label_id_move"] = data["label_id"].astype(str) + ":" + data["move"].astype(str)
    data = data.merge(data_bq, how="left", on="label_id_move")
    data.to_pickle(args.data_path + "/reduced_sig_only.pkl")
    print(data)
    plot_reduced(data, f"{args.figdir}/reduced_sig_only.png")

    data = pd.read_pickle(args.data_path + "/reduced_sig_kmer_pe.pkl")
    data["label_id_move"] = data["label_id"].astype(str) + ":" + data["move"].astype(str)
    data = data.merge(data_bq, how="left", on="label_id_move")
    data.to_pickle(args.data_path + "/reduced_sig_kmer_pe.pkl")
    print(data)
    plot_reduced(data, f"{args.figdir}/reduced_sig_kmer_pe.png")
    return None




if __name__ == "__main__":
    main()




