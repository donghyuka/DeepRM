import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.calibration import calibration_curve
import argparse
import glob

import seaborn as sns
from tqdm import tqdm

from utils.utils import printmessage
from utils.utils import max_f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/plot", help="Output path")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v5_genome_drach-updatedlabel-dorado/BaeklabV3.GP3.depth5_20.twm6astrict.notsampled.drach_nofilter_depth5_20/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v5_genome_drach.tsv", help="Dorado path")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)

    if not os.path.exists(args.dorado):
        print(f"Dorado path does not exist: {args.dorado}")
        args.dorado = None

    return args


def plot_scatter(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,3, figsize = (60,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    # data_df = data_df[data_df["dom_label"] > 0]
    ax = axes[0]
    ax.scatter(data_df["dom_label"], data_df["dom"], s = 10, alpha = 0.5, color = "royalblue")
    r2 = r2_score(data_df["dom_label"], data_df["dom"])
    rho2 = spearmanr(data_df["dom_label"], data_df["dom"])[0] ** 2
    ax.set_title(f"Transformer, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["dom"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    ax = axes[1]
    ax.scatter(data_df["dom_label"], data_df["dom_adj"], s = 10, alpha = 0.5, color = "royalblue")
    r2_adj = r2_score(data_df["dom_label"], data_df["dom_adj"])
    rho2_adj = spearmanr(data_df["dom_label"], data_df["dom_adj"])[0] ** 2
    ax.set_title(f"Transformer+ResNet, R2 = {r2_adj:.3f}, rho2 = {rho2_adj:.3f}")
    ## polyfit
    # z = np.polyfit(data_df["dom_label"], data_df["dom_adj"], 1)
    # p = np.poly1d(z)
    # x = np.linspace(0, 1, 100)
    # ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    if "pred_dorado" in data_df.columns:
        ax = axes[2]
        ax.scatter(data_df["dom_label"], data_df["pred_dorado"], s = 10, alpha = 0.5, color = "forestgreen")
        r2c = r2_score(data_df["dom_label"], data_df["pred_dorado"])
        rho2c = spearmanr(data_df["dom_label"], data_df["pred_dorado"])[0] ** 2
        ax.set_title(f"Dorado, R2 = {r2c:.3f}, rho2 = {rho2c:.3f}")
        # z = np.polyfit(data_df["dom_label"], data_df["pred_dorado"], 1)
        # p = np.poly1d(z)
        # x = np.linspace(0, 1, 100)
        # ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)

    for ax in axes:
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"{modelname}")
    plt.savefig(f"{outdir}-scatter.png")
    plt.close()
    return r2_adj, rho2_adj


def process_inferece(data_df, dorado_pred):
    data_df.rename(columns = {"pred": "dom_adj"}, inplace = True)
    data_df = dorado_pred.merge(data_df, how = "inner", on = "label_id")
    data_df.fillna(0, inplace = True)

    # data_df["pred_mask"] = (data_df["dom_adj"] >= 0.01)
    # data_df["dom_adj"] = data_df["dom_adj"] * data_df["pred_mask"]

    # data_df["pred_mask"] = (data_df["pred_geo"] >= 0.50)
    # data_df["dom_adj"] = data_df["dom_adj"] * data_df["pred_mask"]

    # data_df = data_df[data_df["pred_mask"]]
    #
    # data_df = data_df[data_df["dom_label"] > 0]
    # data_df = data_df[data_df["label"] > 0]

    data_df = data_df[data_df["count"]>10]

    return data_df


def plot_boxplot(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,3, figsize = (60,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    # data_df = data_df[data_df["dom_label"] > 0]

    data_df["dom_label_bin"] = data_df["dom_label"].apply(lambda x: x*100 // 10 / 10)

    ## replace 1.0 with 0.9
    data_df["dom_label_bin"] = data_df["dom_label_bin"].apply(lambda x: 0.9 if x == 1.0 else x)

    ax = axes[0]
    sns.boxplot(x = "dom_label_bin", y = "dom", data = data_df, ax = ax)
    ax.set_title(f"Transformer")
    ax.set_xlabel("GLORI DoM")
    ax.set_ylabel("Predicted DoM")
    ax = axes[1]
    sns.boxplot(x = "dom_label_bin", y = "dom_adj", data = data_df, ax = ax)
    ax.set_title(f"Transformer+ResNet")
    ax.set_xlabel("GLORI DoM")
    ax.set_ylabel("Predicted DoM")
    if "pred_dorado" in data_df.columns:
        ax = axes[2]
        sns.boxplot(x = "dom_label_bin", y = "pred_dorado", data = data_df, ax = ax)
        ax.set_title(f"Dorado")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes:
        ax.set_ylim([0.0, 1.0])
    fig.suptitle(f"{modelname}")
    plt.savefig(f"{outdir}-boxplot.png")
    plt.close()
    return None


def main():
    args = parse_args()
    dorado_pred = pd.read_csv(args.dorado, sep = "\t")
    print(dorado_pred)
    data_dict = {"model": [], "r2": [], "rho2": []}
    # for data_path in tqdm(glob.glob(f"{args.input}/*/")):
    if True:
        data_path = args.input
        data_paths = glob.glob(f"{data_path}/*.tsv")
        data_df_original = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
        data_df_original.fillna(0, inplace = True)
        modelname = data_path
        if modelname.endswith("/"):
            modelname = modelname[:-1]
        modelname = modelname.split("/")[-1]
        outdir = os.path.join(args.output, modelname)
        data_df = process_inferece(data_df_original.copy(), dorado_pred)
        r2, rho2 = plot_scatter(data_df, outdir, modelname)
        plot_boxplot(data_df, outdir, modelname)
        data_dict["model"].append(modelname)
        data_dict["r2"].append(r2)
        data_dict["rho2"].append(rho2)
    data_df = pd.DataFrame(data_dict)
    data_df["model"] = data_df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    data_df.to_csv(f"/extdata4/baeklab/Hyeonseo/m6A/postprocess/checkpoint_summary/evaluation.tsv", sep = "\t", index = False)
    print(data_df)
    summary_df = pd.read_csv(f"/extdata4/baeklab/Hyeonseo/m6A/postprocess/checkpoint_summary/Postprocess-ResNet-20240515-120946.tsv", sep = "\t")
    summary_df["model"] = summary_df["path"].apply(lambda x: x.split("/")[-1][:-3])
    summary_df = summary_df.merge(data_df, how = "inner", on = "model")
    print(summary_df)
    summary_df.to_csv(f"/extdata4/baeklab/Hyeonseo/m6A/postprocess/checkpoint_summary/Postprocess-ResNet-20240515-120946-R2.tsv", sep = "\t", index = False)

    return None


if __name__ == "__main__":
    main()



