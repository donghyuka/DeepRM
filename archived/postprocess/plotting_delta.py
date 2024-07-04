import os

default_n_threads = os.cpu_count() // 2
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.calibration import calibration_curve
import argparse
import glob
from utils.utils import printmessage
from utils.utils import max_f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, nargs="+", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--baseline", "-b", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/m6anet/output/data.site_proba.csv", help="Baseline path")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado_m6a/dorado_m6a_basecalled.pileup.bed", help="Dorado path")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)

    if not os.path.exists(args.baseline):
        print(f"Baseline path does not exist: {args.baseline}")
        args.baseline = None
    if not os.path.exists(args.dorado):
        print(f"Dorado path does not exist: {args.dorado}")
        args.dorado = None

    return args


def gmm_em_lda(pred_arr):
    ## Fit GMM with EM assuming bimodal distribution
    gmm = GaussianMixture(n_components=2)
    gmm.fit(pred_arr.reshape(-1,1))
    ## Find a discriminant using LDA
    threshold = (gmm.means_[0] + gmm.means_[1])/2
    printmessage(f"Threshold: {threshold}")
    return threshold


def process_baseline_inferece(data_path):
    if data_path is None:
        return None
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].str.split(".").str[0]
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"id": "label_id", "probability_modified": "pred_m6anet", "mod_ratio": "dom_m6anet"}, inplace = True)
    data_df = data_df[["label_id", "pred_m6anet", "dom_m6anet"]]
    return data_df


def process_dorado_inferece(data_path):
    if data_path is None:
        return None
    data_df = pd.read_csv(data_path, quoting = 3, sep = "\t", header = None, dtype=str)
    ## Keep column 0, 1, 4, 9
    data_df = data_df[[0, 1, 4, 9]]
    data_df.columns = ["nmid", "pos", "depth", "pred_dorado"]
    data_df["depth"] = data_df["depth"].astype(int)
    data_df = data_df[data_df["depth"] >= 10].copy()
    data_df["pred_dorado"] = data_df["pred_dorado"].str.split(" ").str[1].astype(float) / 100
    data_df["label_id"] = data_df["nmid"].str.split(".").str[0] + ":" + data_df["pos"]
    data_df = data_df[["label_id", "pred_dorado"]].copy()
    return data_df




def plot_scatter(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,4, figsize = (80,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    data_df = data_df[data_df["glori"] > 0]
    ax = axes[0]
    ax.scatter(data_df["glori"], data_df["original"], s = 10, alpha = 0.5, color = "royalblue")
    r2 = r2_score(data_df["glori"], data_df["original"])
    rho2 = spearmanr(data_df["glori"], data_df["original"])[0] ** 2
    ax.set_title(f"Transformer, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["glori"], data_df["original"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    ax = axes[1]
    ax.scatter(data_df["glori"], data_df["pred_adj"], s = 10, alpha = 0.5, color = "darkorchid")
    r2 = r2_score(data_df["glori"], data_df["pred_adj"])
    rho2 = spearmanr(data_df["glori"], data_df["pred_adj"])[0] ** 2
    ax.set_title(f"Transformer+ResNet, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["glori"], data_df["pred_adj"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    if "dom_m6anet" in data_df.columns:
        ax = axes[2]
        ax.scatter(data_df["glori"], data_df["dom_m6anet"], s = 10, alpha = 0.5, color = "tomato")
        r2b = r2_score(data_df["glori"], data_df["dom_m6anet"])
        rho2b = spearmanr(data_df["glori"], data_df["dom_m6anet"])[0] ** 2
        ax.set_title(f"m6Anet, R2 = {r2b:.3f}, rho2 = {rho2b:.3f}")
        z = np.polyfit(data_df["glori"], data_df["dom_m6anet"], 1)
        p = np.poly1d(z)
        x = np.linspace(0, 1, 100)
        ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    if "pred_dorado" in data_df.columns:
        ax = axes[3]
        ax.scatter(data_df["glori"], data_df["pred_dorado"], s = 10, alpha = 0.5, color = "forestgreen")
        r2c = r2_score(data_df["glori"], data_df["pred_dorado"])
        rho2c = spearmanr(data_df["glori"], data_df["pred_dorado"])[0] ** 2
        ax.set_title(f"Dorado, R2 = {r2c:.3f}, rho2 = {rho2c:.3f}")
        z = np.polyfit(data_df["glori"], data_df["pred_dorado"], 1)
        p = np.poly1d(z)
        x = np.linspace(0, 1, 100)
        ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)

    for ax in axes:
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('GLORI DoM')
        ax.set_ylabel('Predicted DoM')
    fig.suptitle(f"{modelname}")
    plt.savefig(f"{outdir}/scatter.png")
    plt.close()
    return r2


def process_inferece(data_df, baseline_pred, dorado_pred):
    data_df["pred_adj"] = data_df["original"] - data_df["pred"]
    data_df = data_df.merge(baseline_pred, how = "left", on = "label_id")
    data_df = data_df.merge(dorado_pred, how = "left", on = "label_id")
    data_df.fillna(0, inplace = True)
    return data_df



def main():
    args = parse_args()
    dorado_pred = process_dorado_inferece(args.dorado)
    baseline_pred = process_baseline_inferece(args.baseline)
    for idx, data_path in enumerate(args.input):
        data_paths = glob.glob(f"{data_path}/*.tsv")
        data_df_original = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
        data_df_original.fillna(0, inplace = True)
        modelname = data_path
        if modelname.endswith("/"):
            modelname = modelname[:-1]
        modelname = modelname.split("/")[-1]
        outdir = os.path.join(args.output, modelname)
        os.makedirs(outdir, exist_ok = True)
        data_df = process_inferece(data_df_original.copy(), baseline_pred, dorado_pred)
        plot_scatter(data_df, outdir, modelname)
    return None


if __name__ == "__main__":
    main()



