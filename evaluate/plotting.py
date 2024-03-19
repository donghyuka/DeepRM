import os

default_n_threads = os.cpu_count() // 2
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, r2_score
from sklearn.mixture import GaussianMixture
import argparse
import glob
from utils.utils import printmessage
import pickle
from utils.utils import max_f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, nargs="+", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--baseline", "-b", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot/m6anet_miclip2_glori_m6ace_drach", help="Baseline path")
    parser.add_argument("--ratio", "-r", type=int, default=10, help="Neg/Pos ratio")
    args = parser.parse_args()
    return args


def gmm_em_lda(pred_arr):
    ## Fit GMM with EM assuming bimodal distribution
    gmm = GaussianMixture(n_components=2)
    gmm.fit(pred_arr.reshape(-1,1))
    ## Find a discriminant using LDA
    threshold = (gmm.means_[0] + gmm.means_[1])/2
    printmessage(f"Threshold: {threshold}")
    return threshold


def process_inferece(data_path):
    data_paths = glob.glob(f"{data_path}/*.tsv")
    data_df = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
    data_df.fillna(0, inplace = True)
    # threshold = gmm_em_lda(data_df["pred"].values)
    threshold = 0.20 ## pre-calculated threshold using GMM-EM-LDA (It is too slow to calculate every time)
    epsilon = 1e-6
    data_df["dom"] = data_df["pred"].apply(lambda x: 1 if x >=threshold else 0)
    ## Groupby label_id and get mean of predictions
    data_df["count"] = 1
    data_df.rename(columns = {"pred": "pred_ari"}, inplace = True)
    data_df["pred_geo"] = np.clip(np.log10(1 + epsilon - data_df["pred_ari"].to_numpy()), None, 0)
    data_df = data_df.groupby("label_id").agg({"label": "first", "pred_ari": "mean", "pred_geo": "mean",
                                               "dom": "mean", "count": "sum"}).reset_index()
    data_df["pred_geo"] = np.clip(data_df["pred_geo"], None, 0)
    data_df["pred_geo"] = (1 - 10**data_df["pred_geo"].to_numpy())
    return data_df


def plot_roc(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_geo"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Transformer (AUC = {roc_auc:.3f})', color = "royalblue")
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_m6anet"])
    roc_auc_m6anet = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3,label=f'm6Anet (AUC = {roc_auc_m6anet:.3f})', color = "tomato")
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{modelname}\nReceiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/roc.png")
    return roc_auc


def plot_pr(data_df, outdir, pr_baseline_level, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_geo"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_geo"])
    ax.plot(recall, precision, lw=3, label=f'Transformer (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "royalblue")
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_m6anet"])
    pr_auc_m6anet = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_m6anet"])
    ax.plot(recall, precision, lw=3,label=f'm6Anet (AUC = {pr_auc_m6anet:.3f}, Max F-1 = {max_f1:.3f})', color = "tomato")
    ax.plot([0, 1], [pr_baseline_level, pr_baseline_level], color='grey', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{modelname}\nPrecision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr.png")
    return pr_auc


def plot_scatter(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,2, figsize = (40,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    data_df = data_df[data_df["label"] == 1]
    ax = axes[0]
    ax.scatter(data_df["dom_label"], data_df["dom"], s = 10, alpha = 0.5, color = "royalblue")
    r2 = r2_score(data_df["dom_label"], data_df["dom"])
    ax.set_title(f"Transformer, R2 = {r2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["dom"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    ax = axes[1]
    ax.scatter(data_df["dom_label"], data_df["dom_m6anet"], s = 10, alpha = 0.5, color = "tomato")
    r2 = r2_score(data_df["dom_label"], data_df["dom_m6anet"])
    ax.set_title(f"m6Anet, R2 = {r2:.3f}")
    z = np.polyfit(data_df["dom_label"], data_df["dom_m6anet"], 1)
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
    return None



def main():
    args = parse_args()
    pr_baseline_level = 1/(1+args.ratio)
    baseline_pred = pd.read_csv(f"{args.baseline}/m6anet.tsv", sep = "\t")
    baseline_pred.rename(columns = {"id": "label_id", "pred": "pred_m6anet", 'dom': 'dom_m6anet'}, inplace = True)
    baseline_pred.drop(["label"], axis = 1, inplace = True)
    result_dict = {"model": [], "roc_auc": [], "pr_auc": []}
    print("==========================")
    print("Model\tROC_AUC\tPR_AUC")
    for idx, data_path in enumerate(args.input):
        modelname = data_path
        if modelname.endswith("/"):
            modelname = modelname[:-1]
        modelname = modelname.split("/")[-1]
        outdir = os.path.join(args.output, modelname)
        os.makedirs(outdir, exist_ok = True)
        data_df = process_inferece(data_path)
        data_df = data_df.merge(baseline_pred, on = "label_id", how = "inner")
        data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
        roc_auc = plot_roc(data_df, outdir, modelname)
        pr_auc = plot_pr(data_df, outdir, pr_baseline_level, modelname)
        print(f"{modelname}\t{roc_auc:.3f}\t{pr_auc:.3f}")
        result_dict["model"].append(modelname)
        result_dict["roc_auc"].append(roc_auc)
        result_dict["pr_auc"].append(pr_auc)
        plot_scatter(data_df, outdir, modelname)
    print("==========================")
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"{args.output}/result.tsv", sep = "\t", index = False)
    return None


if __name__ == "__main__":
    main()



