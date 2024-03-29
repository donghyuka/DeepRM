from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, r2_score
import argparse
import glob
import os
from utils.utils import printmessage
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, nargs="+", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--ratio", "-r", type=int, default=10, help="Neg/Pos ratio")
    args = parser.parse_args()
    return args


def process_inferece(data_path, outdir, modelname):
    data_paths = glob.glob(f"{data_path}/*.tsv")
    data_df = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
    data_df.fillna(0, inplace = True)
    plot_histogram_read(data_df, outdir, modelname)
    # threshold = gmm_em_lda(data_df["pred"].values)
    threshold = 0.2 ## pre-calculated threshold using GMM-EM-LDA (It is too slow to calculate every time)
    epsilon = 1e-15
    data_df["dom"] = data_df["pred"].apply(lambda x: 1 if x >=threshold else 0)
    ## Groupby label_id and get mean of predictions
    data_df["count"] = 1
    data_df.rename(columns = {"pred": "pred_ari"}, inplace = True)
    data_df["pred_geo"] = np.clip(data_df["pred_ari"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred_geo"] = (np.log10(1 - data_df["pred_geo"].to_numpy()))
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
    ax.plot(fpr, tpr, lw=3, label=f'Transformer (AUC = {roc_auc:.2f})', color = "royalblue")
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
    ax.plot(recall, precision, lw=3, label=f'Transformer (AUC = {pr_auc:.2f})', color = "royalblue")
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
    fig, ax = plt.subplots(1,1, figsize = (20,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    data_df = data_df[data_df["label"] == 1]

    ax.scatter(data_df["dom_label"], data_df["dom"], s = 10, alpha = 0.5, color = "royalblue")
    r2 = r2_score(data_df["dom_label"], data_df["dom"])
    ax.set_title(f"Transformer, R2 = {r2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["dom"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('GLORI DoM')
    ax.set_ylabel('Predicted DoM')
    fig.suptitle(f"{modelname}")
    plt.savefig(f"{outdir}/scatter.png")
    return None


def plot_histogram_read(data_df, outdir, modelname):
    ## Plot histogram of predictions
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.hist(data_df["pred"], bins = 100, color = "royalblue", label = "Transformer")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title(f"{modelname}")
    ax.legend()
    plt.savefig(f"{outdir}/hist_read.png")
    return None


def plot_histogram_site(data_df, outdir, modelname):
    ## Plot histogram of predictions
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.hist(data_df[data_df["label"] == 1]["pred_geo"], bins = 100, color = "royalblue", label = "Neg")
    ax.hist(data_df[data_df["label"] == 0]["pred_geo"], bins = 100, color = "tomato", label = "Pos", alpha = 0.5)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title(f"{modelname}")
    ax.legend()
    plt.savefig(f"{outdir}/hist_site.png")
    return None


def main():
    args = parse_args()
    pr_baseline_level = 1/(1+args.ratio)
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
        data_df = process_inferece(data_path, outdir, modelname)
        data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
        roc_auc = plot_roc(data_df, outdir, modelname)
        pr_auc = plot_pr(data_df, outdir, pr_baseline_level, modelname)
        plot_histogram_site(data_df, outdir, modelname)
        print(f"{modelname}\t{roc_auc:.3f}\t{pr_auc:.3f}")
        result_dict["model"].append(modelname)
        result_dict["roc_auc"].append(roc_auc)
        result_dict["pr_auc"].append(pr_auc)
    print("==========================")
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"{args.output}/result.tsv", sep = "\t", index = False)
    return None


if __name__ == "__main__":
    main()



