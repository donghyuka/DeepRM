from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, r2_score
import argparse
import glob
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/m6anet/output/data.site_proba.csv", help="Data path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/miclip2_glori_m6ace.v2.sampled.tsv", help="Label path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    args = parser.parse_args()
    return args


def process_inferece(data_path,label_path):
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].str.split(".").str[0]
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"probability_modified": "pred", "mod_ratio": "dom"}, inplace = True)
    label_df = pd.read_csv(label_path, sep = "\t")
    label_df.rename(columns = {"m6A_level": "dom_label"}, inplace = True)
    data_df = data_df.merge(label_df, on = "id", how = "inner")
    data_df = data_df[["id", "label", "pred", "dom", "dom_label"]]
    return data_df


def plot_roc(data_df, outdir):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})', color = "royalblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/roc.png")
    with open(f"{outdir}/roc.pkl", "wb") as f:
        pickle.dump({"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}, f)
    return None


def plot_pr(data_df, outdir):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred"])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})', color = "royalblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr.png")
    with open(f"{outdir}/pr.pkl", "wb") as f:
        pickle.dump({"precision": precision, "recall": recall, "pr_auc": pr_auc}, f)
    return None


def main():
    args = parse_args()
    outdir = os.path.join(args.output, "m6anet_miclip2_glori_m6ace")
    os.makedirs(outdir, exist_ok = True)
    data_df = process_inferece(args.input, args.label)
    data_df.to_csv(f"{outdir}/m6anet.tsv", sep = "\t", index = False)
    plot_roc(data_df, outdir)
    plot_pr(data_df, outdir)
    return None


if __name__ == "__main__":
    main()



