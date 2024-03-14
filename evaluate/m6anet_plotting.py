from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, r2_score
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/m6anet/output/data.site_proba.csv", help="Data path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/rna004_label_miclip2_m6ace_glori_sampled.tsv", help="Label path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    args = parser.parse_args()
    return args


def process_inferece(data_path,label_path, epsilon = 1e-6):
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].str.split(".").str[0]
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"probability_modified": "pred"}, inplace = True)
    label_df = pd.read_csv(label_path, sep = "\t")
    data_df = data_df.merge(label_df, on = "id", how = "right")
    data_df.fillna(epsilon, inplace = True)
    data_df = data_df.groupby("id").agg({"label": "first", "pred": "mean"}).reset_index()
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
    return None


def plot_pr(data_df, outdir):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred"])
    ax.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AUC = {auc(recall, precision):.2f})', color = "royalblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr.png")
    return None


def main():
    args = parse_args()
    outdir = os.path.join(args.output, "m6anet2_m6ace_miclip2_m6ace_glori")
    os.makedirs(outdir, exist_ok = True)
    data_df = process_inferece(args.input, args.label)
    data_df.to_csv(f"{outdir}/m6anet.tsv", sep = "\t", index = False)
    plot_roc(data_df, outdir)
    plot_pr(data_df, outdir)
    return None


if __name__ == "__main__":
    main()



