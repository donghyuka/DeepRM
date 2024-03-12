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
    parser.add_argument("--input", "-i", type=str, required=True, help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    args = parser.parse_args()
    return args


def process_inferece(data_path, threshold = 0.5):
    data_paths = glob.glob(f"{data_path}/*.tsv")
    data_df = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
    data_df["pred_bin"] = data_df["pred"] > threshold
    ## Groupby label_id and get mean of predictions
    data_df = data_df.groupby("label_id").agg({"label": "first", "pred": "mean", "pred_bin": "mean"}).reset_index()
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
    modelname = args.input
    if modelname.endswith("/"):
        modelname = modelname[:-1]
    modelname = modelname.split("/")[-1]
    outdir = os.path.join(args.output, modelname)
    os.makedirs(outdir, exist_ok = True)
    data_df = process_inferece(args.input)
    data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
    plot_roc(data_df, outdir)
    plot_pr(data_df, outdir)
    return None


if __name__ == "__main__":
    main()



