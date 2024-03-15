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


def process_inferece(data_path):
    data_paths = glob.glob(f"{data_path}/*.tsv")
    data_df = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
    ## TODO: Implement decision boundary auto calculation. For now, use 0.5.
    data_df["dom"] = data_df["pred"].apply(lambda x: 1 if x > 0.5 else 0)
    ## Groupby label_id and get mean of predictions
    data_df = data_df.groupby("label_id").agg({"label": "first", "pred": "mean", "dom": "mean"}).reset_index()
    return data_df


def plot_roc(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred"])
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
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred"])
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
        data_df = process_inferece(data_path)
        data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
        roc_auc = plot_roc(data_df, outdir, modelname)
        pr_auc = plot_pr(data_df, outdir, pr_baseline_level, modelname)
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



