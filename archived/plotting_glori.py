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
import argparse
import glob
from utils.utils import printmessage
from utils.utils import max_f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, nargs="+", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--baseline", "-b", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/m6anet/output/data.site_proba.csv", help="Baseline path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GLORI.depth20.notsampled.drach.tsv", help="Label path")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)
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
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].str.split(".").str[0]
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"id": "label_id", "probability_modified": "pred_m6anet", "mod_ratio": "dom_m6anet"}, inplace = True)
    data_df = data_df[["label_id", "pred_m6anet", "dom_m6anet"]]
    return data_df



def process_inferece(data_df, updated_label_path, outdir, modelname, baseline_pred):

    plot_histogram_read(data_df, outdir, modelname)
    # threshold = gmm_em_lda(data_df["pred"].values)
    threshold = 0.10 ## pre-calculated threshold using GMM-EM-LDA (It is too slow to calculate every time)
    epsilon = 1e-12
    data_df = data_df[(data_df["pred"] < threshold) | ( data_df["pred"] > 1-threshold)].copy()
    data_df["dom"] = data_df["pred"].apply(lambda x: 1 if x >=threshold else 0)
    ## Groupby label_id and get mean of predictions
    data_df["count"] = 1
    data_df.rename(columns = {"pred": "pred_ari"}, inplace = True)
    data_df["pred_geo"] = np.clip(data_df["pred_ari"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred_geo"] = np.log10(1 - data_df["pred_geo"].to_numpy())
    ## groupby label_id and remove max
    data_df = data_df.groupby("label_id").agg({"pred_ari": "mean", "pred_geo": "mean",
                                               "dom": "mean", "count": "sum"}).reset_index()
    data_df["pred_geo"] = np.clip(data_df["pred_geo"], None, 0)
    data_df["pred_geo"] = (1 - 10**data_df["pred_geo"].to_numpy())

    data_df = data_df.merge(baseline_pred, on = "label_id", how = "inner")
    data_df.fillna(0, inplace = True)

    updated_label_df = pd.read_csv(updated_label_path, sep = "\t")
    updated_label_df = updated_label_df[["id", "label", "m6A_level"]]
    updated_label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    data_df = data_df.merge(updated_label_df, on = "label_id", how = "inner")

    return data_df


def plot_roc(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_geo"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Transformer (AUC = {roc_auc:.3f})', color = "royalblue", zorder = 2)
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_m6anet"])
    roc_auc_m6anet = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3,label=f'm6Anet (AUC = {roc_auc_m6anet:.3f})', color = "tomato", zorder = 1)
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{modelname}\nReceiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/roc.png")
    plt.close()
    return roc_auc


def plot_pr(data_df, outdir, modelname):
    pr_baseline_level = data_df["label"].sum() / data_df["label"].count()
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_geo"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_geo"])
    ax.plot(recall, precision, lw=3, label=f'Transformer (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "royalblue", zorder = 2)
    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_m6anet"])
    pr_auc_m6anet = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_m6anet"])
    ax.plot(recall, precision, lw=3,label=f'm6Anet (AUC = {pr_auc_m6anet:.3f}, Max F-1 = {max_f1:.3f})', color = "tomato", zorder = 1)
    ax.plot([0, 1], [pr_baseline_level, pr_baseline_level], color='grey', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{modelname}\nPrecision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr.png")
    plt.close()
    return pr_auc


def plot_scatter(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,2, figsize = (40,20))
    ## ax0: dom_label vs. dom
    ## ax1: dom_label vs. dom_m6anet
    data_df = data_df[data_df["dom_label"] > 0]
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
    ax.scatter(data_df["dom_label"], data_df["dom_m6anet"], s = 10, alpha = 0.5, color = "tomato")
    r2b = r2_score(data_df["dom_label"], data_df["dom_m6anet"])
    rho2b = spearmanr(data_df["dom_label"], data_df["dom_m6anet"])[0] ** 2
    ax.set_title(f"m6Anet, R2 = {r2b:.3f}, rho2 = {rho2b:.3f}")
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
    plt.close()
    return r2


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
    plt.close()
    return None


def plot_histogram_site(data_df, outdir, modelname):
    ## Plot histogram of predictions
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.hist(data_df["pred_geo"], bins = 100, color = "royalblue", label = "Transformer")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title(f"{modelname}")
    ax.legend()
    plt.savefig(f"{outdir}/hist_site.png")
    plt.close()
    return None


def main():
    args = parse_args()
    baseline_pred = process_baseline_inferece(args.baseline)
    result_dict = {"model": [], "label": [],"roc_auc": [], "pr_auc": [], "r2": []}
    print("==========================")
    print("Model\tLabel\tROC_AUC\tPR_AUC\tR2")
    for idx, data_path in enumerate(args.input):
        data_paths = glob.glob(f"{data_path}/*.tsv")
        data_df_original = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
        data_df_original.fillna(0, inplace = True)
        data_df_original.drop(["label"], axis = 1, inplace = True)
        modelname = data_path
        if modelname.endswith("/"):
            modelname = modelname[:-1]
        modelname = modelname.split("/")[-1]

        if os.path.isdir(args.label):
            label_paths = glob.glob(f"{args.label}/*.tsv")
        else:
            label_paths = [args.label]

        for labelpath in label_paths:
            labelname = os.path.basename(labelpath)[:-4]
            modelname_labelname = modelname+'-updatedlabel/'+labelname
            outdir = os.path.join(args.output, modelname_labelname)
            os.makedirs(outdir, exist_ok = True)
            data_df = process_inferece(data_df_original.copy(), labelpath, outdir, modelname, baseline_pred)
            data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
            roc_auc = plot_roc(data_df, outdir, modelname)
            pr_auc = plot_pr(data_df, outdir, modelname)
            r2 = plot_scatter(data_df, outdir, modelname)
            plot_histogram_site(data_df, outdir, modelname)
            print(f"{modelname}\t{labelname}\t{roc_auc:.3f}\t{pr_auc:.3f}\t{r2:.3f}")
            result_dict["model"].append(modelname)
            result_dict["roc_auc"].append(roc_auc)
            result_dict["pr_auc"].append(pr_auc)
            result_dict["r2"].append(r2)
            result_dict["label"].append(labelname)
    print("==========================")
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"{args.output}/result.tsv", sep = "\t", index = False)
    return None


if __name__ == "__main__":
    main()



