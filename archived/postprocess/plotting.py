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
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GP3.depth20.twm6a.notsampled.drach.tsv", help="Label path")
    parser.add_argument("--ratio", "-r", type=int, default=10, help="Neg/Pos ratio")
    parser.add_argument("--depth", "-m", type=int, default=20, help="Minimum depth")
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



def process_inferece(data_df, updated_label_path, outdir, modelname, baseline_pred, dorado_pred, ratio, min_depth = 10):
    plot_histogram_read(data_df, outdir, modelname)
    # threshold = gmm_em_lda(data_df["pred"].values)
    threshold_neg = 0.10 ## pre-calculated threshold using GMM-EM-LDA (It is too slow to calculate every time)
    threshold_pos = 0.99
    threshold_dom_neg = 0.30
    threshold_dom_pos = 0.70
    epsilon = 1e-12
    data_df["count"] = 1
    data_df_bin = data_df[(data_df["original"] <= threshold_neg) | ( data_df["original"] >= threshold_pos)].copy()
    data_df_dom = data_df[(data_df["original"] <= threshold_dom_neg) | ( data_df["original"] >= threshold_dom_pos)].copy()
    data_df_dom["dom"] = data_df_dom["original"].apply(lambda x: 1 if x >=threshold_dom_pos else 0)
    ## Groupby label_id and get mean of predictions
    data_df_bin.rename(columns = {"original": "pred_ari"}, inplace = True)
    data_df_bin["pred_geo"] = np.clip(data_df_bin["pred_ari"].to_numpy(), 0.0, 1 - epsilon)
    data_df_bin["pred_geo"] = np.log10(1 - data_df_bin["pred_geo"].to_numpy())
    ## groupby label_id and remove max
    data_df_bin = data_df_bin.groupby("label_id").agg({"pred_ari": "mean", "pred_geo": "mean"}).reset_index()
    data_df_dom = data_df_dom.groupby("label_id").agg({"dom": "mean"}).reset_index()

    data_df.rename(columns = {"pred": "pred_mlp_ari"}, inplace = True)
    data_df["pred_mlp_geo"] = np.clip(data_df["pred_mlp_ari"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred_mlp_geo"] = np.log10(1 - data_df["pred_mlp_geo"].to_numpy())
    data_df["pred_mlp_dom"] = data_df["pred_mlp_ari"].apply(lambda x: 1 if x >= 0.5 else 0)
    data_df = data_df.groupby("label_id").agg({"pred_mlp_dom": "mean", "pred_mlp_geo": "mean", "pred_mlp_ari": "mean", "count": "sum"}).reset_index()

    data_df_bin = data_df_bin.merge(data_df_dom, on = "label_id", how = "inner")
    data_df = data_df.merge(data_df_bin, on = "label_id", how = "left")
    data_df.fillna(0, inplace = True)

    data_df["pred_geo"] = np.clip(data_df["pred_geo"], None, 0)
    data_df["pred_geo"] = (1 - 10**data_df["pred_geo"].to_numpy())

    data_df["pred_mlp_geo"] = np.clip(data_df["pred_mlp_geo"], None, 0)
    data_df["pred_mlp_geo"] = (1 - 10**data_df["pred_mlp_geo"].to_numpy())

    data_df = data_df[data_df["count"] >= 5].copy()

    if baseline_pred is not None:
        data_df = data_df.merge(baseline_pred, on = "label_id", how = "inner")
    if dorado_pred is not None:
        data_df = data_df.merge(dorado_pred, on = "label_id", how = "inner")

    updated_label_df = pd.read_csv(updated_label_path, sep = "\t")
    updated_label_df = updated_label_df[["id", "label", "m6A_level"]]
    updated_label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    data_df = data_df.merge(updated_label_df, on = "label_id", how = "inner")
    data_df = data_df[data_df["count"] >= min_depth].copy()
    data_df.fillna(0, inplace = True)

    data_df = data_df[data_df["label"] >= 0]

    # ## match positive:negative ratio
    # pos_df = data_df[data_df["label"] == 1]
    # neg_df = data_df[data_df["label"] == 0]
    # if len(neg_df) > len(pos_df)*ratio:
    #     neg_df = neg_df.sample(n = len(pos_df)*ratio, random_state = 42)
    # else:
    #     pos_df = pos_df.sample(n = len(neg_df)//ratio, random_state = 42)
    # data_df = pd.concat([pos_df, neg_df], ignore_index = True)

    data_df["position"] = data_df["label_id"].str.split(":").str[1].astype(int)

    return data_df


def plot_roc(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_geo"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Transformer (AUC = {roc_auc:.3f})', color = "royalblue", zorder = 3)
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_mlp_geo"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Transformer+MLP (AUC = {roc_auc:.3f})', color = "darkorchid", zorder = 3)
    if "pred_m6anet" in data_df.columns:
        fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_m6anet"])
        roc_auc_m6anet = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=3,label=f'm6Anet (AUC = {roc_auc_m6anet:.3f})', color = "tomato", zorder = 2)
        ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    if "pred_dorado" in data_df.columns:
        fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_dorado"])
        roc_auc_dorado = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=3,label=f'Dorado (AUC = {roc_auc_dorado:.3f})', color = "forestgreen", zorder = 1)
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


def plot_pr(data_df, outdir, pr_baseline_level, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot([0, 1], [pr_baseline_level, pr_baseline_level], color='grey', lw=2, linestyle='--')

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_geo"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_geo"])
    ax.plot(recall, precision, lw=3, label=f'Transformer (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "royalblue", zorder = 3)


    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_mlp_geo"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_mlp_geo"])
    ax.plot(recall, precision, lw=3, label=f'Transformer+MLP (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "darkorchid", zorder = 3)

    if "pred_m6anet" in data_df.columns:
        precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_m6anet"])
        pr_auc_m6anet = auc(recall, precision)
        max_f1 = max_f1_score(data_df["label"], data_df["pred_m6anet"])
        ax.plot(recall, precision, lw=3,label=f'm6Anet (AUC = {pr_auc_m6anet:.3f}, Max F-1 = {max_f1:.3f})', color = "tomato", zorder = 2)

    if "pred_dorado" in data_df.columns:
        precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_dorado"])
        pr_auc_dorado = auc(recall, precision)
        max_f1 = max_f1_score(data_df["label"], data_df["pred_dorado"])
        ax.plot(recall, precision, lw=3,label=f'Dorado (AUC = {pr_auc_dorado:.3f}, Max F-1 = {max_f1:.3f})', color = "forestgreen", zorder = 1)

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
    fig, axes = plt.subplots(1,5, figsize = (100,20))
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
    ax.scatter(data_df["dom_label"], data_df["pred_mlp_ari"], s = 10, alpha = 0.5, color = "darkorchid")
    r2 = r2_score(data_df["dom_label"], data_df["pred_mlp_ari"])
    rho2 = spearmanr(data_df["dom_label"], data_df["pred_mlp_ari"])[0] ** 2
    ax.set_title(f"Transformer+MLP, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["pred_mlp_ari"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    ax = axes[2]
    ax.scatter(data_df["dom_label"], data_df["pred_mlp_dom"], s = 10, alpha = 0.5, color = "darkorchid")
    r2 = r2_score(data_df["dom_label"], data_df["pred_mlp_dom"])
    rho2 = spearmanr(data_df["dom_label"], data_df["pred_mlp_dom"])[0] ** 2
    ax.set_title(f"Transformer+MLP+BIN, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["pred_mlp_dom"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    if "dom_m6anet" in data_df.columns:
        ax = axes[3]
        ax.scatter(data_df["dom_label"], data_df["dom_m6anet"], s = 10, alpha = 0.5, color = "tomato")
        r2b = r2_score(data_df["dom_label"], data_df["dom_m6anet"])
        rho2b = spearmanr(data_df["dom_label"], data_df["dom_m6anet"])[0] ** 2
        ax.set_title(f"m6Anet, R2 = {r2b:.3f}, rho2 = {rho2b:.3f}")
        z = np.polyfit(data_df["dom_label"], data_df["dom_m6anet"], 1)
        p = np.poly1d(z)
        x = np.linspace(0, 1, 100)
        ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
    if "pred_dorado" in data_df.columns:
        ax = axes[4]
        ax.scatter(data_df["dom_label"], data_df["pred_dorado"], s = 10, alpha = 0.5, color = "forestgreen")
        r2c = r2_score(data_df["dom_label"], data_df["pred_dorado"])
        rho2c = spearmanr(data_df["dom_label"], data_df["pred_dorado"])[0] ** 2
        ax.set_title(f"Dorado, R2 = {r2c:.3f}, rho2 = {rho2c:.3f}")
        z = np.polyfit(data_df["dom_label"], data_df["pred_dorado"], 1)
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
    ax.hist(data_df["original"], bins = 100, color = "royalblue", label = "Transformer")
    ax.hist(data_df["pred"], bins = 100, color = "tomato", label = "Transformer+MLP")
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
    ax.hist(data_df["pred_mlp_geo"], bins = 100, color = "tomato", label = "Transformer+MLP")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title(f"{modelname}")
    ax.legend()
    plt.savefig(f"{outdir}/hist_site.png")
    plt.close()
    return None


def plot_calibration(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    p_true, p_pred = calibration_curve(data_df["label"], data_df["pred_geo"], n_bins = 10)
    ece = np.abs(p_true - p_pred).mean()
    ax.plot(p_pred, p_true, lw=3, label=f'Transformer (ECE = {ece:.3f})', color = "royalblue", zorder = 3)
    p_true, p_pred = calibration_curve(data_df["label"], data_df["pred_mlp_geo"], n_bins = 10)
    ece = np.abs(p_true - p_pred).mean()
    ax.plot(p_pred, p_true, lw=3, label=f'Transformer+MLP (ECE = {ece:.3f})', color = "darkorchid", zorder = 3)
    if "pred_m6anet" in data_df.columns:
        p_true, p_pred = calibration_curve(data_df["label"], data_df["pred_m6anet"], n_bins = 10)
        ece = np.abs(p_true - p_pred).mean()
        ax.plot(p_pred, p_true,  lw=3,label=f'm6Anet (ECE = {ece:.3f})', color = "tomato", zorder = 2)
    if "pred_dorado" in data_df.columns:
        p_true, p_pred = calibration_curve(data_df["label"], data_df["pred_dorado"], n_bins = 10)
        ece = np.abs(p_true - p_pred).mean()
        ax.plot(p_pred, p_true,  lw=3,label=f'Dorado (ECE = {ece:.3f})', color = "forestgreen", zorder = 1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{modelname}\nCalibration Curve')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/calibration.png")
    plt.close()
    return None




def main():
    args = parse_args()
    pr_baseline_level = 1/(1+args.ratio)
    dorado_pred = process_dorado_inferece(args.dorado)
    baseline_pred = process_baseline_inferece(args.baseline)
    result_dict = {"model": [], "label": [],"roc_auc": [], "pr_auc": [], "r2": []}
    print("==========================")
    print("Model\tLabel\tROC_AUC\tPR_AUC\tR2")
    for idx, data_path in enumerate(args.input):
        data_paths = glob.glob(f"{data_path}/*.tsv")
        data_df_original = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths])
        data_df_original.fillna(0, inplace = True)
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
            modelname_labelname = modelname+'-updatedlabel-dorado/'+labelname
            outdir = os.path.join(args.output, modelname_labelname)
            os.makedirs(outdir, exist_ok = True)
            data_df = process_inferece(data_df_original.copy(), labelpath, outdir, modelname, baseline_pred, dorado_pred,
                                       args.ratio, args.depth)
            data_df.to_csv(f"{outdir}/{modelname}.tsv", sep = "\t", index = False)
            plot_calibration(data_df, outdir, modelname)
            roc_auc = plot_roc(data_df, outdir, modelname)
            pr_auc = plot_pr(data_df, outdir, pr_baseline_level, modelname)
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



