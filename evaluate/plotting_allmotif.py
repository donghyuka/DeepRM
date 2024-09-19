import gc
import os

# from tqdm import tqdm
# import glob
# default_n_threads = os.cpu_count() // 2
# os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
# os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
# os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.calibration import calibration_curve
import argparse
from utils.utils import printmessage
from utils.utils import max_f1_score
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input path", nargs="+")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.bed", help="Dorado path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.tsv", help="Label path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--min_depth", "-md", type=int, default=20, help="Minimum depth")
    parser.add_argument("--max_depth", "-xd", type=int, default=0, help="Maximum depth")
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


def process_m6anet_inferece(data_path):
    data_df = pd.read_csv(data_path)
    data_df["id"] = data_df["transcript_id"].str.split(".").str[0]
    data_df["id"] = data_df["id"] + ":" + data_df["transcript_position"].astype(str)
    data_df.rename(columns = {"id": "label_id", "probability_modified": "pm6a_m6anet", "mod_ratio": "dom_m6anet"}, inplace = True)
    data_df = data_df[["label_id", "pm6a_m6anet", "dom_m6anet"]]
    return data_df


def process_dorado_inferece(data_path):
    if data_path is None:
        return None
    if os.path.exists(data_path.replace(".bed", ".pkl")):
        data_df = pd.read_pickle(data_path.replace(".bed", ".pkl"))
    else:
        data_df = pd.read_csv(data_path, quoting = 3, sep = "\t", header = None, dtype=str)
        ## Keep column 0, 1, 4, 9
        data_df = data_df[[0, 1, 4, 9]]
        data_df.columns = ["nmid", "pos", "depth", "pred_dorado"]
        data_df["depth"] = data_df["depth"].astype(int)
        data_df["pred_dorado"] = data_df["pred_dorado"].str.split(" ").str[1].astype(float) / 100
        data_df["label_id"] = data_df["nmid"].str.split(".").str[0] + ":" + data_df["pos"]
        data_df = data_df[["label_id", "pred_dorado","depth"]].copy()
        data_df.rename(columns = {"depth": "dorado_count"}, inplace = True)
        data_df.to_pickle(data_path.replace(".bed", ".pkl"))

    return data_df


def process_label(label_path):

    label_df = pd.read_pickle(label_path.replace(".tsv", ".pkl"))
    label_df = label_df[["id", "depth", "label", "m6A_level", "5mer", "drach"]]
    label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    label_df = label_df[label_df["label"] >= 0].copy()

    return label_df


def plot_roc(data_df, outdir, modelname, comment=""):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))

    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pm6a_070"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'AIRNA (v0.7) (AUC = {roc_auc:.3f})', color = "royalblue", zorder = 3)

    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_dorado_070"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Dorado (v0.7) (AUC = {roc_auc:.3f})', color = "tomato", zorder = 3)

    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{modelname}{comment}\nReceiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/roc_{comment}.png")
    plt.close()
    return None


def plot_pr(data_df, outdir, modelname ,comment=""):
    pr_baseline_level = data_df["label"].mean()

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot([0, 1], [pr_baseline_level, pr_baseline_level], color='grey', lw=2, linestyle='--')

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pm6a_070"])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=3, label=f'AIRNA (v0.7) (AUC = {pr_auc:.3f})', color = "royalblue", zorder = 3)

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_dorado_070"])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=3, label=f'Dorado (v0.7) (AUC = {pr_auc:.3f})', color = "tomato", zorder = 3)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{modelname}{comment}\nPrecision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr_{comment}.png")
    plt.close()
    return None


def plot_scatter(data_df, outdir, modelname, comment=""):
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1,3, figsize = (60,20))

    data_df = data_df[data_df["dom_label"] > 0]

    ax = axes[0]
    ax.scatter(data_df["dom_label"], data_df["dom_070"], s = 10, alpha = 0.5, color = "royalblue")
    r2 = r2_score(data_df["dom_label"], data_df["dom_070"])
    rho2 = spearmanr(data_df["dom_label"], data_df["dom_070"])[0] ** 2
    ax.set_title(f"AIRNA (v0.7), R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    ## polyfit
    z = np.polyfit(data_df["dom_label"], data_df["dom_070"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)
 
    ax = axes[1]
    ax.scatter(data_df["dom_label"], data_df["pred_dorado_070"], s = 10, alpha = 0.5, color = "tomato")
    r2d = r2_score(data_df["dom_label"], data_df["pred_dorado_070"])
    rho2d = spearmanr(data_df["dom_label"], data_df["pred_dorado_070"])[0] ** 2
    ax.set_title(f"Dorado (v0.7), R2 = {r2d:.3f}, rho2 = {rho2d:.3f}")
    z = np.polyfit(data_df["dom_label"], data_df["pred_dorado_070"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)

    ax = axes[2]
    ax.scatter(data_df["pred_dorado_070"], data_df["dom_070"], s = 10, alpha = 0.5, color = "forestgreen")
    r2 = r2_score(data_df["pred_dorado_070"], data_df["dom_070"])
    rho2 = spearmanr(data_df["pred_dorado_070"], data_df["dom_070"])[0] ** 2
    ax.set_title(f"Dorado vs. AIRNA, R2 = {r2:.3f}, rho2 = {rho2:.3f}")
    z = np.polyfit(data_df["pred_dorado_070"], data_df["dom_070"], 1)
    p = np.poly1d(z)
    x = np.linspace(0, 1, 100)
    ax.plot(x, p(x), color = "grey", linestyle = "--", lw = 3)


    for ax in axes:
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('GLORI DoM')
        ax.set_ylabel('Predicted DoM')

    fig.suptitle(f"{modelname}{comment}")
    plt.savefig(f"{outdir}/scatter_{comment}.png")
    plt.close()
    return r2


def plot_histogram_site(data_df, outdir, modelname, comment=""):
    ## Plot histogram of predictions
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1, 2, figsize = (40,20))

    ax = axes[0]
    ax.set_title("p(m6A) Distribution")
    sns.kdeplot(data_df["pm6a_070"], ax = ax, label = "AIRNA (v0.7)", color = "royalblue", lw = 3, fill=False)
    sns.kdeplot(data_df["pred_dorado_070"], ax = ax, label = "Dorado (v0.7)", color = "tomato", lw = 3, fill=False)
    ax.set_xlabel("p(m6A) Prediction")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    ax.set_title("DoM Distribution, m6A sites")
    data_df = data_df[data_df["dom_label"] > 0].copy()
    sns.kdeplot(data_df["dom_070"], ax = ax, label = "AIRNA (v0.7)", color = "royalblue", lw = 3, fill=False)
    sns.kdeplot(data_df["pred_dorado_070"], ax = ax, label = "Dorado (v0.7)", color = "tomato", lw = 3, fill=False)
    sns.kdeplot(data_df["dom_label"], ax = ax, label = "GLORI", color = "slategrey", lw = 3, fill=False)
    ax.set_xlabel("DoM Prediction")
    ax.set_ylabel("Count")
    ax.legend()

    fig.suptitle(f"{modelname}{comment}")

    plt.savefig(f"{outdir}/hist_site_{comment}.png")
    plt.close()
    return None


def plot_calibration(data_df, outdir, modelname, comment=""):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    p_true, p_pred = calibration_curve(data_df["label"], data_df["pm6a_070"], n_bins = 10)
    ece = np.abs(p_true - p_pred).mean()
    ax.plot(p_pred, p_true, lw=3, label=f'AIRNA (v0.7) (ECE = {ece:.3f})', color = "royalblue", zorder = 2)

    p_true, p_pred = calibration_curve(data_df["label"], data_df["pred_dorado_070"], n_bins = 10)
    ece = np.abs(p_true - p_pred).mean()
    ax.plot(p_pred, p_true, lw=3, label=f'Dorado (v0.7) (ECE = {ece:.3f})', color = "tomato", zorder = 1)


    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{modelname}{comment}\nCalibration Curve')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/calibration_{comment}.png")
    plt.close()
    return None


def get_drach_distance(motif):
    distance = 0
    if motif[0] == "C":
        distance += 1
    if motif[1] in ["C", "T"]:
        distance += 1
    elif motif[3] != "C":
        distance += 1
    if motif[4] == "G":
        distance += 1
    return distance


def main():
    args = parse_args()

    # dorado_pred = process_dorado_inferece(args.dorado)
    # printmessage(f"Dorado v0.7 prediction count: {len(dorado_pred)}")
    # dorado_pred.rename(columns = {"pred_dorado": "pred_dorado_070", "count_dorado": "count_dorado_070"}, inplace = True)
    # print(dorado_pred)
    # label_df = process_label(args.label)
    # printmessage(f"Label count: {len(label_df)}")
    # print(label_df)
    #
    # dorado_pred = dorado_pred.merge(label_df[["label_id"]], on = "label_id", how = "right")

    for input_path in args.input:
        modelname = "-".join(input_path.split("/")[-2].split("-")[:7])
        outdir = f"{args.output}/{modelname}-D{args.min_depth}"
        data_path = f"{outdir}/inference_{modelname}.pkl"

        if os.path.exists(data_path):
            data_df = pd.read_pickle(data_path)
            print(data_df)
        else:
            data_df = pd.read_pickle(input_path)
            printmessage(f"AIRNA v0.7 prediction  count: {len(data_df)}")
            data_df.rename(columns = {"pm6a": "pm6a_070", "dom": "dom_070", "count_pm6a": "count_pm6a_070", "count_dom": "count_dom_070"}, inplace = True)

            data_df = data_df.merge(label_df, on = "label_id", how = "right")
            data_df = data_df.merge(dorado_pred, on = "label_id", how = "outer")
            data_df.fillna(0, inplace = True)
            gc.collect()

            print(data_df)
            printmessage("Data merged")

            os.makedirs(outdir, exist_ok = True)
            data_df.to_pickle(data_path)
            printmessage(f"Data saved to {data_path}")

        if args.max_depth is None or args.max_depth == 0:
            data_df = data_df[data_df["depth"] >= args.min_depth].copy()
        else:
            data_df = data_df[(data_df["depth"] >= args.min_depth) & (data_df["depth"] <= args.max_depth)].copy()

        data_df_selected = data_df[(data_df["count_pm6a_070"] >= 20) & (data_df["dorado_count"] >= 20)]
        # data_df_selected = data_df_selected[(data_df_selected["label"] ==  0)|(data_df_selected["dom_label"] > 0.1)]
        data_df_selected["threshold"] = 1-((0.9 ** (1-data_df_selected["dom_070"])) * (0.02 ** (data_df_selected["dom_070"])))
        data_df_selected["pm6a_070"] = data_df_selected.apply(lambda x: x["pm6a_070"] if x["pm6a_070"] > x["threshold"] else 0, axis = 1)
        # data_df_selected = data_df[data_df["depth"] >= 20]
        # data_df_selected = data_df_selected[(data_df_selected["label"] ==  0)|(data_df_selected["dom_label"] > 0.05)]

        # data_df_selected["edit_distance"] = data_df_selected["5mer"].apply(get_drach_distance)
        # data_df_selected = data_df_selected[data_df_selected["edit_distance"] <= 1]

        comment = "_070_depth_20_inner_lb"
        plot_roc(data_df_selected, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_selected, outdir, modelname, comment)
        printmessage("Plotted PR")
        plot_scatter(data_df_selected, outdir, modelname, comment)
        printmessage("Plotted scatter")
        # plot_histogram_site(data_df_selected, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_selected, outdir, modelname, comment)
        # printmessage("Plotted calibration")

        data_df_nondrach = data_df_selected[~data_df_selected["drach"]]

        comment = "_070_depth_20_nondrach_inner_lb"
        plot_roc(data_df_nondrach, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_nondrach, outdir, modelname, comment)
        printmessage("Plotted PR")
        plot_scatter(data_df_nondrach, outdir, modelname, comment)
        printmessage("Plotted scatter")
        # plot_histogram_site(data_df_nondrach, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_nondrach, outdir, modelname, comment)
        # printmessage("Plotted calibration")

        data_df_drach = data_df_selected[data_df_selected["drach"]]

        comment = "_070_depth_20_drach_inner_lb"
        plot_roc(data_df_drach, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_drach, outdir, modelname, comment)
        printmessage("Plotted PR")
        plot_scatter(data_df_drach, outdir, modelname, comment)
        printmessage("Plotted scatter")
        # plot_histogram_site(data_df_drach, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_drach, outdir, modelname, comment)
        # printmessage("Plotted calibration")
    return None

if __name__ == "__main__":
    main()



