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
    parser.add_argument("--dorado", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-genomic/gene_df_final.pkl", help="Dorado path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/v2Genomic.GP3.depth5_None.twm6astrict.pkl", help="Label path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--min_depth", "-md", type=int, default=20, help="Minimum depth")
    parser.add_argument("--max_depth", "-xd", type=int, default=0, help="Maximum depth")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)
    return args

def process_dorado_inferece(data_path):
    df = pd.read_pickle(data_path)
    df = df.reset_index(drop = True)
    df = df[["genome_id", "dom", "count_dom"]]
    df = df.rename(columns = {"dom": "pred_dorado_070", "count_dom": "count_dorado_070"})
    return df

def process_label(label_path):
    df = pd.read_pickle(label_path)
    df = df.reset_index()
    df = df.rename(columns = {"m6A_level": "dom_label"})
    df = df[df["label"]>=0]
    return df


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
    # max_f1 = max_f1_score(data_df["label"], data_df["pm6a_070"])
    ax.plot(recall, precision, lw=3, label=f'AIRNA (v0.7) (AUC = {pr_auc:.3f})', color = "royalblue", zorder = 3)

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_dorado_070"])
    pr_auc = auc(recall, precision)
    # max_f1 = max_f1_score(data_df["label"], data_df["pred_dorado_070"])
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


def main():
    args = parse_args()



    # label_df = process_label(args.label)
    # print(label_df)
    # printmessage(f"Label count: {len(label_df)}")
    # dorado_pred = process_dorado_inferece(args.dorado)
    # printmessage(f"Dorado v0.7 prediction count: {len(dorado_pred)}")
    #
    # dorado_pred = dorado_pred.merge(label_df[["genome_id"]], on = "genome_id", how = "right")
    # print(dorado_pred)

    for input_path in args.input:
        modelname = "-".join(input_path.split("/")[-2].split("-")[:7])
        outdir = f"{args.output}/{modelname}-D{args.min_depth}-Genomic"
        data_path = f"{outdir}/inference_{modelname}.pkl"

        if os.path.exists(data_path):
            data_df = pd.read_pickle(data_path)
            print(data_df)
        else:
            data_df = pd.read_pickle(input_path)
            printmessage(f"AIRNA v0.7 prediction  count: {len(data_df)}")
            data_df.rename(columns = {"pm6a": "pm6a_070", "dom": "dom_070", "count_pm6a": "count_pm6a_070", "count_dom": "count_dom_070"}, inplace = True)
            data_df.reset_index(inplace = True, drop = True)
            data_df = data_df[["genome_id", "pm6a_070", "dom_070", "count_pm6a_070", "count_dom_070"]]
            data_df = data_df.merge(label_df, on = "genome_id", how = "right")
            data_df = data_df.merge(dorado_pred, on = "genome_id", how = "outer")
            data_df.fillna(0, inplace = True)
            gc.collect()

            print(data_df)
            printmessage("Data merged")

            os.makedirs(outdir, exist_ok = True)
            data_df.to_pickle(data_path)
            printmessage(f"Data saved to {data_path}")


        data_df_selected = data_df[(data_df["count_pm6a_070"] >= 20) & (data_df["count_dorado_070"] >= 20)]
        data_df_selected = data_df_selected[(data_df_selected["label"] ==  0)|(data_df_selected["dom_label"] > 0.05)]

        comment = "_070_depth_20_inner_glori"
        plot_roc(data_df_selected, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_selected, outdir, modelname, comment)
        printmessage("Plotted PR")
        # plot_scatter(data_df_selected, outdir, modelname, comment)
        # printmessage("Plotted scatter")
        # plot_histogram_site(data_df_selected, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_selected, outdir, modelname, comment)
        # printmessage("Plotted calibration")

        data_df_nondrach = data_df_selected[~data_df_selected["drach"]]

        comment = "_070_depth_20_nondrach_inner_glori"
        plot_roc(data_df_nondrach, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_nondrach, outdir, modelname, comment)
        printmessage("Plotted PR")
        # plot_scatter(data_df_nondrach, outdir, modelname, comment)
        # printmessage("Plotted scatter")
        # plot_histogram_site(data_df_nondrach, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_nondrach, outdir, modelname, comment)
        # printmessage("Plotted calibration")

        data_df_drach = data_df_selected[data_df_selected["drach"]]

        comment = "_070_depth_20_drach_inner_glori"
        plot_roc(data_df_drach, outdir, modelname, comment)
        printmessage("Plotted ROC")
        plot_pr(data_df_drach, outdir, modelname, comment)
        printmessage("Plotted PR")
        # plot_scatter(data_df_drach, outdir, modelname, comment)
        # printmessage("Plotted scatter")
        # plot_histogram_site(data_df_drach, outdir, modelname, comment)
        # printmessage("Plotted histogram")
        # plot_calibration(data_df_drach, outdir, modelname, comment)
        # printmessage("Plotted calibration")
    return None

if __name__ == "__main__":
    main()



