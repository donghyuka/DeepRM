from utils.utils import parse_refflat
import pandas as pd
import numpy as np


def get_stop_codon_pos(refflat_row):
    ## Get the exon starts and ends
    exon_starts = refflat_row["exonStarts"]
    exon_ends = refflat_row["exonEnds"]
    strand = refflat_row["strand"]
    if strand=="+":
        stop_codon_pos = refflat_row["cdsEnd"] - 1
    elif strand=="-":
        stop_codon_pos = refflat_row["cdsStart"]
    else:
        raise ValueError("Strand must be either + or -")
    exons=np.stack((exon_starts,exon_ends),axis=1)
    exon_cumsum=np.concatenate(([0],np.cumsum(exons[:,1]-exons[:,0])))
    exon_index = np.searchsorted(exons[:,0],stop_codon_pos,side="right")-1

    mrna_coordinate=exon_cumsum[exon_index]+(stop_codon_pos-exons[exon_index,0])
    mrna_length=exon_cumsum[-1]

    if strand=="-":
        mrna_coordinate=mrna_length-mrna_coordinate-1

    return mrna_coordinate, mrna_length

#
# refflat_df = parse_refflat()
# refflat_df["coding"] = refflat_df.index.str.startswith("NM")
# refflat_df = refflat_df[refflat_df["coding"]]
# refflat_df[["stop_codon_pos","txlen"]] = refflat_df.apply(lambda x: get_stop_codon_pos(x), axis=1, result_type="expand")
# refflat_df.reset_index(inplace=True, drop=False)
# refflat_df = refflat_df[["NMID", "stop_codon_pos", "txlen"]]
# print(refflat_df)
#
# path = "/extdata4/baeklab/Hyeonseo/m6A/inference/plot/AIRNA-DW-v2-20240827-175235-22-373000-D20/inference_AIRNA-DW-v2-20240827-175235-22-373000.pkl"
# df = pd.read_pickle(path)
# print(df)
#
# df["NMID"] = df["label_id"].apply(lambda x: x.split(":")[0])
# df["pos"] = df["label_id"].apply(lambda x: int(x.split(":")[1]))
# df = df.merge(refflat_df, how="inner", on="NMID")
# df["distance_stop"] = df["pos"] - df["stop_codon_pos"]
# print(df)
# df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/plot/AIRNA-DW-v2-20240827-175235-22-373000-D20/inference_AIRNA-DW-v2-20240827-175235-22-373000_stop.pkl")
#
# df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/plot/AIRNA-DW-v2-20240827-175235-22-373000-D20/inference_AIRNA-DW-v2-20240827-175235-22-373000_stop.pkl")
#
# def get_group(row):
#     if row["drach"]:
#         drach = "DRACH"
#     else:
#         drach = "non-DRACH"
#     if row["label"] == 0:
#         label = "cA"
#     else:
#         label = "m6A"
#     return f"{drach} {label}"
#
# df = df[(df["count_pm6a_070"] >= 20) & (df["dorado_count"] >= 20)]
# df_pos = df[df["label"] == 1]
# df_neg = df[df["label"] == 0].sample(len(df_pos))
# df = pd.concat([df_pos, df_neg])
# df["group"] = df.apply(lambda x: get_group(x), axis=1)
#
# print(df)
#
#
# ## Make Histogram
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fig, ax = plt.subplots(figsize=(20,10))
# sns.histplot(data=df, x="distance_stop", hue="group",
#              ax=ax, kde=False, stat="density",fill=False,
#              linewidth = 3, element="step", legend=True,
#              binwidth=30, binrange=(-1000,1000), alpha = 0.5,
#              common_norm=False)
#
# ax.set_title("Distance from stop codon")
# ax.set_xlabel("Distance from stop codon (nt)")
# ax.set_ylabel("Density")
#
# plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/distance_stop.png", dpi=300)



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
import os


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



def main():

    data_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/plot/AIRNA-DW-v2-20240827-175235-22-373000-D20/inference_AIRNA-DW-v2-20240827-175235-22-373000_stop.pkl")

    data_df_selected = data_df[(data_df["count_pm6a_070"] >= 100) & (data_df["dorado_count"] >= 100)]
    data_df_selected = data_df_selected[(data_df_selected["label"] ==  0)|(data_df_selected["dom_label"] > 0.1)]
    data_df_selected = data_df_selected[(data_df_selected["distance_stop"]<=100)&(data_df_selected["distance_stop"]>=-300)]

    print(data_df_selected)
    print(data_df_selected["label"].value_counts())

    outdir = "/extdata4/baeklab/Hyeonseo/m6A/inference/plot/AIRNA-DW-v2-20240827-175235-22-373000-D20/"
    modelname = "AIRNA-DW-v2-20240827-175235-22-373000-D20"

    comment = "_070_depth_100_nearstop"
    plot_pr(data_df_selected, outdir, modelname, comment)
    printmessage("Plotted PR")

    data_df_nondrach = data_df_selected[~data_df_selected["drach"]]
    comment = "_070_depth_100_nearstop_nondrach"
    plot_pr(data_df_nondrach, outdir, modelname, comment)
    printmessage("Plotted PR")

    data_df_drach = data_df_selected[data_df_selected["drach"]]
    comment = "_070_depth_100_nearstop_drach"
    plot_pr(data_df_drach, outdir, modelname, comment)
    printmessage("Plotted PR")

    return None

if __name__ == "__main__":
    main()




