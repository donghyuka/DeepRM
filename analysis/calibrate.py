from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os, argparse, glob
from utils.utils import max_f1_score

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--val", dest="val_path", type=str, required=True, help="Val file path")
    args.add_argument("--test", dest="test_path", type=str, required=True, help="Test file path")
    args.add_argument("--out", dest="out_path", type=str, required=True, help="Output directory")
    args = args.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    return args

def plot_calibration_curve_read(pred_df, out_path):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(pred_df["label"], pred_df["pred"], n_bins=20)
    ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
    ax.plot(mean_predicted_value, fraction_of_positives, color = "royalblue", label=f"Transformer (ECE={ece:.3f})", linewidth = 3)
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.savefig(out_path)
    plt.close()
    return fraction_of_positives, mean_predicted_value

def plot_calibration_curve_site(pred_df, out_path):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(pred_df["label"], pred_df["pred_geo"], n_bins=20)
    ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
    ax.plot(mean_predicted_value, fraction_of_positives, label=f"Transformer (ECE={ece:.3f})", color = "royalblue", linewidth = 3)
    fraction_of_positives, mean_predicted_value = calibration_curve(pred_df["label"], pred_df["pred_geo_emp"], n_bins=20)
    ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
    ax.plot(mean_predicted_value, fraction_of_positives, label=f"Calibrated (ECE={ece:.3f})", color = "tomato", linewidth = 3)
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.savefig(out_path)
    plt.close()
    return None


def transform_score_to_empirical(pred_df, fraction_of_positives, mean_predicted_value):
    pred_df["pred_emp"] = pred_df["pred"].apply(lambda x: np.interp(x, mean_predicted_value, fraction_of_positives))
    return pred_df


def process_inferece(data_df, updated_label_path, ratio, min_depth = 20):

    epsilon = 1e-12
    ## Groupby label_id and get mean of predictions
    data_df["count"] = 1
    data_df["pred_geo"] = np.clip(data_df["pred"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred_geo"] = np.log10(1 - data_df["pred_geo"].to_numpy())
    data_df["pred_geo_emp"] = np.clip(data_df["pred_emp"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred_geo_emp"] = np.log10(1 - data_df["pred_geo_emp"].to_numpy())
    ## groupby label_id and remove max
    data_df = data_df.groupby("label_id").agg({"pred_geo_emp": "mean", "pred_geo": "mean",
                                                "count": "sum"}).reset_index()
    data_df["pred_geo"] = np.clip(data_df["pred_geo"], None, 0)
    data_df["pred_geo"] = (1 - 10**data_df["pred_geo"].to_numpy())
    data_df["pred_geo_emp"] = np.clip(data_df["pred_geo_emp"], None, 0)
    data_df["pred_geo_emp"] = (1 - 10**data_df["pred_geo_emp"].to_numpy())
    data_df = data_df[data_df["count"] >= min_depth].copy()

    updated_label_df = pd.read_csv(updated_label_path, sep = "\t")
    updated_label_df = updated_label_df[["id", "label", "m6A_level"]]
    updated_label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    data_df = data_df.merge(updated_label_df, on = "label_id", how = "inner")

    data_df = data_df[data_df["label"] >= 0]

    ## match positive:negative ratio
    pos_df = data_df[data_df["label"] == 1]
    neg_df = data_df[data_df["label"] == 0]
    if len(neg_df) > len(pos_df)*ratio:
        neg_df = neg_df.sample(n = len(pos_df)*ratio, random_state = 42)
    else:
        pos_df = pos_df.sample(n = len(neg_df)//ratio, random_state = 42)
    data_df = pd.concat([pos_df, neg_df], ignore_index = True)

    return data_df


def plot_roc(data_df, outdir, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_geo"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Transformer (AUC = {roc_auc:.3f})', color = "royalblue", zorder = 3)
    fpr, tpr, _ = roc_curve(data_df["label"], data_df["pred_geo_emp"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=3, label=f'Calibrated (AUC = {roc_auc:.3f})', color = "tomato", zorder = 3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{modelname}\nReceiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(f"{outdir}/roc.png")
    plt.close()
    return None


def plot_pr(data_df, outdir, pr_baseline_level, modelname):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot([0, 1], [pr_baseline_level, pr_baseline_level], color='grey', lw=2, linestyle='--')

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_geo"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_geo"])
    ax.plot(recall, precision, lw=3, label=f'Transformer (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "royalblue", zorder = 3)

    precision, recall, _ = precision_recall_curve(data_df["label"], data_df["pred_geo_emp"])
    pr_auc = auc(recall, precision)
    max_f1 = max_f1_score(data_df["label"], data_df["pred_geo_emp"])
    ax.plot(recall, precision, lw=3, label=f'Calibrated (AUC = {pr_auc:.3f}, Max F-1 = {max_f1:.3f})', color = "tomato", zorder = 3)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{modelname}\nPrecision-Recall')
    ax.legend(loc="lower left")
    plt.savefig(f"{outdir}/pr.png")
    plt.close()
    return None


def main():
    args = parse_args()
    val_pred_df = pd.concat([pd.read_csv(x, sep='\t') for x in glob.glob(f"{args.val_path}/*.tsv")])
    test_pred_df = pd.concat([pd.read_csv(x, sep='\t') for x in glob.glob(f"{args.test_path}/*.tsv")])
    fop, mpv = plot_calibration_curve_read(val_pred_df, f"{args.out_path}/val_calibration_curve.png")
    test_pred_df = transform_score_to_empirical(test_pred_df, fop, mpv)
    test_pred_df.to_pickle(f"{args.out_path}/test_pred_df.pkl")
    data_df = process_inferece(test_pred_df, updated_label_path = "/extdata2/baeklab/Hyeonseo/m6A/inference/inference/inference/calibration/BaeklabV2_GP3.depth20.twm6astrict.notsampled.drach.tsv", ratio = 10)
    data_df.to_csv(f"{args.out_path}/test_pred_df.tsv", sep="\t", index=False)
    plot_roc(data_df, args.out_path, "Transformer")
    plot_pr(data_df, args.out_path, 1/11, "Transformer")
    plot_calibration_curve_site(data_df, f"{args.out_path}/test_calibration_curve.png")
    return None

if __name__ == "__main__":
    main()


