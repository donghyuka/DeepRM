from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, r2_score

PLOTDIR = ""

def plot_roc_prc(pred_df, expname, modelname, comment = ""):
    ## Remove rows with label -1
    pred_df = pred_df[pred_df["label"] >= 0]

    ## Match TP and TN count
    pos_n = pred_df[pred_df["label"] == 1].shape[0]
    neg_n = pred_df[pred_df["label"] == 0].shape[0]

    pos_df = pred_df[pred_df["label"] == 1]
    neg_df = pred_df[pred_df["label"] == 0].sample(n=pos_n*10, random_state=42)

    pred_df = pd.concat([pos_df, neg_df], axis=0)

    x = pred_df["preds"]
    y = pred_df["label"]

    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='ROC')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC (AUC = {roc_auc:.3f})')

    ax = axes[1]
    precision, recall, thresholds = precision_recall_curve(y, x)
    prc_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=2, label='PRC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PRC (AUC = {prc_auc:.3f})')

    f1 = max_f1_score(y, x)
    plt.suptitle(f'{expname} {modelname} (F1={f1:.3f})')

    plt.tight_layout()
    plt.savefig(f'{PLOTDIR}{expname}_{modelname}_{comment}.png')

    return None


def max_f1_score(y_true, y_pred):
    f1_scores = []
    for threshold in np.arange(0, 1, 0.01):
        y_pred_ = y_pred > threshold
        f1_scores.append(f1_score(y_true, y_pred_))
    return max(f1_scores)


def plot_pred_distribution(pred_df, expname, modelname, comment=""):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.histplot(pred_df["preds"], ax=ax, kde=True, bins=20)
    ax.set_title(f'{expname} {modelname}')
    plt.savefig(f'{PLOTDIR}{expname}_{modelname}_{comment}_hist.png')
    print("Saved to", f'{PLOTDIR}{expname}_{modelname}_{comment}_hist.png')

    return None

def plot_scatter(pred_df,expname, modelname, comment=""):
    pred_df = pred_df[pred_df["glori_label"]>0]

    ## Draw scatter plot
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_xlabel("Preds")
    ax.set_ylabel("Label")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.plot([0,1],[0,1], color="red", linestyle="--")

    sns.scatterplot(x=pred_df["preds"], y=pred_df["glori_label"], ax=ax, s=10)

    r2 = r2_score(pred_df["glori_label"], pred_df["preds"])
    ax.set_title(f"Preds vs Label (R2={r2:.3f})")
    plt.savefig(f'{PLOTDIR}{expname}_{modelname}_{comment}_scatter.png')

    return None
