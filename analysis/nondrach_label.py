import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import itertools as it

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


def comb_label(data_df):
    cols_list = ["SAC", "GLORI", "MICLIP2", "M6ACE"]
    required_subsets = list(it.chain.from_iterable(it.combinations(cols_list, r) for r in range(0, len(cols_list)+1)))
    label_dict = {}

    for required in required_subsets:
        non_required = [col for col in cols_list if col not in required]
        optional_subsets = list(it.chain.from_iterable(it.combinations(non_required, r) for r in range(0, len(non_required)+1)))
        for optional in optional_subsets:
            for support in range(0, len(optional)+1):
                label_name = f"LAB-R[{'.'.join(required)}]-O[{'.'.join(optional)}]-S{support}"
                label = get_label(data_df, required, optional, support)
                if label is not None:
                    print(f"{label_name}: {np.sum([label==1]):,} vs. {np.sum([label==0]):,}")
                    label_dict[label_name] = label

    return label_dict


def get_label(data_df, required, optional, support):

    if len(required) == 0 and (len(optional) == 0 or support == 0):
        return None

    required = list(required)
    optional = list(optional)

    if len(required) == 0:
        has_all_required = np.ones(len(data_df)).astype(bool)
    else:
        has_all_required = data_df[required].all(axis = 1)

    if len(optional) == 0 or support == 0:
        has_support = np.ones(len(data_df))
    else:
        has_support = data_df[optional].sum(axis = 1) >= support

    required_and_optional = required + optional
    has_negative_support = (data_df[required_and_optional].sum(axis = 1) == 0)
    label = has_all_required & has_support
    label = label.astype(int)
    label[(~has_negative_support)&(~label)] = -1

    return label


def main():
    # label_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.pkl")
    # label_df = label_df[["id", "depth", "label", "m6A_level", "5mer", "drach"]]
    # label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    # print(label_df)
    #
    # additional_label_df = pd.read_csv("/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/m6A_Jungmin_110823.tsv", sep = "\t")
    # additional_label_df["label_id"] = additional_label_df["NMID"] + ":" + additional_label_df["transcript_coordinate"].astype(str)
    # additional_label_df = additional_label_df[["label_id", "SAC", "GLORI", "MICLIP2", "M6ACE"]]
    # additional_label_df[["SAC", "GLORI", "MICLIP2", "M6ACE"]] = additional_label_df[["SAC", "GLORI", "MICLIP2", "M6ACE"]].astype(int)
    # print(additional_label_df)
    #
    # label_df = label_df.merge(additional_label_df, how = "left", on = "label_id")
    # print(label_df)
    #
    # data_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup.pkl")
    # data_df = data_df.merge(label_df, how = "right", on = "label_id")
    # print(data_df)
    #
    # dorado_pred = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.pkl")
    # dorado_pred.rename(columns = {"pred_dorado": "pred_dorado_070", "count_dorado": "count_dorado_070"}, inplace = True)
    # dorado_pred = dorado_pred.merge(label_df[["label_id"]], on = "label_id", how = "right")
    # print(dorado_pred)
    #
    # data_df = data_df.merge(dorado_pred, how = "outer", on = "label_id").fillna(0)
    # data_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup_label.pkl")
    #
    data_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup_label.pkl")
    data_df = data_df[(data_df["count_pm6a"] >= 20) & (data_df["dorado_count"] >= 20)]
    data_df = data_df[data_df["label"]>=-1]
    data_df["GLORI"] = data_df["dom_label"] >= 0.1
    data_df = data_df[["SAC", "GLORI", "MICLIP2", "M6ACE", "pred_dorado_070", "pm6a", "drach"]]
    label_dict = comb_label(data_df)
    label_df = pd.DataFrame(label_dict)

    label_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup_label_comb.pkl")

    # data_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pileup_label_comb.pkl")
    # label_list = [x for x in data_df.columns if x.startswith("LAB")]

    result_dict = {"label_name":[],
                   "pr_auc_dorado_drach":[],
                   "pr_auc_airna_drach":[],
                   "pr_auc_dorado_non_drach":[],
                   "pr_auc_airna_non_drach":[]}

    for label_name in label_df.columns:
        data_df_label = data_df[["pred_dorado_070", "pm6a", "drach"]]
        data_df_label[label_name] = label_df[label_name]
        data_df_label = data_df_label[data_df_label[label_name] >= 0]
        data_df_label_drach = data_df_label[data_df_label["drach"]]
        data_df_label_non_drach = data_df_label[~data_df_label["drach"]]
        precision, recall, _ = precision_recall_curve(data_df_label_drach[label_name], data_df_label_drach["pred_dorado_070"])
        pr_auc_dorado_drach = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(data_df_label_drach[label_name], data_df_label_drach["pm6a"])
        pr_auc_airna_drach = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(data_df_label_non_drach[label_name], data_df_label_non_drach["pred_dorado_070"])
        pr_auc_dorado_non_drach = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(data_df_label_non_drach[label_name], data_df_label_non_drach["pm6a"])
        pr_auc_airna_non_drach = auc(recall, precision)

        result_dict["label_name"].append(label_name)
        result_dict["pr_auc_dorado_drach"].append(pr_auc_dorado_drach)
        result_dict["pr_auc_airna_drach"].append(pr_auc_airna_drach)
        result_dict["pr_auc_dorado_non_drach"].append(pr_auc_dorado_non_drach)
        result_dict["pr_auc_airna_non_drach"].append(pr_auc_airna_non_drach)

    result_df = pd.DataFrame(result_dict)
    print(result_df)
    result_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/inference/inference/AIRNA-DW-v2-20240827-175235-22-373000-token_normalise_dwell_all_npz-ON0090_allmotif_pileup/pr_auc.tsv", sep = "\t", index = False)

    return None

if __name__ == "__main__":
    main()




