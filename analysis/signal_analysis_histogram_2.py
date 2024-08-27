import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
from utils.utils import printmessage


def plot_hist(ca_df, m6a_df, feature, out_path, position):
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(2,2,figsize=(40,20))
    motif_group_list = ["CAA","CAG","UAA","UAG"]

    for i, motif in enumerate(motif_group_list):
        ax = axes[i//2, i%2]

        ca_values = np.stack(ca_df[ca_df["motif_group"]==motif][feature].values, axis = 0)
        m6a_values = np.stack(m6a_df[m6a_df["motif_group"]==motif][feature].values, axis = 0)

        assert ca_values.shape[1] == m6a_values.shape[1]
        ca_n_samples = ca_values.shape[0]
        m6a_n_samples = m6a_values.shape[0]

        ca_values = ca_values[:,position]
        m6a_values = m6a_values[:,position]

        print(ca_values.shape, m6a_values.shape)

        if "log10" in feature:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
        else:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)

        ax.legend()
        ax.set_title(f"{feature} {motif}")

    plt.savefig(f"{out_path}/histogram_{feature}_pos{position}_YAR_motif.png", dpi=300)
    plt.close()
    return None


def plot_hist_2(ca_df, m6a_df, feature, out_path, position):
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(2,2,figsize=(40,20))
    motif_group_list = ["CAA","CAG","UAA","UAG"]

    for i, motif in enumerate(motif_group_list):
        ax = axes[i//2, i%2]

        ca_values = np.stack(ca_df[ca_df["motif_group"]==motif][feature].values, axis = 0)
        m6a_values = np.stack(m6a_df[m6a_df["motif_group"]==motif][feature].values, axis = 0)

        assert ca_values.shape[1] == m6a_values.shape[1]
        ca_n_samples = ca_values.shape[0]
        m6a_n_samples = m6a_values.shape[0]

        ca_values = ca_values[:,position-1:position+2].max(axis=1)
        m6a_values = m6a_values[:,position-1:position+2].max(axis=1)

        if "log10" in feature:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
        else:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)

        ax.legend()
        ax.set_title(f"{feature} {motif}")

    plt.savefig(f"{out_path}/histogram_{feature}_pos{position-1}-{position+1}_max_YAR_motif.png", dpi=300)
    plt.close()
    return None


def plot_hist_3(ca_df, m6a_df, feature, out_path, position):
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(2,2,figsize=(40,20))
    motif_group_list = ["CAA","CAG","UAA","UAG"]

    for i, motif in enumerate(motif_group_list):
        ax = axes[i//2, i%2]

        ca_values = np.stack(ca_df[ca_df["motif_group"]==motif][feature].values, axis = 0)
        m6a_values = np.stack(m6a_df[m6a_df["motif_group"]==motif][feature].values, axis = 0)

        assert ca_values.shape[1] == m6a_values.shape[1]
        ca_n_samples = ca_values.shape[0]
        m6a_n_samples = m6a_values.shape[0]

        ca_values = ca_values[:,position-1:position+2].mean(axis=1)
        m6a_values = m6a_values[:,position-1:position+2].mean(axis=1)

        if "log10" in feature:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
        else:
            sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)
            sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density",
                         binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)

        ax.legend()
        ax.set_title(f"{feature} {motif}")

    plt.savefig(f"{out_path}/histogram_{feature}_pos{position-1}-{position+1}_mean_YAR_motif.png", dpi=300)
    plt.close()
    return None


def motif_to_group(motif):
    return motif[1:4]

# def motif_to_group(motif):
#     front = motif[1]
#     rear = motif[3]
#
#     purine = ["A","G"]
#     pyrimidine = ["C","U"]
#
#     if front in purine and rear in purine:
#         return "RAR"
#     elif front in purine and rear in pyrimidine:
#         return "RAY"
#     elif front in pyrimidine and rear in purine:
#         return "YAR"
#     elif front in pyrimidine and rear in pyrimidine:
#         return "YAY"
#     else:
#         raise ValueError("Invalid motif")
#



def main():

    feature_list = ["signal_len_log10", "signal_len"]
    cb_list = ["_CB0","_CB1","_CB2"]
    ca_df_list = []
    m6a_df_list = []

    for i in range(3):
        cb_idx = cb_list[i]
        ca_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")
        m6a_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")

        ca_df["motif_group"] = ca_df["5mer"].apply(lambda x: motif_to_group(x))
        m6a_df["motif_group"] = m6a_df["5mer"].apply(lambda x: motif_to_group(x))

        ca_df["signal_len"] = ca_df["signal_len"].apply(lambda x: np.float32(x))
        m6a_df["signal_len"] = m6a_df["signal_len"].apply(lambda x: np.float32(x))
        ca_df["signal_len_log10"] = ca_df["signal_len"].apply(lambda x: np.log10(x))
        m6a_df["signal_len_log10"] = m6a_df["signal_len"].apply(lambda x: np.log10(x))

        ca_df_list.append(ca_df)
        m6a_df_list.append(m6a_df)

        printmessage(f"Loaded {cb_idx}, length: {len(ca_df)}, {len(m6a_df)}")

    ca_df = pd.concat(ca_df_list, axis=0).reset_index(drop=True)
    m6a_df = pd.concat(m6a_df_list, axis=0).reset_index(drop=True)

    printmessage("Loaded all")

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis"
    os.makedirs(out_path, exist_ok = True)
    position = 23

    for feature in feature_list:
        plot_hist(ca_df, m6a_df, feature, out_path, position)
        plot_hist_2(ca_df, m6a_df, feature, out_path, position)
        plot_hist_3(ca_df, m6a_df, feature, out_path, position)
        printmessage(f"Plotted {feature}")

    return None

if __name__ == "__main__":
    main()
