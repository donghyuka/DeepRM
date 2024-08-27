import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
from utils.utils import printmessage


def plot_hist(ca_values, m6a_values, feature, out_path, position):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))

    assert ca_values.shape[1] == m6a_values.shape[1]
    data_width = ca_values.shape[1]
    ca_n_samples = ca_values.shape[0]
    m6a_n_samples = m6a_values.shape[0]

    ca_values = ca_values[:,position]
    m6a_values = m6a_values[:,position]

    if "log10" in feature:
        sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density", binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
        sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density", binrange=(0, 4), binwidth=0.25, fill=False, linewidth = 3, element="step", alpha = 0.5)
    else:
        sns.histplot(ca_values, color="royalblue", label=f"cA (n={ca_n_samples:,})", ax=ax, kde=False, stat="density", binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)
        sns.histplot(m6a_values, color="tomato", label=f"m6A (n={m6a_n_samples:,})", ax=ax, kde=False, stat="density", binrange=(0, 600), binwidth=12, fill=False, linewidth = 3, element="step", alpha = 0.5)

    ax.legend()
    plt.savefig(f"{out_path}/histogram_{feature}.png", dpi=300)
    plt.close()
    return None

def main():

    feature_list = ["signal_len_log10"]
    cb_list = ["_CB0","_CB1","_CB2","_ALL"]
    ca_df_list = []
    m6a_df_list = []
    spacer_list = ["CC","AA","GU"]



    for i in range(3):
        cb_idx = cb_list[i]
        ca_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")
        m6a_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")

        ca_df["signal_len"] = ca_df["signal_len"].apply(lambda x: np.float32(x))
        m6a_df["signal_len"] = m6a_df["signal_len"].apply(lambda x: np.float32(x))
        ca_df["signal_len_log10"] = ca_df["signal_len"].apply(lambda x: np.log10(x))
        m6a_df["signal_len_log10"] = m6a_df["signal_len"].apply(lambda x: np.log10(x))

        ca_df["end_5mer"] = ca_df["motif"].str[-3:] + spacer_list[i]
        m6a_df["end_5mer"] = m6a_df["motif"].str[-3:] + spacer_list[i]

        ca_df_list.append(ca_df)
        m6a_df_list.append(m6a_df)

        printmessage(f"Loaded {cb_idx}")

    ca_df = pd.concat(ca_df_list, axis=0).reset_index(drop=True)
    m6a_df = pd.concat(m6a_df_list, axis=0).reset_index(drop=True)

    printmessage("Loaded all")

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis/end_motif"
    os.makedirs(out_path, exist_ok = True)
    position = 23

    nuc_list = ["A","C","G","U"]


    motif_list = [f"{nuc1}{nuc2}{nuc3}{end}" for nuc1 in nuc_list for nuc2 in nuc_list for nuc3 in nuc_list for end in spacer_list]


    for motif in motif_list:
        for feature in feature_list:

            ca_values = np.stack(ca_df[ca_df["end_5mer"]==motif][feature].values, axis = 0)
            m6a_values = np.stack(m6a_df[ca_df["end_5mer"]==motif][feature].values, axis = 0)
            feature_renamed = feature.replace("signal_", f"normalised_signal_") + f"_pos{position}_end_{motif}"
            plot_hist(ca_values, m6a_values, feature_renamed, out_path, position)
            printmessage(f"Plotted {feature_renamed} for {motif}")

    return None

if __name__ == "__main__":
    main()
