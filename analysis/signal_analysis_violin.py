import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
from utils.utils import printmessage


def plot_violin(ca_values, m6a_values, feature, out_path, ylims=None):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))

    assert ca_values.shape[1] == m6a_values.shape[1]
    data_width = ca_values.shape[1]
    ca_n_samples = ca_values.shape[0]
    m6a_n_samples = m6a_values.shape[0]

    ca_data_flat = ca_values.flatten()
    m6a_data_flat = m6a_values.flatten()
    ca_data_index = np.tile(np.arange(data_width), ca_n_samples)
    m6a_data_index = np.tile(np.arange(data_width), m6a_n_samples)
    ca_str = f"cA (n={ca_n_samples:,})"
    m6a_str = f"m6A (n={m6a_n_samples:,})"

    ca_data_class = np.repeat(ca_str, ca_n_samples * data_width)
    m6a_data_class = np.repeat(m6a_str, m6a_n_samples * data_width)

    data_flat = np.concatenate([ca_data_flat, m6a_data_flat])
    data_index = np.concatenate([ca_data_index, m6a_data_index])
    data_class = np.concatenate([ca_data_class, m6a_data_class])

    data_df = pd.DataFrame({"value": data_flat, "position": data_index, "class": data_class})

    sns.violinplot(x="position", y="value", hue="class", data=data_df, ax=ax, split=True, inner="quartile",
                     palette={ca_str: "royalblue", m6a_str: "tomato"}, fill=True, linewidth=1)

    ax.set_title(feature)
    ax.set_ylabel(feature)

    if ylims is not None:
        ax.set_ylim(ylims)

    ax.legend()
    plt.savefig(f"{out_path}/violin_{feature}.png", dpi=300)
    plt.close()
    return None

def main():

    feature_list = ["signal_mean","signal_len_log10","signal_len"]
    cb_list = ["_CB0","_CB1","_CB2","_ALL"]
    ca_data_dict = {feature: [] for feature in feature_list}
    m6a_data_dict = {feature: [] for feature in feature_list}
    ylim_dict_list = []

    for i in range(3):
        cb_idx = cb_list[i]
        ylim_dict = {}
        ca_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")
        m6a_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled{cb_idx}.pkl")

        ca_df["signal_len_log10"] = ca_df["signal_len"].apply(lambda x: np.log10(np.float32(x)))
        m6a_df["signal_len_log10"] = m6a_df["signal_len"].apply(lambda x: np.log10(np.float32(x)))

        for feature in feature_list:
            ca_values = np.stack(ca_df[feature].values, axis = 0)
            m6a_values = np.stack(m6a_df[feature].values, axis = 0)
            ca_data_dict[feature].append(ca_values)
            m6a_data_dict[feature].append(m6a_values)

            ymin = min(np.min(ca_values), np.min(m6a_values))
            ymax = max(np.max(ca_values), np.max(m6a_values))

            ylim_dict[feature] = (ymin, ymax)
        ylim_dict_list.append(ylim_dict)

        printmessage(f"Loaded {cb_idx}")

    ylim_dict = {}
    for feature in feature_list:
        mins = []
        maxs = []
        for ylim_dict_cb in ylim_dict_list:
            ymin, ymax = ylim_dict_cb[feature]
            mins.append(ymin)
            maxs.append(ymax)
        ymin = min(mins)
        ymax = max(maxs)
        ylim_dict[feature] = (ymin, ymax)

        ca_data_all = np.concatenate(ca_data_dict[feature], axis = 0)
        m6a_data_all = np.concatenate(m6a_data_dict[feature], axis = 0)
        ca_data_dict[feature].append(ca_data_all)
        m6a_data_dict[feature].append(m6a_data_all)

    printmessage("Loaded all")

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis"
    os.makedirs(out_path, exist_ok = True)

    for i in [3, 0, 1, 2]:
        cb_idx = cb_list[i]
        for feature in feature_list:
            ca_values = ca_data_dict[feature][i]
            m6a_values = m6a_data_dict[feature][i]

            feature_renamed = feature.replace("signal_", f"normalised_signal_") + cb_idx
            ylims = ylim_dict[feature]
            plot_violin(ca_values, m6a_values, feature_renamed, out_path, ylims)

            printmessage(f"Plotted {feature_renamed} for {cb_idx}")

    return None

if __name__ == "__main__":
    main()
