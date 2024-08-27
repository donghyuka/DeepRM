import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


def plot(mean_dict, ci95_dict, length_dict, title, out_path):

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))

    colour_list = ['royalblue', 'tomato', 'skyblue', 'lightsalmon', 'orange']

    for i, key in enumerate(mean_dict.keys()):
        colour = colour_list[i]
        mean = mean_dict[key]
        ci95 = ci95_dict[key]
        length = length_dict[key]
        width = len(mean)
        ax.plot(mean, color=colour, label=f"{key} (n={length:,})")
        ax.fill_between(np.arange(width), mean-ci95, mean+ci95,
                        color=colour, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel(title)

    ax.set_xticks(np.arange(0, width, 3))
    ax.set_xticklabels(np.arange(0, width, 3))


    ## Vline at center
    ax.axvline(x=width//2, color="black", linestyle="--")

    ax.legend()
    plt.savefig(f"{out_path}/{title}_runs.png", dpi=300)
    plt.close()

    return

def main():

    feature_list = ["signal_mean","signal_median","signal_std","signal_len_log10","signal_amp","signal_rms","bq"]

    ylim_dict = {}
    ON0092_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/result.pkl")
    ON0093_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/result.pkl")
    ON0096_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/result.pkl")
    ON0098_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/result.pkl")
    ON0099_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/result.pkl")

    df_dict = {"ON0092": ON0092_df, "ON0093": ON0093_df, "ON0096": ON0096_df, "ON0098": ON0098_df, "ON0099": ON0099_df}

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis"
    os.makedirs(out_path, exist_ok = True)
    for feature in feature_list:
        feature_renamed = feature.replace("signal_", f"normalised_signal_")
        mean_dict = {df_name: df[feature]["mean"] for df_name, df in df_dict.items()}
        ci95_dict = {df_name: df[feature]["ci95"] for df_name, df in df_dict.items()}
        length_dict = {df_name: df[feature]["len"] for df_name, df in df_dict.items()}

        plot(mean_dict, ci95_dict, length_dict, feature_renamed, out_path)

    return None

def main():

    feature_list = ["signal_mean","signal_median","signal_std","signal_len_log10","signal_amp","signal_rms","bq"]

    ylim_dict = {}
    ON0092_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/result.pkl")
    ON0093_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/result.pkl")
    ON0096_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/result.pkl")
    ON0098_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/result.pkl")
    ON0099_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/result.pkl")

    df_dict = {"ON0092": ON0092_df, "ON0093": ON0093_df, "ON0096": ON0096_df, "ON0098": ON0098_df, "ON0099": ON0099_df}
 

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis"
    os.makedirs(out_path, exist_ok = True)
    for feature in feature_list:
        feature_renamed = feature.replace("signal_", f"unnormalised_signal_")
        mean_dict = {df_name: df[feature]["mean"] for df_name, df in df_dict.items()}
        ci95_dict = {df_name: df[feature]["ci95"] for df_name, df in df_dict.items()}
        length_dict = {df_name: df[feature]["len"] for df_name, df in df_dict.items()}

        plot(mean_dict, ci95_dict, length_dict, feature_renamed, out_path)

    return None

if __name__ == "__main__":
    main()
