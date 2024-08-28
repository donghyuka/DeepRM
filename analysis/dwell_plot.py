import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import glob


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
    pos_path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver082724_npz/score-perfect/train/pos"
    neg_path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver082724_npz/score-perfect/train/neg"

    pos_files = glob.glob(f"{pos_path}/*.npz")
    neg_files = glob.glob(f"{neg_path}/*.npz")

    sample_size = 1000

    pos_files = np.random.choice(pos_files, sample_size)
    neg_files = np.random.choice(neg_files, sample_size)

    pos_data = np.concatenate([np.load(x)["dwell_token"] for x in pos_files])
    neg_data = np.concatenate([np.load(x)["dwell_token"] for x in neg_files])

    pos_mean = np.mean(pos_data, axis=0)
    pos_ci95 = 1.96 * np.std(pos_data, axis=0) / np.sqrt(len(pos_data))
    pos_len = len(pos_data)

    neg_mean = np.mean(neg_data, axis=0)
    neg_ci95 = 1.96 * np.std(neg_data, axis=0) / np.sqrt(len(neg_data))
    neg_len = len(neg_data)

    mean_dict = {"pos": pos_mean, "neg": neg_mean}
    ci95_dict = {"pos": pos_ci95, "neg": neg_ci95}
    length_dict = {"pos": pos_len, "neg": neg_len}

    plot(mean_dict, ci95_dict, length_dict,
         "normalised_log_dwell", "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis")

    return None


if __name__ == "__main__":
    main()

