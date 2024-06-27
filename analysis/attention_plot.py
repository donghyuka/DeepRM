import pandas as pd
import glob
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


path = "/extdata4/baeklab/Hyeonseo/m6A/inference/attention/BERMUDA-Basecaller-v2-20240420-222045-4-52000-baeklab_v2_depth20_sampled_balanced_drach"
label_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GP3.depth20.twm6astrict.sampled.drach.balanced.tsv"

def plot_attention(df, comment, minmax_dict):
    os.makedirs(f"{path}/plot/{comment}", exist_ok=True)


    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        move_list = row["move"].tolist()
        centre_start = move_list.index(9)
        centre_end = move_list.index(10)
        plt.rcParams.update({'font.size': 32})
        fig, axes = plt.subplots(figsize = (80, 60), nrows = 3, ncols = 4)
        for i in range(10):
            ax = axes[i // 4, i % 4]
            title = f"Self-Attention Layer {i}"

            sns.heatmap(row[f"attention_{i}"], ax = ax, cmap = "rocket", vmin = minmax_dict[i][0], vmax= minmax_dict[i][1],
                        cbar = True, square = True)

            ## draw white square for centre start and end
            ax.add_patch(plt.Rectangle((centre_start, centre_start), centre_end - centre_start, centre_end - centre_start,
                                       fill = False, edgecolor = "chartreuse", linewidth = 5))

            ax.set_title(title)
        # plt.tight_layout()
        fig.suptitle(f"ID: {row['label_id']}, Prediction: {row['prediction']}, Label: {row['label']}")
        plt.savefig(f"{path}/plot/{comment}/{row['block_id']}.png")
        plt.close()

    return None

def load():
    files = glob.glob(f"{path}/inference/*.pkl")
    id_list = glob.glob(("/extdata4/baeklab/Hyeonseo/m6A/inference/attention/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v2_depth20_sampled_balanced_drach/plot/true_positive/*.png"))
    id_list = [x.split("/")[-1].split(".")[0] for x in id_list]


    df_list = []
    for file in tqdm.tqdm(files):
        df_list.append(pd.read_pickle(file))

    df = pd.concat(df_list)

    # df2 = df[(df["label"] == 1) & (df["prediction"] > 0.99)]
    # print(df2[["label", "prediction"]])
    # df2.to_pickle(f"{path}/true_positive.pkl")
    #
    # df2 = df[(df["label"] == 0) & (df["prediction"] > 0.99)]
    # print(df2[["label", "prediction"]])
    # df2.to_pickle(f"{path}/false_positive.pkl")
    #
    # df2 = df[(df["label"] == 1) & (df["prediction"] < 0.01)]
    # print(df2[["label", "prediction"]])
    # df2.to_pickle(f"{path}/false_negative.pkl")
    #
    # df2 = df[(df["label"] == 0) & (df["prediction"] < 0.01)]
    # print(df2[["label", "prediction"]])
    # df2.to_pickle(f"{path}/true_negative.pkl")

    df2 = df[df["block_id"].isin(id_list)]
    df2.to_pickle(f"{path}/compare_baseline.pkl")

    return None

def plot():
    label_df = pd.read_csv(label_path, sep = "\t")
    label_df = label_df[["id", "m6A_level"]]

    # tp_df  = pd.read_pickle(f"{path}/true_positive.pkl")
    # tp_df = tp_df.merge(label_df, left_on = "label_id", right_on = "id", how = "inner")
    # tp_df = tp_df[tp_df["m6A_level"] > 0.9]
    # print(tp_df[["label", "prediction"]])
    # tp_df = tp_df.sort_values("prediction", ascending = False)[:20]
    #
    # fp_df  = pd.read_pickle(f"{path}/false_positive.pkl")
    # print(fp_df[["label", "prediction"]])
    # fp_df = fp_df.sort_values("prediction", ascending = False)[:20]
    #
    # tn_df  = pd.read_pickle(f"{path}/true_negative.pkl")
    # print(tn_df[["label", "prediction"]])
    # tn_df = tn_df.sort_values("prediction", ascending = True)[:20]
    #
    # fn_df  = pd.read_pickle(f"{path}/false_negative.pkl")
    # fn_df = fn_df.merge(label_df, left_on = "label_id", right_on = "id", how = "inner")
    # fn_df = fn_df[fn_df["m6A_level"] > 0.9]
    # print(fn_df[["label", "prediction"]])
    # fn_df = fn_df.sort_values("prediction", ascending = True)[:20]
    #
    # cat_df = pd.concat([tp_df, fp_df, tn_df, fn_df], ignore_index = True)
    #
    # minmax_dict = {}
    # for i in range(6):
    #     attn_array = np.concatenate(cat_df[f"attention_{i}"].values)
    #     percentile_99 = np.percentile(attn_array, 99.9)
    #     percentile_1 = np.percentile(attn_array, 0.1)
    #     minmax_dict[i] = (percentile_1, percentile_99)
    #
    # plot_attention(tp_df, "true_positive", minmax_dict)
    # plot_attention(fp_df, "false_positive", minmax_dict)
    # plot_attention(tn_df, "true_negative", minmax_dict)
    # plot_attention(fn_df, "false_negative", minmax_dict)

    compare_df = pd.read_pickle(f"{path}/compare_baseline.pkl")
    compare_df = compare_df.merge(label_df, left_on = "label_id", right_on = "id", how = "inner")
    minmax_dict = {}
    for i in range(10):
        attn_array = np.concatenate(compare_df[f"attention_{i}"].values)
        percentile_99 = np.percentile(attn_array, 99.9)
        percentile_1 = np.percentile(attn_array, 0.1)
        minmax_dict[i] = (percentile_1, percentile_99)
    plot_attention(compare_df, "compare_baseline", minmax_dict)

    return None

if __name__ == "__main__":
    load()
    plot()