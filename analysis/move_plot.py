import pandas as pd
import glob
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


path = "/extdata4/baeklab/Hyeonseo/m6A/inference/attention/BERMUDA-Proto-v27-20240405-133915-13-108000-baeklab_v2_depth20_sampled_balanced_drach/"
label_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GP3.depth20.twm6astrict.sampled.drach.balanced.tsv"


def load():
    files = glob.glob(f"{path}/inference/*.pkl")

    df_list = []
    for file in tqdm.tqdm(files):
        df_list.append(pd.read_pickle(file)[["label_id", "label", "prediction", "move"]])

    df = pd.concat(df_list)

    df2 = df[(df["label"] == 1) & (df["prediction"] > 0.95)]
    print(df2[["label", "prediction"]])
    df2.to_pickle(f"{path}/true_positive_move.pkl")

    df2 = df[(df["label"] == 0) & (df["prediction"] > 0.95)]
    print(df2[["label", "prediction"]])
    df2.to_pickle(f"{path}/false_positive_move.pkl")

    df2 = df[(df["label"] == 1) & (df["prediction"] < 0.05)]
    print(df2[["label", "prediction"]])
    df2.to_pickle(f"{path}/false_negative_move.pkl")

    df2 = df[(df["label"] == 0) & (df["prediction"] < 0.05)]
    print(df2[["label", "prediction"]])
    df2.to_pickle(f"{path}/true_negative_move.pkl")
    return None

def plot():
    # label_df = pd.read_csv(label_path, sep = "\t")
    # label_df = label_df[["id", "m6A_level"]]
    #
    # tp_df  = pd.read_pickle(f"{path}/true_positive_move.pkl")
    # tp_df = tp_df.merge(label_df, left_on = "label_id", right_on = "id", how = "inner")
    # tp_df = tp_df[tp_df["m6A_level"] > 0.95]
    # tp_df["group"] = "True Positive"
    # print(tp_df[["label", "prediction"]])
    #
    # fp_df  = pd.read_pickle(f"{path}/false_positive_move.pkl")
    # fp_df["group"] = "False Positive"
    # print(fp_df[["label", "prediction"]])
    #
    # tn_df  = pd.read_pickle(f"{path}/true_negative_move.pkl")
    # tn_df["group"] = "True Negative"
    # print(tn_df[["label", "prediction"]])
    #
    # fn_df  = pd.read_pickle(f"{path}/false_negative_move.pkl")
    # fn_df = fn_df.merge(label_df, left_on = "label_id", right_on = "id", how = "inner")
    # fn_df = fn_df[fn_df["m6A_level"] > 0.95]
    # fn_df["group"] = "False Negative"
    # print(fn_df[["label", "prediction"]])
    #
    # cat_df = pd.concat([tp_df, fp_df, tn_df, fn_df], ignore_index = True)
    # cat_df.to_pickle(f"{path}/move_df.pkl")

    cat_df = pd.read_pickle(f"{path}/move_df.pkl")

    print(cat_df)

    move_hist(cat_df)

    return None


def move_hist(df):
    def get_centre_width(arr, group):
        arr = arr.tolist()
        start = arr.index(9)
        try:
            end = arr.index(10)
        except:
            end = arr[::-1].index(9)
            print(group)
        return end - start

    df["centre_width"] = df.apply(lambda x: get_centre_width(x["move"], x["group"]), axis = 1)
    df["centre_width"] = df["centre_width"].apply(lambda x: 0 if x < 0 else x)

    fig, axes = plt.subplots(figsize = (20,20), nrows = 4, ncols = 1)
    bins = np.arange(0, 20, 1)
    for group, ax in zip(["True Positive", "False Positive", "True Negative", "False Negative"], axes):
        sns.histplot(df[df["group"] == group]["centre_width"], kde = True, label = group, ax = ax,
                     alpha = 0.5, bins = bins, stat = "density")
        ax.set_xlim(0, 20)
        ax.set_title(group)
    plt.savefig(f"{path}/move_hist.png")
    return None

if __name__ == "__main__":
    # load()
    plot()