import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns


def plot(df_dict, title, out_path):

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))

    colour_list = ['slategrey', 'goldenrod', 'silver', 'gold', 'crimson']
    values_list = []
    group_list = []
    group_list_unique = []

    # for i, key in enumerate(df_dict.keys()):
    #     values = df_dict[key][title].values
    #     groupname = f"{key} (n={len(values):,})"
    #     values_list.append(values)
    #     group_list.append(np.repeat(groupname, len(values)))
    #     group_list_unique.append(groupname)
    #
    # values_list = np.concatenate(values_list)
    # group_list = np.concatenate(group_list)
    # palette = dict(zip(group_list_unique, colour_list))
    #
    # df = pd.DataFrame({title: values_list, "group": group_list})
    #
    # df.to_pickle(f"{out_path}/{title}_runs.pkl")

    df = pd.read_pickle(f"{out_path}/{title}_runs.pkl")
    palette = dict(zip(df["group"].unique(), colour_list))

    sns.histplot(data=df, x=title, hue="group", palette=palette, ax=ax, kde=False, stat="density",fill=False,
                 linewidth = 3, element="step", legend=True, binwidth=5, alpha = 0.5)

    ax.set_title(title)
    ax.set_xlabel(f"{title} (nt)")
    ax.set_ylabel("Density")

    plt.savefig(f"{out_path}/{title}_runs.png", dpi=300)
    plt.close()

    return

def main():

    feature_list = ["5p_dist","3p_dist"]

    ON0092_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/distance_analysis/result.pkl").sample(1000000)
    ON0093_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/distance_analysis/result.pkl").sample(1000000)
    ON0096_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/distance_analysis/result.pkl").       sample(1000000)
    ON0098_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/distance_analysis/result.pkl").       sample(1000000)
    ON0099_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/distance_analysis/result.pkl").       sample(1000000)

    df_dict = {"ON0092": ON0092_df, "ON0093": ON0093_df, "ON0096": ON0096_df, "ON0098": ON0098_df, "ON0099": ON0099_df}


    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis"
    os.makedirs(out_path, exist_ok = True)
    for feature in feature_list:
        plot(df_dict, feature, out_path)

    return None

if __name__ == "__main__":
    main()
