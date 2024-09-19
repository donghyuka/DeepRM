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
    motif_list = [ "AAACU"
                  ,"AGACA"
                  ,"AGACC"
                  ,"AGACU"
                  ,"GAACA"
                  ,"GAACC"
                  ,"GAACU"
                  ,"GGACA"
                  ,"GGACC"
                  ,"GGACU"
                  ,"UAACU"
                  ,"UGACA"
                  ,"UGACU"]

    ylim_dict = {}
    ON0092_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled.pkl")
    ON0093_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled.pkl")
    ON0096_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled.pkl")
    ON0098_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled.pkl")
    ON0099_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/sampled.pkl")
    df_dict = {"ON0092": ON0092_df, "ON0093": ON0093_df, "ON0096": ON0096_df, "ON0098": ON0098_df, "ON0099": ON0099_df}

    # ON0092_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/"
    # ON0093_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/"
    # ON0096_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/"
    # ON0098_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/"
    # ON0099_path = f"/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/signal_analysis/"
    # path_dict = {"ON0092": ON0092_path, "ON0093": ON0093_path, "ON0096": ON0096_path, "ON0098": ON0098_path, "ON0099": ON0099_path}
    #
    # df_dict = {}
    # for key, path in path_dict.items():
    #     print(key)
    #     df = pd.concat([pd.read_pickle(f"{path}/sampled_CB{i}.pkl") for i in range(3)]).reset_index(drop=True)
    #     df.to_pickle(f"{path}/sampled.pkl")
    #     df_dict[key] = df

    for key, df in df_dict.items():
        df["signal_len_log10"] = df["signal_len"].apply(lambda x: np.log10(x))
        for feature in feature_list:
            df[feature] = df[feature].apply(lambda x: np.array(x, dtype=np.float32))

    print(df_dict)

    out_path = "/extdata4/baeklab/Hyeonseo/m6A/plot/feature_analysis_drach"
    os.makedirs(out_path, exist_ok = True)
    for motif in motif_list:
        motif_df_dict = {df_name: df[df["5mer"]==motif] for df_name, df in df_dict.items()}
        print(motif)
        for feature in feature_list:
            feature_renamed = f"{motif}_{feature}"

            data_dict = {df_name: np.stack(df[feature].values) for df_name, df in motif_df_dict.items()}

            mean_dict = {df_name: np.mean(data, axis=0) for df_name, data in data_dict.items()}
            ci95_dict = {df_name: 1.96*np.std(data, axis=0)/np.sqrt(data.shape[0]) for df_name, data in data_dict.items()}
            length_dict = {df_name: data.shape[0] for df_name, data in data_dict.items()}

            plot(mean_dict, ci95_dict, length_dict, feature_renamed, out_path)

    return None

if __name__ == "__main__":
    main()
