import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm


def standardise_trim_segment_signal(signal,move,sp,ts,ns,offset,scale):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)
    signal = (signal + offset) * scale

    move = np.array(move, dtype=int)
    stride = move[0]
    move = move[1:]
    move_idx = len(signal) - (np.nonzero(move)[0][1:] * stride)
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) < 5:
        return None
    signal = np.array([np.mean(x) for x in signal])
    return signal


def extract_kmer_signal(path, sample):

    s_paths = f"{path}/signal_raw/*.pkl"
    s_paths = glob.glob(s_paths)
    s_paths = np.random.choice(s_paths, sample)
    b_paths = [x.replace("signal_raw", "move_df_split") for x in s_paths]

    df_list = []

    for s_path, b_path in tqdm(zip(s_paths, b_paths), desc="Extracting kmer signal", total=len(s_paths)):
        try:
            data = pd.read_pickle(s_path)
            data2 = pd.read_pickle(b_path)
        except:
            continue
        data = data.merge(data2, on="read_id", how="inner")
        data["signal"] = data.apply(lambda x: standardise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"], x["offset"], x["scale"]), axis=1)
        data["signal"] = data["signal"].apply(lambda x: x[2:-2])
        data["5mer"] = data["seq"].apply(lambda x:[x[i:i+5] for i in range(0, len(x)-4)])
        data["len_signal"] = data["signal"].apply(lambda x: len(x))
        data["len_5mer"] = data["5mer"].apply(lambda x: len(x))
        data = data[data["len_signal"] == data["len_5mer"]]
        data = data[["signal","5mer"]].explode(["signal","5mer"])
        df_list.append(data)

    df = pd.concat(df_list, axis=0)
    print(df)
    return df


def main():
    cell = extract_kmer_signal("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/intermediates", 10)
    ivt = extract_kmer_signal("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/intermediates", 10)
    cell["class"] = "CELL"
    ivt["class"] = "IVT"
    data = pd.concat([cell, ivt], axis=0)
    data.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/plot/kmer_signal.pkl")
    ## Plot distribution for each 5mer. There can be 4^5 = 1024 5mers, so we will plot only 30 5mers with the highest counts.
    selected_5mers = data["5mer"].value_counts().index[1:31]
    data = data[data["5mer"].isin(selected_5mers)]
    fig, axes = plt.subplots(figsize=(40, 20), nrows=5, ncols=6)
    for i, (name, group) in enumerate(data.groupby("5mer")):
        ax = axes[i//6, i%6]
        ivt_group = group[group["class"] == "IVT"]
        cell_group = group[group["class"] == "CELL"]
        sns.kdeplot(ivt_group["signal"], ax=ax, label="IVT")
        sns.kdeplot(cell_group["signal"], ax=ax, label="CELL")
        ax.set_title(f"{name} (n={len(ivt_group)};{len(cell_group)})")
        ax.legend()
        ax.set_xlim(40,140)
    fig.tight_layout()
    plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/kmer_signal.png")
    plt.close(fig)

    return None


if __name__ == "__main__":
    main2()