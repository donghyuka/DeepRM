import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm
import argparse



def normalise_trim_segment_signal(signal,move,sp,ts,ns, quantile_a, quantile_b, shift_mult, scale_mult):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)

    quantile_a_value = np.quantile(signal, quantile_a)
    quantile_b_value = np.quantile(signal, quantile_b)

    q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value))
    signal = (signal - q_shift) / q_scale

    move = np.array(move, dtype=int)
    stride = move[0]
    move = move[1:]
    move_idx = len(signal) - (np.nonzero(move)[0][1:] * stride)
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) < 5:
        return None
    return signal


def extract_kmer_signal(path, bam_name="debug_actb", sample = 0, trim = 0):

    b_paths = f"{path}/{bam_name}/*.pkl"
    b_paths = glob.glob(b_paths)
    if sample:
        b_paths = np.random.choice(b_paths, sample)
    s_paths = [x.replace(bam_name, "signal_raw") for x in b_paths]


    df_list = []

    for s_path, b_path in tqdm(zip(s_paths, b_paths), desc="Extracting kmer signal", total=len(s_paths)):
        try:
            data = pd.read_pickle(s_path)
            data2 = pd.read_pickle(b_path)
        except:
            continue

        data = data.merge(data2, on="read_id", how="inner")
        if trim:
            data["signal"] = data["signal"].apply(lambda x: x[trim:])

        df_list.append(data)

    df = pd.concat(df_list, axis=0)
    return df



def get_left_soft_clip(cigar):
    if "S" in cigar:
        left_soft_clip = cigar.split("S")[0]
        if left_soft_clip.isdigit():
            left_soft_clip = int(left_soft_clip)
        else:
            left_soft_clip = 0

    else:
        left_soft_clip = 0
    return left_soft_clip

def main( sample = 100, trim = 0):
    args = argparse.ArgumentParser()
    args.add_argument("--key", type=str, required=True)
    args = args.parse_args()
    key = args.key

    ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/intermediates"
    ivt = extract_kmer_signal(ivt_path, bam_name=key, sample = sample, trim = trim)
    print(ivt)
    cell_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/intermediates"
    cell = extract_kmer_signal(cell_path, bam_name="move_df_split", sample = sample)
    print(cell)

    cell["class"] = "CELL"
    ivt["class"] = "IVT"
    data = pd.concat([cell, ivt], axis=0)

    print(data)

    quant_a = 0.2
    quant_b = 0.8
    shift_mult = 0.48
    scale_mult = 0.59


    data["signal"] = data.apply(lambda x: normalise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                        quant_a, quant_b, shift_mult, scale_mult), axis=1)
    data = data.dropna(subset=["signal"], axis=0)
    data["signal"] = data["signal"].apply(lambda x: x[2:-2])
    data["5mer"] = data["seq"].apply(lambda x:[x[i:i+5] for i in range(0, len(x)-4)])
    data["len_signal"] = data["signal"].apply(lambda x: len(x))
    data["len_5mer"] = data["5mer"].apply(lambda x: len(x))
    data = data[data["len_signal"] == data["len_5mer"]]

    data = data[["signal","5mer", "class"]].copy().explode(["signal","5mer"])
    data["signal"] = data["signal"].apply(np.mean)
    print(data)

    data.to_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/plot/{key}.pkl")


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
        ax.set_xlim(-3,3)
    fig.tight_layout()
    plt.savefig(f"/extdata4/baeklab/Hyeonseo/m6A/plot/{key}_normalised.png")
    plt.close(fig)

    return None


if __name__ == "__main__":
    main(sample=100, trim=0)
