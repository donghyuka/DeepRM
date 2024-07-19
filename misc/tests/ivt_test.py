# import numpy as np
#
# path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block_v2/intermediates/signal_raw/4-9-25.pkl"
# import pandas as pd
# from matplotlib import pyplot as plt
#
# data = pd.read_pickle(path)
# print(data)
# data = data.sample(20)
#
# fig, axes = plt.subplots(figsize=(20, 20), nrows=5, ncols=4)
# for i, (idx, row) in enumerate(data.iterrows()):
#     axes[i//4, i%4].plot(row["signal"])
#     axes[i//4, i%4].set_title(row["read_id"])
# plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/ivt.png")
#

from tqdm import tqdm
import numpy as np
import pysam
import matplotlib.pyplot as plt

def remove_polya(seq):
    ## Remove polyA tail from the sequence
    seq = seq[::-1]
    i = 0
    for i, s in enumerate(seq):
        if s != "A":
            break
    seq = seq[i:]
    return seq[::-1]

bam_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.bam"

result_list = []


def draw_histogram(hist_arr, out_path):
    bin_edges = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(len(hist_arr)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        ax.fill_between([bin_start, bin_end], hist_arr[i], color = "blue", alpha = 0.5)
    ax.set_title("Prediction Histogram")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Frequency")
    plt.savefig(f"{out_path}/ivt_histogram.png")
    return None

import pandas as pd
from utils.utils import is_drach

# df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/modkit_extract.tsv"
# df = pd.read_csv(df_path, sep="\t", header=0)
# df.to_pickle(df_path.replace(".tsv", ".pkl"))
#
# df["drach"] = df["ref_kmer"].apply(is_drach)
# df = df[df["drach"]].copy()
# df.to_pickle(df_path.replace(".tsv", "_drach.pkl"))

df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/modkit_extract_drach.pkl")
print(df)
score_arr = df["mod_qual"].values
hist, bin_edges = np.histogram(score_arr, bins = 100, range = (0, 1))
draw_histogram(hist, "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/")