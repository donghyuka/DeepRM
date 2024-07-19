## Dynamic Time Warping
## Goal: Warp the original time series to have an equal number of samples for each base (interval).

import numpy as np
import scipy.interpolate
import torch
from dtaidistance import dtw
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

nrows=5
target_length = 30

def get_template(motifs,kmer_dict):
    templates = []
    for m in motifs:
        win_size = 5
        kmers = [m[i:i+win_size] for i in range(0, len(m) - win_size + 1)]
        template = [kmer_dict[k] for k in kmers]
        templates.append(template)
    return templates

kmer_path = "/extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt"
kmer_df = pd.read_csv(kmer_path, sep="\t", header=0, skiprows=6)
kmer_df["level_mean"] = (kmer_df["level_mean"] - 80.876) / 17.270
kmer_dict = dict(zip(kmer_df["kmer"], kmer_df["level_mean"]))


path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/train/pos/000000119.pkl"
data = pd.read_pickle(path).sample(nrows).reset_index(drop=True)
signals = data["signal_token"].values
motifs = data["kmer_token"].values
step_sizes = (np.stack(data["segment_len_arr"].values) * 6) / target_length
templates = get_template(motifs, kmer_dict)
templates = np.stack(templates).repeat(target_length, axis=1)

print(templates.shape)
templates = scipy.interpolate.CubicSpline(np.arange(templates.shape[1]), templates, axis=1)(np.arange(templates.shape[1]))
print(templates.shape)

## Dynamic Time Warping

interpolated = []
warps = []
for i in range(nrows):
    original = signals[i]
    template = templates[i]
    step_size = step_sizes[i]

    steps = step_size.repeat(target_length).cumsum()
    print(steps.shape)
    interp = np.interp(steps, np.arange(len(original)), original)
    interpolated.append(interp)
    warped = dtw.warp(interp, template)[0]
    warps.append(warped)

fig, axes = plt.subplots(figsize=(20,5*nrows), nrows=nrows, ncols=1)
for i, x in enumerate(range(nrows)):
    axes[i].plot(signals[x], label=f"Original")
    axes[i].plot(interpolated[x], label=f"Interpolated")
    axes[i].plot(templates[x], label=f"Template")
    axes[i].plot(warps[x], label=f"Warped")
    axes[i].legend()

plt.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/dtw.png")
plt.close(fig)