import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/old/block_v2/intermediates/signal_raw/9-9-9.pkl"
data = pd.read_pickle(path)
path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/old/block_v2/intermediates/move_df_split/9-9-9.pkl"
data2 = pd.read_pickle(path)
data = data.merge(data2, on="read_id", how="inner")
print([x[1] for x in data["mv"].tolist()])
data = data.sample(4)
fig, axes = plt.subplots(figsize=(40, 20), nrows=4, ncols=1)
for i, (idx, row) in enumerate(data.iterrows()):


    axes[i].set_title(row["read_id"])
    axes[i].set_ylim(40,120)
    axes[i].set_xlim(0,10000)

    signal = row["signal"][row["sp"]:]
    axes[i].plot(((signal + row["offset"]) * row["scale"]), color="grey")

    move = np.array(row["mv"])[1:]
    move_idx = np.where(move == 1)[0][1:] * 6 + row["ts"]
    for m in move_idx:
        axes[i].axvline(m, color="blue", linestyle="-")

fig.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/ivt.png")
plt.close(fig)


path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/intermediates/signal_raw/9-9-9.pkl"
data = pd.read_pickle(path)
path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/intermediates/move_df_split/9-9-9.pkl"
data2 = pd.read_pickle(path)
data = data.merge(data2, on="read_id", how="inner")
print(data)
data = data.sample(4)
fig, axes = plt.subplots(figsize=(40, 20), nrows=4, ncols=1)
for i, (idx, row) in enumerate(data.iterrows()):


    axes[i].set_title(row["read_id"])
    axes[i].set_ylim(40,120)
    axes[i].set_xlim(0,10000)

    signal = row["signal"][row["sp"]:]
    axes[i].plot(((signal + row["offset"]) * row["scale"]), color="grey")

    move = np.array(row["mv"])[1:]
    move_idx = np.where(move == 1)[0][1:] * 6 + row["ts"]
    for m in move_idx:
        axes[i].axvline(m, color="blue", linestyle="-")

fig.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/transcriptome.png")
plt.close(fig)


# airna = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/airna.pkl"
# dorado = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado.pkl"
#
# airna = pd.read_pickle(airna)
# dorado = pd.read_pickle(dorado)
#
# print(airna)
# print(dorado)
#
# print(dorado["pred_dorado"].describe())
# airna = airna["pm6a"].values
# dorado = dorado["pred_dorado"].values
#
#
# fig, ax = plt.subplots(figsize=(20, 10))
# sns.histplot(airna, bins=100, color="blue", alpha=0.5, label="AIRNA", stat="density")
# sns.histplot(dorado, bins=100, color="red", alpha=0.5, label="Dorado", stat="density")
# ax.set_title("Prediction Histogram")
# ax.set_xlabel("Prediction")
# ax.set_ylabel("Frequency")
# plt.legend()
# plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/airna_dorado.png")
# plt.close()
#
