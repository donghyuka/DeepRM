import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block_v2/token_standardise_drach_ivt/*.pkl"

path = glob.glob(path)
path = np.random.choice(path, 10)
data = pd.concat([pd.read_pickle(p) for p in path], axis=0)
print(data)
data["5mer"] = data["motif"].str[8:13]
data["signal"] = data["signal"].apply(lambda x: x[10])
data = data.groupby("5mer")
group = ["AAACA","AAACC","AAACT","AGACA","AGACC","AGACT",
         "GAACA","GAACC","GAACT","GGACA","GGACC","GGACT",
         "TAACA","TAACC","TAACT","TGACA","TGACC","TGACT"]
fig, axes = plt.subplots(figsize=(40, 20), nrows=3, ncols=6)
for i, group in enumerate(group):
    ax = axes[i//6, i%6]
    try:
        df = data.get_group(group)
    except:
        continue
    for _, row in df.iterrows():
        ax.plot(row["signal"], color="grey", alpha=0.1, lw=0.3)
    ax.set_title(group)
    ax.set_ylim(-3,3)
    ax.set_xlim(0,100)

fig.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/ivt_tok.png")
plt.close(fig)

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/token_standardise_drach_v2/*.pkl"

path = glob.glob(path)
path = np.random.choice(path, 10)
data = pd.concat([pd.read_pickle(p) for p in path], axis=0)
print(data)
data["5mer"] = data["motif"].str[8:13]
data["signal"] = data["signal"].apply(lambda x: x[10])
data = data.groupby("5mer")
group = ["AAACA","AAACC","AAACT","AGACA","AGACC","AGACT",
         "GAACA","GAACC","GAACT","GGACA","GGACC","GGACT",
         "TAACA","TAACC","TAACT","TGACA","TGACC","TGACT"]
fig, axes = plt.subplots(figsize=(40, 20), nrows=3, ncols=6)
for i, group in enumerate(group):
    ax = axes[i//6, i%6]
    try:
        df = data.get_group(group)
    except:
        continue
    for _, row in df.iterrows():
        ax.plot(row["signal"], color="grey", alpha=0.1, lw=0.3)
    ax.set_title(group)
    ax.set_ylim(-3,3)
    ax.set_xlim(0,100)

fig.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/transcriptome_tok.png")
plt.close(fig)