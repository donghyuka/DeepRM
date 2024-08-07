path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.Genomic.GP3.depth5_None.twm6astrict.drach.tsv"

import pandas as pd

df = pd.read_csv(path, sep="\t")
df = df[df["label"]==1]
df = df[df["m6A_level"]>0]
df = df[["m6A_level", "5mer"]]

import seaborn as sns
import matplotlib.pyplot as plt

## Sort by mean m6A level
df_sort_key = df.groupby("5mer")["m6A_level"].mean()
df["sort_key"] = df["5mer"].map(df_sort_key)
df = df.sort_values("sort_key", ascending=False)


## Draw violin plot
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(20, 10))
sns.violinplot(x="5mer", y="m6A_level", data=df, ax=ax)
plt.xticks(rotation=45)
fig.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/m6A_level_violin.png")