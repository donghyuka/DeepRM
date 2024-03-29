import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/block_df.pkl"
df = pd.read_pickle(path)
score_arr = df["penalty"].values
threshold = 10
score_arr = score_arr[score_arr < threshold]
score_arr_pos = (threshold - score_arr) / threshold
path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado/intermediates/block_df.pkl"
df = pd.read_pickle(path)
score_arr = df["penalty"].values
threshold = 10
score_arr = score_arr[score_arr < threshold]
score_arr_neg = (threshold - score_arr) / threshold
fig, ax = plt.subplots(figsize=(20,20))

sns.histplot(score_arr_neg, ax=ax, label="cA", color="tomato", bins=10, alpha=0.5)
sns.histplot(score_arr_pos, ax=ax, label="m6A", color="royalblue", bins=10, alpha=0.5)
ax.set_xlim(0,1)
ax.set_xlabel("Score")
ax.set_ylabel("Density")
ax.legend()

fig.savefig("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/score_hist_2.png", dpi=300)