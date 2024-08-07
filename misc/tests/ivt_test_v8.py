import pandas as pd

ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240626-093708-12-215000-token_normalise_drach-ivt_notrim_pileup/pileup.pkl"
cell_path = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240626-093708-12-215000-token_normalise_drach_v2_pileup/pileup.pkl"

ivt = pd.read_pickle(ivt_path)
cell = pd.read_pickle(cell_path)

ivt_preds = ivt[ivt["count_dom"] >= 20]["dom"].values
cell_preds = cell[cell["count_dom"] >= 20]["dom"].values

ivt_preds = ivt_preds[~pd.isnull(ivt_preds)]
cell_preds = cell_preds[~pd.isnull(cell_preds)]

ivt_preds = ivt_preds[ivt_preds >= 0]
cell_preds = cell_preds[cell_preds >= 0]


print("IVT", len(ivt_preds[ivt_preds > 0.5])/len(ivt_preds))
print("CELL", len(cell_preds[cell_preds > 0.5])/len(cell_preds))


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.histplot(cell_preds, bins=100, ax=ax[0], stat="density")
sns.histplot(ivt_preds, bins=100, ax=ax[1], stat="density")
for a in ax:
    a.set_xlim(0, 1)
    a.set_ylim(0, 60)
    a.set_xlabel("Prediction")
    a.set_ylabel("Density")
ax[0].set_title("CELL")
ax[1].set_title("IVT")
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/prediction_histogram.png")
