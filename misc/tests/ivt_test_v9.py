import pandas as pd

airna = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240626-093708-12-215000-token_normalise_drach-ivt_notrim_pileup/pileup.pkl"
airna = pd.read_pickle(airna)
airna = airna[airna["count_dom"] >= 20]
print(airna)
airna = airna["dom"].values

dorado = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/old/dorado_output.modkit.drach.pkl")
dorado = dorado[dorado["dorado_count"] >= 20]
print(dorado)
dorado = dorado["pred_dorado"].values

airna = airna[~pd.isnull(airna)]
dorado = dorado[~pd.isnull(dorado)]

print("TOTAL")
print(f"AIRNA: {len(airna)}")
print(f"Dorado: {len(dorado)}")

print("LARGER THAN 0.5")
print(f"AIRNA: {len(airna[airna > 0.5])}")
print(f"Dorado: {len(dorado[dorado > 0.5])}")

print("LARGER THAN 0.75")
print(f"AIRNA: {len(airna[airna > 0.75])}")
print(f"Dorado: {len(dorado[dorado > 0.75])}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.histplot(airna, bins=100, ax=ax[0], stat="density")
sns.histplot(dorado, bins=100, ax=ax[1], stat="density")
for a in ax:
    a.set_xlim(0, 1)
    a.set_ylim(0, 100)
    a.set_xlabel("Prediction")
    a.set_ylabel("Density")
ax[0].set_title("AIRNA")
ax[1].set_title("Dorado")
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/prediction_histogram_dorado.png")
