import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
from utils.utils import is_drach

result = []

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/token_standardise_drach/*.pkl"

path = glob.glob(path)
path = np.random.choice(path, 10)
data = pd.concat([pd.read_pickle(p) for p in path], axis=0)
data["5mer"] = data["motif"].str[8:13]
print(data["signal"])
data["signal"] = data["signal"].apply(lambda x: np.mean(x[10]))
data["group"] = "IVT"
result.append(data)

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/token_standardise_drach_v2/*.pkl"

path = glob.glob(path)
path = np.random.choice(path, 10)
data = pd.concat([pd.read_pickle(p) for p in path], axis=0)
data["5mer"] = data["motif"].str[8:13]
print(data["signal"])
data["signal"] = data["signal"].apply(lambda x: np.mean(x[10]))
data["group"] = "CELL"
result.append(data)

data = pd.concat(result, axis=0).reset_index(drop=True)
data["drach"] = data["5mer"].apply(is_drach)
data = data[data["drach"]]
print(data)

## Box plot

fig, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x="5mer", y="signal", hue="group", data=data, ax=ax)
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/ivt_tok.png")

