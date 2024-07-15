pod5_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/raw/pod5"
import pod5
from utils import augmentations as aug
import glob
import numpy as np

pod5s = glob.glob(f"{pod5_path}/*pass*.pod5")
sample = 10
pod5s = np.random.choice(pod5s, sample)

for pod5_path in pod5s:
    pod5_data = pod5.load(pod5_path)
    signal = pod5_data["signal"]
    signal = torch.tensor(signal).float().to(1)
    print(signal)
    print(signal.shape)
    break
