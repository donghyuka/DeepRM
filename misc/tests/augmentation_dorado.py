pod5_path = "/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v14/source.pkl"
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
    pad_mask = pod5_data["pad_mask"]
    signal = torch.tensor(signal).float().to(1)
    pad_mask = torch.tensor(pad_mask).float().to(1)
    aug_list = aug.get_aug_list(1.0)
    signal = aug.augment_signal(aug_list, signal, pad_mask)
    print(signal)
    print(signal.shape)
    print(pad_mask)
    print(pad_mask.shape)
    break
