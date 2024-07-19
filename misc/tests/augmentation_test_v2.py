import numpy as np
import torch
from utils import augmentations as aug
import matplotlib.pyplot as plt
import pandas as pd
import time
from functools import partial
from tqdm import tqdm
import glob



def get_aug_list(fraction):
    ## MOVING AVERAGE MAGNITUDE WARP
    movmag = partial(aug.moving_magnitude_warp, fraction=fraction, min_sigma=0.1, max_sigma=0.2, n_knots = 40)
    ## WINDOWED TIME WARP
    winwarp = partial(aug.window_warp, fraction=fraction, min_window_ratio = 0.05, max_window_ratio = 0.10,
                      min_window_count = 5, max_window_count = 20, sigma = 0.4)
    ## TIME WARP
    timewarp = partial(aug.time_warp, fraction=fraction, min_sigma=0.2, max_sigma=0.4, n_knots = 30)
    ## GAUSSIAN JITTER
    jitter = partial(aug.jitter, fraction=fraction, min_sigma=0.15, max_sigma=0.3)
    ## SPIKE NOISE
    spike = partial(aug.jitter, fraction=fraction, min_sigma=0.3, max_sigma=1.0, dropout = 0.95)
    ## STEP NOISE
    step = partial(aug.step, fraction=fraction, min_sigma=0.15, max_sigma=0.3, dropout = 0.95)
    ## SLOPE NOISE
    slope = partial(aug.slope, fraction=fraction, magnitude = 0.5,)
    ## DRIFT NOISE
    drift = partial(aug.drift, fraction=fraction, min_sigma=0.15, max_sigma=0.3, n_knots = 10)
    aug_list = [movmag, winwarp, timewarp, jitter, spike, step, drift, slope]
    return aug_list



def augment_signal(aug_list, signal, pad_mask, shuffle_order = True):

    if shuffle_order:
        aug_list = np.random.permutation(aug_list)

    for aug_func in aug_list:
        signal = aug_func(signal, pad_mask=pad_mask)

    return signal

paths = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/train/pos/*.pkl"
paths = glob.glob(paths)
for path in tqdm(list(paths)[42:43]):
    data = pd.read_pickle(path)
    data = data["signal_token"].values
    data_list = []
    pad_mask = []
    for i in range(len(data)):
        pad_len = 1200 - len(data[i])
        if pad_len > 0:
            data_list.append(np.pad(data[i], (0,pad_len)))
            pad_mask.append(np.concatenate([np.ones(len(data[i])), np.zeros(pad_len)]))
        else:
            data_list.append(data[i][:1200])
            pad_mask.append(np.ones(1200))
    data = np.stack(data_list)
    data = torch.tensor(data, dtype=torch.float)
    pad_mask = torch.tensor(np.stack(pad_mask), dtype=torch.int)
    # data = data.to(1)
    # pad_mask = pad_mask.to(1)
    data_aug = augment_signal(get_aug_list(0.3), data, pad_mask)


nrows=100
fig, axes = plt.subplots(figsize=(20,5*nrows), nrows=nrows, ncols=1)
xs = np.random.choice(range(len(data)), nrows)
for i, x in enumerate(xs):
    axes[i].plot(data[x], label=f"Original")
    axes[i].plot(data_aug[x], label=f"Augmented")

plt.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/augmentation_v2.png")
plt.close(fig)