import numpy as np
import torch
from utils import augmentations as aug
import matplotlib.pyplot as plt
import pandas as pd
import time

path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/val/pos/000016697.pkl"
data = pd.read_pickle(path)
data["signal_len"] = data["signal_token"].apply(lambda x: len(x))
data = data[(data["signal_len"] > 600) & (data["signal_len"] < 1200)]
print(data)

nrows=10
fig, axes = plt.subplots(figsize=(20,40), nrows=nrows, ncols=1)

data = data["signal_token"].values
data_list = []
pad_mask = []
for i in range(nrows):
    pad_len = 1200 - len(data[i])
    data_list.append(np.pad(data[i], (0,pad_len)))
    pad_mask.append([1]*len(data[i]) + [0]*pad_len)
data = np.stack(data_list)
data = torch.tensor(data).float()
pad_mask = torch.tensor(pad_mask).float()

start = time.time()

## GAUSSIAN JITTER
t_data = aug.jitter(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Jitter")

## MAGNITUDE WARP
t_data = aug.magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.6, n_knots = 20, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Mag Warp")

## MOVING AVERAGE MAGNITUDE WARP
t_data = aug.moving_magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.8, n_knots = 60, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Moving Warp")

## SPIKE NOISE
t_data = aug.jitter(data, fraction=1.0, min_sigma=1.0, max_sigma=2.0, dropout = 0.95, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Spike")

## WINDOWED TIME WARP
t_data = aug.window_warp(data, fraction=1.0, pad_mask = pad_mask)
# t_data = aug.window_warp(data, fraction=1.0, window_ratio = 0.1, window_count=5, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Window Warp")

## TIME WARP
t_data = aug.time_warp(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, n_knots = 10, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Time Warp")

## STEP NOISE
t_data = aug.step(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, dropout = 0.95, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Step")

## SLOPE NOISE
t_data = aug.slope(data, fraction=1.0, magnitude = 1.0, pad_mask = pad_mask,)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Slope")

## DRIFT NOISE
t_data = aug.drift(data, fraction=1.0, min_sigma=0.5, max_sigma=1.0, n_knots = 40, pad_mask = pad_mask)
print(t_data[0])
for i in range(nrows):
    axes[i].plot(t_data[i], label=f"Drift")

## Original
print(data[0])
for i in range(nrows):
    axes[i].plot(data[i], label=f"Original")

for ax in axes:
    ax.set_xlim(0,800)
    ax.set_ylim(-3,3)

    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/augmentation.png")
plt.close(fig)
print(time.time() - start)