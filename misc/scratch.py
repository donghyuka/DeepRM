import numpy as np
import torch
from utils import augmentations as aug
import matplotlib.pyplot as plt
import pandas as pd

path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/val/pos/000016697.pkl"
data = pd.read_pickle(path)
data["signal_len"] = data["signal_token"].apply(lambda x: len(x))
data = data[(data["signal_len"] > 600) & (data["signal_len"] < 1200)]
print(data)

nrows=4
fig, axes = plt.subplots(figsize=(20,10), nrows=nrows, ncols=1)

data = data["signal_token"].sample(5).values
data_list = []
pad_mask = []
for i in range(nrows):
    pad_len = 1200 - len(data[i])
    data_list.append(np.pad(data[i], (0,pad_len)))
    pad_mask.append([1]*len(data[i]) + [0]*pad_len)
data = np.stack(data_list)
data = torch.tensor(data).float().unsqueeze(2)
pad_mask = torch.tensor(pad_mask).float().unsqueeze(2)

## GAUSSIAN JITTER
t_data = aug.jitter(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Jitter")

## MAGNITUDE WARP
t_data = aug.magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.6, n_knots = 20) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Mag Warp")

## MOVING AVERAGE MAGNITUDE WARP
t_data = aug.moving_magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.8, n_knots = 60) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Moving Warp")

## SPIKE NOISE
t_data = aug.jitter(data, fraction=1.0, min_sigma=1.0, max_sigma=2.0, dropout = 0.95) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Spike")

## WINDOWED TIME WARP
t_data = aug.window_warp(data, fraction=1.0, window_ratio = 0.1, window_count=5)
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Window Warp")

## TIME WARP
t_data = aug.time_warp(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, n_knots = 40)
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Time Warp")

## STEP NOISE
t_data = aug.step(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, dropout = 0.95) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Step")

## SLOPE NOISE
t_data = aug.slope(data, fraction=1.0) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Slope")

## DRIFT NOISE
t_data = aug.drift(data, fraction=1.0) * pad_mask
print(t_data[0,:,0])
for i in range(nrows):
    axes[i].plot(t_data[i,:,0], label=f"Drift")

## Original
print(data[0,:,0])
for i in range(nrows):
    axes[i].plot(data[i,:,0], label=f"Original")

for ax in axes:
    ax.set_xlim(0,600)
    ax.set_ylim(-3,3)

    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/augmentation.png")
plt.close(fig)

