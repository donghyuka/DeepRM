import numpy as np
import torch
from utils import augmentations as aug
import matplotlib.pyplot as plt
import pandas as pd
import time
import cupy

path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/val/pos/000016697.pkl"
df = pd.read_pickle(path)
df["signal_len"] = df["signal_token"].apply(lambda x: len(x))
df = df[df["signal_len"] < 1200]
data = df["signal_token"].values
data_list = []
pad_mask = []
for i in range(len(data)):
    pad_len = 1200 - len(data[i])
    data_list.append(np.pad(data[i], (0,pad_len)))
    pad_mask.append([1]*len(data[i]) + [0]*pad_len)
data = np.stack(data_list)
data = torch.tensor(data).float()
pad_mask = torch.tensor(pad_mask).float()

data = data.to(1)
pad_mask = pad_mask.to(1)

with torch.no_grad():

    ## GAUSSIAN JITTER
    start = time.time()
    t_data = aug.jitter(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, pad_mask = pad_mask)
    print("JITTER: ",time.time() - start)

    ## SPIKE NOISE
    start = time.time()
    t_data = aug.jitter(data, fraction=1.0, min_sigma=1.0, max_sigma=2.0, dropout = 0.95, pad_mask = pad_mask)
    print("SPIKE: ",time.time() - start)

    ## STEP NOISE
    start = time.time()
    t_data = aug.step(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, dropout = 0.95, pad_mask = pad_mask)
    print("STEP: ",time.time() - start)

    ## MAGNITUDE WARP
    start = time.time()
    t_data = aug.magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.6, n_knots = 20, pad_mask = pad_mask)
    print("MAGNITUDE: ",time.time() - start)

    ## SLOPE NOISE
    start = time.time()
    t_data = aug.slope(data, fraction=1.0, magnitude = 1.0, pad_mask = pad_mask)
    print("SLOPE: ",time.time() - start)

    ## DRIFT NOISE
    start = time.time()
    t_data = aug.drift(data, fraction=1.0, min_sigma=0.5, max_sigma=1.0, n_knots = 40, pad_mask = pad_mask)
    print("DRIFT: ",time.time() - start)

    ## TIME WARP
    start = time.time()
    t_data = aug.time_warp(data, fraction=1.0, min_sigma=0.1, max_sigma=0.2, n_knots = 40, pad_mask = pad_mask)
    print("TIME: ",time.time() - start)

    ## MOVING AVERAGE MAGNITUDE WARP
    start = time.time()
    t_data = aug.moving_magnitude_warp(data, fraction=1.0, min_sigma=0.2, max_sigma=0.8, n_knots = 60, pad_mask = pad_mask)
    print("MOVING: ",time.time() - start)

    ## WINDOWED TIME WARP
    start = time.time()
    t_data = aug.window_warp(data, fraction=1.0, pad_mask = pad_mask)
    print("WINDOWED: ",time.time() - start)

def signal_sliding_win(signal_segmented, stride = 6, win_size = 5):
    signal_segmented = cupy.lib.stride_tricks.sliding_window_view(signal_segmented, win_size * stride)[::stride]
    return signal_segmented

print(df["signal_token"])

df["signal_token"] = df["signal_token"].apply(lambda x: signal_sliding_win(x))

print(df["signal_token"])