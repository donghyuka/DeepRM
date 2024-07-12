import numpy as np
import torch
from utils import augmentations as aug
import matplotlib.pyplot as plt
import pandas as pd
import time

signal_stride = 6
kmer_size = 5

def signal_sliding_win(signal_segmented, stride = 6, win_size = 5):
    signal_segmented = np.lib.stride_tricks.sliding_window_view(signal_segmented, win_size * stride)[::stride]
    return signal_segmented

path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver070124/score-perfect/val/pos/000016697.pkl"
df = pd.read_pickle(path)
df["signal_len"] = df["signal_token"].apply(lambda x: len(x))
df = df[df["signal_len"] < 1200]
data = df["signal_token"].values
data_list = []
pad_mask = []
for i in range(len(data)):
    pad_len = 1200 - len(data[i])
    data_list.append(np.pad(data[i], (0,pad_len+24)))
    pad_mask.append([1]*(len(data[i])//signal_stride) + [0]*(pad_len//signal_stride))
data = np.stack(data_list)
data = torch.tensor(data).float()
pad_mask = torch.tensor(pad_mask).float()

data = data.unfold(1, signal_stride * kmer_size, signal_stride).clip(-10,10) * pad_mask.unsqueeze(-1)


df["signal_token"] = df["signal_token"].apply(lambda x: signal_sliding_win(x, signal_stride, kmer_size))
data2 = df["signal_token"].values
data_list = []
for i in range(len(data2)):
    pad_len = 200 - data2[i].shape[0]
    x = np.concatenate([data2[i], np.zeros((pad_len, kmer_size * signal_stride))], axis = 0)
    data_list.append(x)

data2 = np.stack(data_list)
data2 = torch.tensor(data2).float()


print(data.shape)
print(data2.shape)

## Check equality
print(data[0][:,0])
print(data2[0][:,0])
print(torch.allclose(data, data2))

