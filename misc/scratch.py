import tsai.data.transforms as tfms
import tsai.data.core as core
import torch

## make a batch of random time series data
data = torch.randn(1, 1, 128)
print(data)

## Augment
t_data = tfms.TSTimeWarp(magnitude=0.5)(data, split_idx=0)
print(t_data)

print(t_data == data)

