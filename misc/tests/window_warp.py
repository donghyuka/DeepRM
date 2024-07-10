import torch
from torch import Tensor
from typing import Tuple

def window_warp(X: Tensor, fraction: float = 0.5, window_ratio: float = 0.2, scales: Tuple = (0.5, 2.0),
                window_count: int = 1, pad_mask: Tensor = None) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = int(n_data * fraction)
    idx_aug = torch.randperm(n_data)[:n_aug]

    scales_per_sample = torch.tensor(scales, device=X.device).repeat(n_data, window_count).view(n_data, window_count)
    warp_size = max(int(window_ratio * n_timesteps), 1)

    result = X.clone()

    window_starts = torch.randint(low=0, high=n_timesteps - warp_size, size=(n_data, window_count), device=X.device)
    window_ends = window_starts + warp_size

    for i in idx_aug:
        sample = X[i]
        for j in range(window_count):
            start_seg = sample[:window_starts[i, j]]
            warp_ts_size = max(int(warp_size * scales_per_sample[i, j]), 1)
            window_seg = torch.nn.functional.interpolate(
                sample[window_starts[i, j]:window_ends[i, j]].unsqueeze(0).unsqueeze(0),
                size=warp_ts_size,
                mode='linear',
                align_corners=False
            ).squeeze()
            end_seg = sample[window_ends[i, j]:]
            warped = torch.cat((start_seg, window_seg, end_seg))
            result[i] = torch.nn.functional.interpolate(
                warped.unsqueeze(0).unsqueeze(0),
                size=n_timesteps,
                mode='linear',
                align_corners=False
            ).squeeze()

    return result