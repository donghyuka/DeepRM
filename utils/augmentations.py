## Modified from: https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py
## Drift augmentation modified from: https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/drift.py
## Slope augmentation modified from: https://github.com/timeseriesAI/tsai/blob/main/tsai/data/transforms.py
## Warps modified from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py



import numpy as np
import scipy.interpolate
from typing import List, Dict, Any, Optional, Tuple, Union
from torch import Tensor
import torch 


def jitter(X: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 0.0, min_sigma: float = 0.5) -> Tensor:
    ## Can be used for both jitter and spike augmentation

    gauss = torch.normal(mean, 1.0, X.shape)
    random_sigma = torch.rand(X.shape[0], device=X.device) * (max_sigma - min_sigma) + min_sigma
    gauss = gauss * random_sigma
    gauss = gauss.clip(-3*random_sigma, 3*random_sigma)
    mask = torch.rand(X.shape[0], device=X.device) < fraction
    gauss = gauss * mask
    dropout_mask = torch.rand(X.shape, device=X.device) < (1-dropout)
    gauss = gauss * dropout_mask

    return X + gauss


def magnitude_warp(X: Tensor, fraction: float = 0.5,
                   max_sigma: float = 0.0, min_sigma: float = 0.5,
                   n_knots: int = 4) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = np.arange(n_timesteps)
    random_warps = np.random.normal(loc=1.0, scale=1.0, size=(n_data, n_knots + 2, n_features))
    random_sigma = np.random.uniform(min_sigma, max_sigma, size=(n_data, 1, n_features))
    random_warps = (random_warps - 1) * random_sigma + 1
    warp_steps = (np.ones( (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T

    result = X.clone()

    for i in idx_aug:
        warper = np.array([scipy.interpolate.CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                for dim in range(n_features)])
        warper = np.clip(warper, 0.0, None)
        warper = torch.tensor(warper, device=X.device).T
        result[i] = X[i] * warper

    return result


def window_warp(X: Tensor, fraction: float = 0.5, window_ratio: float = 0.2, scales: Tuple = (0.5, 2.0),
                   window_count: int = 1) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = np.arange(n_timesteps)
    scales_per_sample = np.random.choice(scales, (n_data, window_count))
    warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)

    result = X.clone()
    window_starts = np.random.randint(low=0, high=n_timesteps - warp_size, size=(n_data, window_count))
    window_ends = window_starts + warp_size

    for i in idx_aug:
        sample = X[i]
        for dim in range(n_features):
            for j in range(window_count):
                start_seg = sample[:window_starts[i, j], dim]
                warp_ts_size = max(round(warp_size * scales_per_sample[i, j]), 1)
                window_seg = np.interp(
                    x=np.linspace(0, warp_size - 1, num=warp_ts_size),
                    xp=np.arange(warp_size),
                    fp=X[i][window_starts[i, j] : window_ends[i, j], dim],
                )
                end_seg = sample[window_ends[i, j] :, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                result[i, :, dim] = torch.tensor(np.interp(
                    orig_steps,np.linspace(0, n_timesteps - 1.0, num=warped.size),warped), device=X.device)


    return result


def time_warp(X: Tensor, fraction: float = 0.5, max_sigma: float = 0.0, min_sigma: float = 0.5,
                   n_knots: int = 4) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = np.arange(n_timesteps)
    random_warps = np.random.normal(loc=1.0, scale=1.0, size=(n_data, n_knots + 2, n_features))
    random_sigma = np.random.uniform(min_sigma, max_sigma, size=(n_data, 1, n_features))
    random_warps = (random_warps - 1) * random_sigma + 1
    warp_steps = (np.ones( (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T

    result = X.clone()

    for i in idx_aug:
        for dim in range(n_features):
            time_warp = scipy.interpolate.CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (n_timesteps-1)/time_warp[-1]
            result[i,:,dim] =  torch.tensor(np.interp(orig_steps, np.clip(scale*time_warp, 0, n_timesteps-1), X[i][:,dim]), device=X.device)

    return result


def moving_magnitude_warp(X: Tensor, fraction: float = 0.5,
                   max_sigma: float = 0.0, min_sigma: float = 0.5,
                   n_knots: int = 4, window_size: int = 7,) -> Tensor:

    assert window_size % 2 ==1, "Window size must be odd"

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = np.arange(n_timesteps)
    random_warps = np.random.normal(loc=1.0, scale=1.0, size=(n_data, n_knots + 2, n_features))
    random_sigma = np.random.uniform(min_sigma, max_sigma, size=(n_data, 1, n_features))
    random_warps = (random_warps - 1) * random_sigma + 1
    warp_steps = (np.ones( (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T
    result = X.clone()

    movavg = torch.avg_pool1d(X.permute(0,2,1), window_size, 1, count_include_pad=False)
    print((movavg[:,:,0].repeat(1, 1, window_size//2)).shape)
    movavg = torch.cat([movavg[:,:,0].repeat(1, window_size//2).unsqueeze(1),
                        movavg,
                        movavg[:,:,-1].repeat(1, window_size//2).unsqueeze(1)], dim=2)
    movavg = movavg.permute(0,2,1)

    for i in idx_aug:
        warper = np.array([scipy.interpolate.CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                           for dim in range(n_features)])
        warper = np.clip(warper, 0.0, None)
        warper = torch.tensor(warper, device=X.device).T
        diff = warper * (X[i] - movavg[i])
        result[i] = X[i] + diff

    return result


def step(X: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 0.0, min_sigma: float = 0.5) -> Tensor:

    gauss = torch.normal(mean, 1.0, X.shape, device=X.device)
    random_sigma = torch.rand(X.shape[0]) * (max_sigma - min_sigma) + min_sigma
    gauss = gauss * random_sigma
    gauss = gauss.clip(-3*random_sigma, 3*random_sigma)
    mask = torch.rand(X.shape[0], device=X.device) < fraction
    gauss = gauss * mask
    dropout_mask = torch.rand(X.shape, device=X.device) < (1-dropout)
    gauss = gauss * dropout_mask

    ## Cumsum to convert spike to step
    gauss = torch.cumsum(gauss, dim=1)

    return X + gauss


def slope(X: Tensor, fraction: float = 0.5,  magnitude = 1.0) -> Tensor:

    if X.shape[-1] == 1:
        flat_x = X.squeeze(-1)
    else:
        flat_x = X.reshape(X.shape[0], -1)

    series_range = flat_x.max(dim=-1, keepdim=True)[0] - flat_x.min(dim=-1, keepdim=True)[0]
    trend = torch.linspace(0, 1, flat_x.shape[-1], device=X.device) * series_range
    slope = (torch.rand((X.shape[0],1), device=X.device) - 0.5)
    trend = magnitude * 2 * slope * trend
    trend -= trend.median(-1, keepdim=True)
    mask = torch.rand(X.shape[0], device=X.device) < fraction
    trend = trend * mask.unsqueeze(-1)
    trend = trend.unsqueeze(-1)

    return X + trend


def drift(X: Tensor, fraction: float = 0.5,
                   max_sigma: float = 0.0, min_sigma: float = 0.5,
                   n_knots: int = 4) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = np.arange(n_timesteps)
    random_warps = np.random.normal(loc=0.0, scale=1.0, size=(n_data, n_knots + 2, n_features))
    random_sigma = np.random.uniform(min_sigma, max_sigma, size=(n_data, 1, n_features))
    random_warps = random_warps * random_sigma
    warp_steps = (np.ones( (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T

    result = X.clone()

    for i in idx_aug:
        warper = np.array([scipy.interpolate.CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                           for dim in range(n_features)])
        warper = torch.tensor(warper, device=X.device).T
        result[i] = X[i] + warper

    return result

