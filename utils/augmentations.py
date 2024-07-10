## Modified from: https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py
## Drift augmentation modified from: https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/drift.py
## Slope augmentation modified from: https://github.com/timeseriesAI/tsai/blob/main/tsai/data/transforms.py
## Warps modified from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py

## This script procides accelerated implementations of various time series augmentations.
## This script is for on-line augmentation, in which the speed is crucial.
## Multichannel support has been removed for speed.
## Supports both CPU and GPU, and considerably faster on GPU.


import torch
from torch import Tensor
import numpy as np
from utils.interp1d import interp1d
from utils.cubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def jitter(X: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 1.0, min_sigma: float = 0.5, pad_mask: Tensor = None) -> Tensor:
    ## Can be used for both jitter and spike augmentation

    gauss = torch.normal(mean, 1.0, X.shape, device=X.device)
    random_sigma = torch.rand((X.shape[0],1), device=X.device) * (max_sigma - min_sigma) + min_sigma
    gauss = gauss * random_sigma
    gauss = gauss.clip(-3 * max_sigma, 3 * max_sigma)

    mask = torch.rand((X.shape[0],1), device=X.device) < fraction
    gauss = gauss * mask
    dropout_mask = torch.rand(X.shape, device=X.device) < (1-dropout)
    gauss = gauss * dropout_mask
    result =  X + gauss
    if pad_mask is not None:
        result = result * pad_mask
    return result


def step(X: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 1.0, min_sigma: float = 0.5, pad_mask: Tensor = None) -> Tensor:

    gauss = torch.normal(mean, 1.0, X.shape, device=X.device)
    random_sigma = torch.rand((X.shape[0],1), device=X.device) * (max_sigma - min_sigma) + min_sigma
    gauss = gauss * random_sigma
    gauss = gauss.clip(-3 * max_sigma, 3 * max_sigma)

    mask = torch.rand((X.shape[0],1), device=X.device) < fraction
    gauss = gauss * mask
    dropout_mask = torch.rand(X.shape, device=X.device) < (1-dropout)
    gauss = gauss * dropout_mask

    ## Cumsum to convert spike to step
    gauss = torch.cumsum(gauss, dim=1)
    result =  X + gauss

    if pad_mask is not None:
        result = result * pad_mask

    return result


def slope(X: Tensor, fraction: float = 0.5,  magnitude: float = 1.0, pad_mask: Tensor = None) -> Tensor:

    if pad_mask is None:
        pad_mask = torch.ones(X.shape, device=X.device)

    lengths = X.shape[-1] / pad_mask.sum(dim=-1, keepdim=True)

    series_range = X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0]
    trend = torch.linspace(0, 1, X.shape[-1], device=X.device)
    slope = (torch.rand((X.shape[0],1), device=X.device) - 0.5) * 2
    trend = trend * magnitude * lengths * series_range * slope
    trend_mean = trend[:,-1:] * 0.5 / lengths
    trend = trend - trend_mean
    mask = torch.rand((X.shape[0],1), device=X.device) < fraction
    trend = trend * mask
    result = X + trend
    result = result * pad_mask

    return result


def drift(X: Tensor, fraction: float = 0.5,
                   max_sigma: float = 1.0, min_sigma: float = 0.5,
                   n_knots: int = 4, pad_mask: Tensor = None) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]

    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps =torch.normal(0.0, 1.0, (n_aug, n_knots + 2), device=X.device)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = random_warps * random_sigma
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)

    result = X.clone()
    warper = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                   ).evaluate(orig_steps).squeeze(-1)
    result[idx_aug] += warper

    if pad_mask is not None:
        result = result * pad_mask

    return result



def magnitude_warp(X: Tensor, fraction: float = 0.5,
                   max_sigma: float = 1.0, min_sigma: float = 0.5,
                   n_knots: int = 4, pad_mask: Tensor = None) -> Tensor:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]

    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps = torch.normal(1.0, 1.0, (n_aug, n_knots + 2), device=X.device)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = random_warps * random_sigma
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)

    result = X.clone()
    warper = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                ).evaluate(orig_steps).squeeze(-1).clip(0.0, None)
    result[idx_aug] *= warper

    if pad_mask is not None:
        result = result * pad_mask

    return result


def moving_magnitude_warp(X: Tensor, fraction: float = 0.5,
                          max_sigma: float = 1.0, min_sigma: float = 0.5,
                          n_knots: int = 4, window_size: int = 7, pad_mask: Tensor = None) -> Tensor:

    assert window_size % 2 ==1, "Window size must be odd"

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps = torch.normal(1.0, 1.0, (n_aug, n_knots + 2), device=X.device)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = random_warps * random_sigma
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)
    result = X.clone()

    movavg = torch.avg_pool1d(X[idx_aug], window_size, 1, count_include_pad=False)
    movavg = torch.cat([movavg[:,0:1].repeat(1, window_size//2),
                        movavg,
                        movavg[:,-1:].repeat(1, window_size//2)], dim=1)

    warper = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                ).evaluate(orig_steps).squeeze(-1).clip(0.0, None)

    diff = warper * (X[idx_aug] - movavg)
    result[idx_aug] += diff

    if pad_mask is not None:
        result = result * pad_mask

    return result


def time_warp(X: Tensor, fraction: float = 0.5, max_sigma: float = 1.0, min_sigma: float = 0.5,
              n_knots: int = 4, pad_mask: Tensor = None) -> Tensor:

    if pad_mask is None:
        pad_mask = torch.ones(X.shape, device=X.device)

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device, dtype=torch.float32)
    random_warps = np.random.normal(loc=1.0, scale=1.0, size=(n_aug, n_knots + 2))
    random_sigma = np.random.uniform(min_sigma, max_sigma, size=(n_aug, 1))
    random_warps = (random_warps - 1) * random_sigma + 1
    warp_steps = np.linspace(0, n_timesteps - 1., num=n_knots + 2).T

    result = X.clone()
    warp_steps = torch.tensor(warp_steps, device=X.device, dtype=torch.float32)
    random_warps = torch.tensor(random_warps, device=X.device, dtype=torch.float32)

    time_warp = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, (warp_steps * random_warps).unsqueeze(-1))
                                   ).evaluate(orig_steps).squeeze(-1)
    pad_idx_mask = torch.zeros_like(X, device=X.device, dtype=torch.int64)
    pad_idx_mask[torch.arange(X.shape[0], device=X.device, dtype=torch.int64), pad_mask.sum(dim=-1).to(torch.int64)] = 1
    warped_pad_idx = (time_warp * pad_idx_mask).sum(dim=-1, keepdim=True).to(torch.int64).clip(0,n_timesteps)
    warped_mask = (torch.arange(n_timesteps, device=X.device).repeat(n_aug,1) < warped_pad_idx).float()
    time_warp = (((n_timesteps-1)/time_warp[:,-1:])*time_warp).clip(0, n_timesteps-1)

    result[idx_aug] = interp1d(orig_steps, X[idx_aug], time_warp) * warped_mask

    return result



def window_warp(X: Tensor, fraction: float = 0.5,
                min_window_ratio: float = 0.05, max_window_ratio: float = 0.1,
                min_window_count: int = 5, max_window_count: int = 10,
                sigma: float = 0.5,
                pad_mask: Tensor = None) -> Tensor:

    if pad_mask is None:
        pad_mask = torch.ones(X.shape, device=X.device)

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device, dtype=torch.float32)
    random_warps = torch.ones((n_aug, n_timesteps), device=X.device, dtype=torch.float32)

    window_count_per_sample = torch.randint(min_window_count, max_window_count, (n_aug,) , device=X.device)
    totals = window_count_per_sample.sum()
    max_window_size = int(n_timesteps * max_window_ratio)
    min_window_size = int(n_timesteps * min_window_ratio)
    window_starts = torch.randint(0, n_timesteps-max_window_size, (totals,), device=X.device)
    window_sizes = torch.randint(min_window_size, max_window_size, (totals,), device=X.device)
    max_size = window_sizes.max().item()
    arange_mask = torch.arange(max_size, device=X.device).unsqueeze(0) < window_sizes.unsqueeze(1)
    aranges = torch.arange(max_size, device=X.device).unsqueeze(0).expand(window_sizes.size(0), -1)[arange_mask]
    window_indices = window_starts.repeat_interleave(window_sizes) + aranges
    window_sample_tensor = torch.arange(n_aug, device=X.device).repeat_interleave(window_count_per_sample).repeat_interleave(window_sizes)

    random_values = torch.normal(1.0, sigma, (totals,), device=X.device).clip(0.0, None)
    random_values = random_values.repeat_interleave(window_sizes)
    random_warps[window_sample_tensor, window_indices] = random_values

    result = X.clone()
    time_warp = torch.cumsum(random_warps, dim=1)


    pad_idx_mask = torch.zeros_like(X, device=X.device, dtype=torch.int64)
    pad_idx_mask[torch.arange(X.shape[0], device=X.device, dtype=torch.int64), pad_mask.sum(dim=-1).to(torch.int64)] = 1
    warped_pad_idx = (time_warp * pad_idx_mask).sum(dim=-1, keepdim=True).to(torch.int64).clip(0,n_timesteps)
    warped_mask = (torch.arange(n_timesteps, device=X.device).repeat(n_aug,1) < warped_pad_idx).float()

    time_warp = (((n_timesteps-1)/time_warp[:,-1:])*time_warp).clip(0, n_timesteps-1)
    result[idx_aug] = interp1d(orig_steps, X[idx_aug], time_warp) * warped_mask

    ## Do not use pad_mask in time warp

    return result
