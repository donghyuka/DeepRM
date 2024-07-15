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
from typing import Tuple


def jitter(X: Tensor, pad_mask: Tensor, seg_len: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 1.0, min_sigma: float = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
    ## Can be used for both jitter and spike augmentation


    gauss = torch.normal(mean, 1.0, X.shape, device=X.device).clip(-3, 3)
    random_sigma = torch.rand((X.shape[0],1), device=X.device) * (max_sigma - min_sigma) + min_sigma
    gauss = gauss * random_sigma

    mask = torch.rand((X.shape[0],1), device=X.device) < fraction
    gauss = gauss * mask
    dropout_mask = torch.rand(X.shape, device=X.device) < (1-dropout)
    gauss = gauss * dropout_mask
    result =  X + gauss
    result = result * pad_mask

    return result, pad_mask, seg_len


def step(X: Tensor, pad_mask: Tensor, seg_len: Tensor, fraction: float = 0.5, dropout = 0.0, mean: float = 0,
           max_sigma: float = 1.0, min_sigma: float = 0.5) -> Tuple[Tensor, Tensor, Tensor]:

    gauss = torch.normal(mean, 1.0, X.shape, device=X.device).clip(-3, 3)
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

    result = result * pad_mask

    return result, pad_mask, seg_len


def slope(X: Tensor, pad_mask: Tensor, seg_len: Tensor, fraction: float = 0.5,  magnitude: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:

    lengths = X.shape[-1] / pad_mask.sum(dim=-1, keepdim=True).clip(1, None)

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

    return result, pad_mask, seg_len


def drift(X: Tensor, pad_mask: Tensor, seg_len: Tensor,
          fraction: float = 0.5, max_sigma: float = 1.0, min_sigma: float = 0.5, n_knots: int = 4) -> Tuple[Tensor, Tensor, Tensor]:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]

    n_aug = np.random.binomial(n_data, fraction)

    if n_aug < 1:
        return X, pad_mask, seg_len

    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps =torch.normal(0.0, 1.0, (n_aug, n_knots + 2), device=X.device).clip(-3, 3)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = random_warps * random_sigma
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)

    result = X.clone()
    warper = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                ).evaluate(orig_steps).squeeze(-1)
    result[idx_aug] += warper
    result = result * pad_mask

    return result, pad_mask, seg_len


def moving_magnitude_warp(X: Tensor, pad_mask: Tensor, seg_len: Tensor, fraction: float = 0.5,
                          max_sigma: float = 1.0, min_sigma: float = 0.5,
                          n_knots: int = 4, window_size: int = 7) -> Tuple[Tensor, Tensor, Tensor]:

    assert window_size % 2 ==1, "Window size must be odd"

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    if n_aug < 1:
        return X, pad_mask, seg_len

    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps = torch.normal(0.0, 1.0, (n_aug, n_knots + 2), device=X.device)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = ((random_warps * random_sigma) + 1.0).clip(0.01, 1.0 + 3.0 * max_sigma)
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)
    result = X.clone()

    movavg = torch.avg_pool1d(X[idx_aug], window_size, 1, count_include_pad=False)
    movavg = torch.cat([movavg[:,0:1].repeat(1, window_size//2),
                        movavg,
                        movavg[:,-1:].repeat(1, window_size//2)], dim=1)

    warper = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                ).evaluate(orig_steps).squeeze(-1).clip(0.0, 1.0 + 3.0 * max_sigma)


    diff = warper * (X[idx_aug] - movavg)
    result[idx_aug] += diff
    result = result * pad_mask

    return result, pad_mask, seg_len


def time_warp(X: Tensor, pad_mask: Tensor, seg_len: Tensor,
              fraction: float = 0.5, max_sigma: float = 1.0, min_sigma: float = 0.5,
              n_knots: int = 4) -> Tuple[Tensor, Tensor, Tensor]:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    if n_aug < 1:
        return X, pad_mask, seg_len


    orig_steps = torch.arange(n_timesteps, device=X.device)
    random_warps = torch.normal(0.0, 1.0, (n_aug, n_knots + 2), device=X.device)
    random_sigma = torch.rand(n_aug, 1, device=X.device) * (max_sigma - min_sigma) + min_sigma
    random_warps = ((random_warps * random_sigma) + 1.0).clip(0.001, 1.0 + 3.0 * max_sigma)
    warp_steps = torch.linspace(0, n_timesteps - 1., n_knots + 2, device=X.device)

    result = X.clone()
    time_warp = NaturalCubicSpline(natural_cubic_spline_coeffs(warp_steps, random_warps.unsqueeze(-1))
                                   ).evaluate(orig_steps).squeeze(-1).clip(0.001,None).cumsum(dim=1)

    pad_idx = pad_mask[idx_aug].sum(dim=-1).to(torch.int64) - 1
    time_warp = ((pad_idx/time_warp[torch.arange(n_aug, dtype=torch.int, device = X.device),pad_idx]).unsqueeze(-1)*time_warp).clip(0, n_timesteps-1)

    warped_seg_idx = seg_len[idx_aug].cumsum(dim=-1)
    warped_seg_idx = interp1d(time_warp, orig_steps.float(), warped_seg_idx).int()
    warped_seg_idx[:,1:] = warped_seg_idx[:,1:] - warped_seg_idx[:,:-1]
    warped_seg_idx = warped_seg_idx.clip(1, None)

    interp = interp1d(orig_steps, X[idx_aug], time_warp)
    result[idx_aug] = interp
    result = result * pad_mask

    seg_len = seg_len.clone()
    seg_len[idx_aug] = warped_seg_idx

    return result, pad_mask, seg_len



def window_warp(X: Tensor, pad_mask: Tensor, seg_len: Tensor, fraction: float = 0.5,
                min_window_ratio: float = 0.05, max_window_ratio: float = 0.1,
                min_window_count: int = 5, max_window_count: int = 10,
                sigma: float = 0.5) -> Tuple[Tensor, Tensor, Tensor]:

    n_data = X.shape[0]
    n_timesteps = X.shape[1]
    n_aug = np.random.binomial(n_data, fraction)
    idx_aug = np.random.choice(n_data, n_aug, replace=False)

    if n_aug < 1:
        return X, pad_mask, seg_len

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

    random_values = torch.normal(1.0, sigma, (totals,), device=X.device).clip(0.001, 1.0+3.0 * sigma)
    random_values = random_values.repeat_interleave(window_sizes)
    random_warps[window_sample_tensor, window_indices] = random_values

    time_warp = torch.cumsum(random_warps, dim=1)

    pad_idx = pad_mask[idx_aug].sum(dim=-1).to(torch.int64) - 1
    time_warp = ((pad_idx/time_warp[torch.arange(n_aug, dtype=torch.int, device = X.device),pad_idx]).unsqueeze(-1)*time_warp).clip(0, n_timesteps-1)
    interp = interp1d(orig_steps, X[idx_aug], time_warp)

    result = X.clone()
    result[idx_aug] = interp
    result = result * pad_mask

    warped_seg_idx = seg_len[idx_aug].cumsum(dim=-1)
    warped_seg_idx = interp1d(time_warp, orig_steps.float(), warped_seg_idx).int()
    warped_seg_idx[:,1:] = warped_seg_idx[:,1:] - warped_seg_idx[:,:-1]
    warped_seg_idx = warped_seg_idx.clip(1, None)
    seg_len = seg_len.clone()
    seg_len[idx_aug] = warped_seg_idx

    return result, pad_mask, seg_len
