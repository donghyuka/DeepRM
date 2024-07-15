import torch
from typing import Union

def hard_to_soft_mask(t: torch.Tensor, sigma_factor: Union[float,int] = 1.0) -> torch.Tensor:
    x = torch.arange(t.shape[1], device=t.device).unsqueeze(0).repeat(t.shape[0],1)
    t_sigma = t.sum(dim = 1, keepdim=True) * sigma_factor / 2
    t_mu = t_sigma + (x * t * torch.cat([torch.zeros(t.shape[0],1, device=t.device), t[:,:-1]], dim = 1).logical_not()).sum(dim = 1, keepdim=True)
    t2 = torch.exp(-((x - t_mu) ** 2) / (2 * t_sigma ** 2))
    return t2