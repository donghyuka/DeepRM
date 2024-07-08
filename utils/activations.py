import torch
import torch.nn as nn
import torch.nn.functional as F

##

class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


def get_activation_fn(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return Swish()
    elif activation == "swiglu":
        return SwiGLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"The following activation function is not supported: {activation}")