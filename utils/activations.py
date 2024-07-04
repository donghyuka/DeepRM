import torch
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