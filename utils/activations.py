import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(torch.nn.Module):
    """
    SwiGLU activation function module.

    This module splits the input tensor into two halves along the last dimension,
    applies the SiLU activation function to the second half, and then multiplies
    the two halves element-wise.

    Methods:
        forward(x): Applies the SwiGLU activation function to the input tensor.
    """
    def forward(self, x):
        """
        Applies the SwiGLU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, *, 2*C) where * means any number of additional dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (N, *, C).
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Swish(torch.nn.Module):
    """
    Swish activation function module.

    This module applies the Swish activation function, which is defined as x * sigmoid(beta * x),
    where beta is a learnable parameter.

    Attributes:
        beta (torch.nn.Parameter): Learnable parameter for the Swish activation function.

    Methods:
        forward(x): Applies the Swish activation function to the input tensor.
    """
    def __init__(self):
        """
        Initializes the Swish activation function module.
        """
        super().__init__()
        self.beta = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        """
        Applies the Swish activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Swish activation function.
        """
        return x * F.sigmoid(self.beta * x)

def get_activation_fn(activation: str):
    """
    Returns the activation function module based on the given activation name.

    Args:
        activation (str): Name of the activation function. Supported values are "relu", "gelu", "swish", "swiglu", "silu", and "elu".

    Returns:
        torch.nn.Module: Activation function module.

    Raises:
        RuntimeError: If the given activation function name is not supported.
    """
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