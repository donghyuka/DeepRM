"""
Activation function utilities for DeepRM.
"""

from deeprm.utils import check_deps

check_deps.check_torch_available()
import torch.nn as nn


def get_activation_fn(activation: str):
    """
    Returns the activation function module based on the given activation name.

    Args:
        activation (str): Name of the activation function. Supported values are "relu", "gelu", "silu", and "elu".

    Returns:
        torch.nn.Module: Activation function module.

    Raises:
        RuntimeError: If the given activation function name is not supported.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"The following activation function is not supported: {activation}")
