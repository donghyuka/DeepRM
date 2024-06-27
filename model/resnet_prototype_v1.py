import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

def _get_activation_fn(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim = -1)
    else:
        raise ValueError(f"Activation function {activation} not supported.")


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int = 1, padding = 'same',
                 dropout: float = 0.1,activation: str = "silu", initrange: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        self.init_weights(initrange)

    def init_weights(self, initrange: float):
        self.conv1.weight.data.uniform_(-initrange, initrange)
        self.conv2.weight.data.uniform_(-initrange, initrange)
        if len(self.shortcut) > 0:
            self.shortcut[0].weight.data.uniform_(-initrange, initrange)
        return None

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out




class ResNetModel(nn.Module):

    def __init__(self, n_layers_kmer: int, n_layers_sig: int, n_layers_merged: int, n_layers_final: int,
                 d_kmer: int, d_signal: int, d_merged: int, d_final: int,
                 dropout: float = 0.1, kernel_size: int = 5, kmer_size: int = 5, act : str = 'silu') -> None:
        super().__init__()

        act_func = _get_activation_fn(act)

        self.start_module_kmer = [nn.Conv1d(4*kmer_size, d_kmer, kernel_size, stride = 1, padding = 'same'),
                                     nn.BatchNorm1d(d_kmer), act_func]

        self.residual_module_kmer = []
        for _ in range(n_layers_kmer):
            self.residual_module_kmer.append(ResidualBlock(kernel_size, d_kmer, d_kmer, stride = 1, padding = 'same',
                                                           activation = act, dropout = dropout))

        self.start_module_signal = [nn.Conv1d(3, d_kmer, kernel_size, stride = 1, padding = 'same'),
                                    nn.BatchNorm1d(d_kmer), act_func]

        self.residual_module_signal = []
        for _ in range(n_layers_sig):
            self.residual_module_signal.append(ResidualBlock(kernel_size, d_signal, d_signal, stride = 1,
                                                             padding = 'same', activation = act, dropout = dropout))

        self.residual_module_merged = [ResidualBlock(kernel_size, d_signal+d_kmer, d_merged, stride = 1,
                                                     padding = 'same', activation = act, dropout = dropout)]

        for _ in range(n_layers_merged-1):
            self.residual_module_merged.append(ResidualBlock(kernel_size, d_merged, d_merged, stride = 1,
                                                             padding = 'same', activation = act, dropout = dropout))


        self.regression_head = [nn.Linear(d_merged, d_final), act_func]
        current_dim = d_final
        for _ in range(n_layers_final-2):
            next_dim = current_dim//2
            self.regression_head.append(nn.Linear(current_dim, next_dim))
            self.regression_head.append(act_func)
            current_dim = next_dim

        self.regression_head.append(nn.Linear(current_dim, 1))
        self.regression_head.append(nn.Linear(1, 1))

        self.start_module_kmer = nn.Sequential(*self.start_module_kmer)
        self.start_module_signal = nn.Sequential(*self.start_module_signal)
        self.residual_module_kmer = nn.Sequential(*self.residual_module_kmer)
        self.residual_module_signal = nn.Sequential(*self.residual_module_signal)
        self.residual_module_merged = nn.Sequential(*self.residual_module_merged)
        self.regression_head = nn.Sequential(*self.regression_head)

        self.max_bq = 40.0
        self.max_move = 17.0


    def init_weights(self, initrange = 0.1):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.weight.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_bq: Tensor, src_move: Tensor, target_mask: Tensor) -> Tensor:

        pad_mask = (src_move > 0).float().unsqueeze(1)
        src_kmer = torch.flatten(src_kmer, start_dim = -2).permute(0, 2, 1)
        src_bq = src_bq / self.max_bq
        src_move = src_move / self.max_move
        src_signal = torch.stack([src_signal, src_bq, src_move], dim = 1)

        src_kmer = self.start_module_kmer(src_kmer) * pad_mask
        src_signal = self.start_module_signal(src_signal) * pad_mask

        kmer_out = self.residual_module_kmer(src_kmer)
        signal_out = self.residual_module_signal(src_signal)
        merged_out = torch.cat([kmer_out, signal_out], dim = 1) * pad_mask
        merged_out = self.residual_module_merged(merged_out) * pad_mask
        merged_out = merged_out.permute(0, 2, 1)
        output = self.regression_head(merged_out).squeeze(-1)

        pad_mask = pad_mask.squeeze(1)
        pad_mask_sum = pad_mask.sum(dim = 1)
        output = output * pad_mask
        output = output.sum(dim = 1)
        output = output / pad_mask_sum
        output = torch.sigmoid(output)
        output = torch.clamp(output, min = 0.0, max = 1.0)
        output[torch.isnan(output)] = 0.0

        return output

    ## END OF TransformerModel
