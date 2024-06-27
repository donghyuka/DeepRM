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

class MaskedSequential(nn.Sequential):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        for module in self._modules.values():
            input = module(input, mask)
        return input

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int = 1, padding = 'same',
                 dropout: float = 0.1,activation: str = "silu", initrange: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels//4)
        self.bn3 = nn.BatchNorm1d(in_channels//4)
        self.conv1 = nn.Conv1d(in_channels, in_channels//4, 1, stride, padding)
        self.conv2 = nn.Conv1d(in_channels//4, in_channels//4, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(in_channels//4, out_channels, 1, stride, padding)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
                self.activation
            )
        self.init_weights(initrange)

    def init_weights(self, initrange: float):
        self.conv1.weight.data.uniform_(-initrange, initrange)
        self.conv2.weight.data.uniform_(-initrange, initrange)
        self.conv3.weight.data.uniform_(-initrange, initrange)
        if len(self.shortcut) > 0:
            self.shortcut[0].weight.data.uniform_(-initrange, initrange)
        return None

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.dropout(out)
        out += self.shortcut(x) * pad_mask
        return out




class ResNetModel(nn.Module):

    def __init__(self, n_layers_kmer: int, n_layers_sig: int, n_layers_merged: int, n_layers_final: int,
                 d_kmer: int, d_signal: int, d_merged: int, d_final: int,
                 dropout: float = 0.1, kernel_size: int = 5, kmer_size: int = 5, act : str = 'silu') -> None:
        super().__init__()

        act_func = _get_activation_fn(act)

        self.start_module_kmer = [nn.AvgPool1d(5,5,0),
                                  nn.Conv1d(4*kmer_size, d_kmer, 1, stride = 1, padding = 'same'),
                                  nn.BatchNorm1d(d_kmer), act_func,
                                  nn.Conv1d(d_kmer, d_kmer, 1, stride = 1, padding = 'same'),
                                  nn.BatchNorm1d(d_kmer)]

        self.residual_module_kmer = []
        for _ in range(n_layers_kmer):
            self.residual_module_kmer.append(ResidualBlock(kernel_size, d_kmer, d_kmer, stride = 1, padding = 'same',
                                                           activation = act, dropout = dropout))


        self.start_module_signal = [nn.Conv1d(2, d_signal, 1, stride = 1, padding = 'same'),
                                    nn.BatchNorm1d(d_signal), act_func,
                                    nn.Conv1d(d_signal, d_signal, 1, stride = 1, padding = 'same'),
                                    nn.BatchNorm1d(d_signal)]

        self.residual_module_signal_1 = []
        for _ in range(n_layers_sig):
            self.residual_module_signal_1.append(ResidualBlock(kernel_size, d_signal, d_signal, stride = 1,
                                                             padding = 'same', activation = act, dropout = dropout))

        self.end_module_signal = [nn.Conv1d(d_signal, d_signal, 5, stride = 5, padding = 'valid'),
                                  nn.BatchNorm1d(d_signal), act_func,
                                  nn.Conv1d(d_signal, d_signal, 1, stride = 1, padding = 'same'),
                                  nn.BatchNorm1d(d_signal)]


        self.residual_module_signal_2 = []
        for _ in range(n_layers_sig):
            self.residual_module_signal_2.append(ResidualBlock(kernel_size, d_signal, d_signal, stride = 1,
                                                             padding = 'same', activation = act, dropout = dropout))


        self.residual_module_merged = [ResidualBlock(kernel_size, d_signal+d_kmer, d_merged, stride = 1,
                                                       padding = 'same', activation = act, dropout = dropout),]

        for _ in range(n_layers_merged-1):
            self.residual_module_merged.append(ResidualBlock(kernel_size, d_merged, d_merged, stride = 1,
                                                               padding = 'same', activation = act, dropout = dropout))


        self.regression_head_1 = [nn.Linear(d_merged, d_merged), act_func]
        current_dim = d_merged

        for _ in range(2):
            next_dim = current_dim//4
            self.regression_head_1.append(nn.Linear(current_dim, next_dim))
            self.regression_head_1.append(act_func)
            current_dim = next_dim

        current_dim = current_dim * 200
        self.regression_head_2 = [nn.Linear(current_dim, d_final), act_func]
        current_dim = d_final

        for _ in range(n_layers_final-2):
            self.regression_head_2.append(nn.Linear(current_dim, current_dim))
            self.regression_head_2.append(act_func)

        for _ in range(2):
            next_dim = current_dim//4
            self.regression_head_2.append(nn.Linear(current_dim, next_dim))
            self.regression_head_2.append(act_func)
            current_dim = next_dim

        self.regression_head_2.append(nn.Linear(current_dim, 1))
        self.regression_head_2.append(nn.Linear(1, 1))

        self.start_module_kmer = nn.Sequential(*self.start_module_kmer)
        self.start_module_signal = nn.Sequential(*self.start_module_signal)
        self.end_module_signal = nn.Sequential(*self.end_module_signal)
        self.residual_module_kmer = MaskedSequential(*self.residual_module_kmer)
        self.residual_module_signal_1 = MaskedSequential(*self.residual_module_signal_1)
        self.residual_module_signal_2 = MaskedSequential(*self.residual_module_signal_2)
        self.residual_module_merged = MaskedSequential(*self.residual_module_merged)
        self.regression_head_1 = nn.Sequential(*self.regression_head_1)
        self.regression_head_2 = nn.Sequential(*self.regression_head_2)

        self.max_bq = 40.0
        self.max_move = 17.0

        ## Convert to SyncBatchNorm
        self.start_module_kmer = nn.SyncBatchNorm.convert_sync_batchnorm(self.start_module_kmer)
        self.start_module_signal = nn.SyncBatchNorm.convert_sync_batchnorm(self.start_module_signal)
        self.end_module_signal = nn.SyncBatchNorm.convert_sync_batchnorm(self.end_module_signal)
        self.residual_module_kmer = nn.SyncBatchNorm.convert_sync_batchnorm(self.residual_module_kmer)
        self.residual_module_signal_1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.residual_module_signal_1)
        self.residual_module_signal_2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.residual_module_signal_2)
        self.residual_module_merged = nn.SyncBatchNorm.convert_sync_batchnorm(self.residual_module_merged)



    def init_weights(self, initrange = 0.1):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.weight.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_bq: Tensor, src_move: Tensor, target_mask: Tensor) -> Tensor:

        pad_mask = (src_move > 0).float().unsqueeze(1)
        src_kmer = torch.flatten(src_kmer, start_dim = -2).permute(0, 2, 1)
        src_move = src_move / self.max_move
        src_signal = torch.stack([src_signal, src_move], dim = 1)

        signal_out = self.start_module_signal(src_signal) * pad_mask
        signal_out = self.residual_module_signal_1(signal_out, pad_mask)
        pad_mask = torch.nn.AvgPool1d(5,5,0)(pad_mask)
        signal_out = self.end_module_signal(signal_out) * pad_mask
        signal_out = self.residual_module_signal_2(signal_out, pad_mask)

        kmer_out = self.start_module_kmer(src_kmer) * pad_mask
        kmer_out = self.residual_module_kmer(kmer_out, pad_mask)

        merged_out = torch.cat([kmer_out, signal_out], dim = 1) * pad_mask
        merged_out = self.residual_module_merged(merged_out, pad_mask)
        merged_out = merged_out.permute(0, 2, 1)

        pad_mask = pad_mask.squeeze(1).unsqueeze(-1)
        output = self.regression_head_1(merged_out) * pad_mask
        output = torch.flatten(output, start_dim = 1)
        output = self.regression_head_2(output)
        output = output.squeeze(-1)
        output = torch.sigmoid(output)
        output = torch.clamp(output, min = 0.0, max = 1.0)
        output[torch.isnan(output)] = 0.0

        return output

    ## END OF TransformerModel
