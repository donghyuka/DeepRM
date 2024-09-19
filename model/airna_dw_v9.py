import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.activations import get_activation_fn


class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 40, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1,
                 sig_emb_depth: int = 6, signal_stride = 6) -> None:

        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.dwell_embedding = nn.Linear(2, d_model)
        self.joint_embedding = nn.Linear(3 * d_model, d_model)
        self.signal_embedding = SignalConv1d(seq_len, signal_stride, kmer_size, d_model,
                                             act = t_act, dropout = encoder_dropout)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, lin_act, lin_depth, lin_dropout, seq_len)
        self.regression_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.regression_head)

        ## Weight Initialization
        self.init_weights()

        self.kmer_size = kmer_size
        self.signal_stride = signal_stride
        self.unit_size = int((seq_len + kmer_size - 1) / block_len)
        self.target_start_idx = (block_len // 2) * self.unit_size - (kmer_size // 2)
        self.target_end_idx = self.target_start_idx + self.unit_size
        self.seq_len = seq_len
        self.max_bq = max_bq
        self.block_len = block_len
        self.activation = get_activation_fn(t_act)

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.dwell_embedding.weight.data.uniform_(-initrange, initrange)
        self.joint_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.signal_embedding.init_weights(initrange)
        return None

    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_seg_len: Tensor, src_dwell_bq: Tensor) -> Tensor:

        with torch.no_grad():
            src_seg_len_flat = torch.cat([src_seg_len, self.seq_len - src_seg_len.sum(dim = 1, keepdims=True)], dim = 1).flatten()
            src_kmer = ((src_kmer - 65).clip(None,8)%5).unfold(1, self.kmer_size, 1) ## This converts ACGTU to 01233 and unfolds to kmer_size.
            src_kmer = (src_kmer * (4**torch.arange(self.kmer_size, device = src_kmer.device, dtype = torch.int)).unsqueeze(0).unsqueeze(0)).sum(dim = -1) + 1
            src_kmer = torch.cat([src_kmer, torch.zeros(src_kmer.shape[0], 1, device = src_kmer.device, dtype = torch.int)], dim = 1).flatten().repeat_interleave(src_seg_len_flat).reshape(src_seg_len.shape[0], self.seq_len).int()
            src_pad_mask = torch.arange(self.seq_len, device=src_signal.device).repeat(src_signal.size(0), 1) >= src_seg_len.sum(dim=1, keepdim=True)
            src_dwell_bq = torch.cat([src_dwell_bq, torch.zeros(src_dwell_bq.shape[0], 1, src_dwell_bq.shape[2], device = src_dwell_bq.device, dtype = torch.float32)], dim = 1).flatten(end_dim=1).repeat_interleave(src_seg_len_flat,dim=0).reshape(src_dwell_bq.shape[0], self.seq_len, src_dwell_bq.shape[2])
            target_mask = (torch.arange(src_seg_len.shape[1]+1,device=src_seg_len.device, dtype = torch.int)==self.block_len//2)
            target_mask = target_mask.repeat(src_seg_len.shape[0]).repeat_interleave(src_seg_len_flat).reshape(src_seg_len.shape[0], self.seq_len).int()
            src_signal = src_signal.unsqueeze(1)

        kmer_embedding = self.kmer_embedding(src_kmer)
        dwell_embedding = self.dwell_embedding(src_dwell_bq)
        signal_embedding = self.signal_embedding(src_signal).permute(0,2,1)

        joint_embedding = torch.cat([kmer_embedding, dwell_embedding, signal_embedding], dim = -1)
        joint_embedding = self.activation(joint_embedding)
        joint_embedding = self.joint_embedding(joint_embedding)
        pos_encoding = self.pos_encoding(src_kmer.shape[0])
        ## add all embeddings and dropout
        joint_embedding = joint_embedding + pos_encoding
        output = self.transformer_encoder(src=joint_embedding, mask = None, src_key_padding_mask = src_pad_mask)

        ## apply regression head to each token:
        output = self.regression_head(output)
        output = output.squeeze(-1)

        target_mask_sum = target_mask.sum(dim = 1)
        output = output * target_mask
        output = output.sum(dim = 1)
        output = output / target_mask_sum

        output = torch.sigmoid(output)

        return output

    ## END OF TransformerModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, batch_size) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = self.pe.repeat(batch_size, 1, 1)
        return pe

    ## END OF PositionalEncoding


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, lin_act: str, lin_depth: int, lin_dropout: float, seq_length: int):
        super().__init__()
        layer_list=  []
        for i in range(lin_depth-1):
            layer_list.append(nn.Linear(d_model, d_model))
            layer_list.append(nn.BatchNorm1d(seq_length))
            layer_list.append(get_activation_fn(lin_act))
            layer_list.append(nn.Dropout(lin_dropout))

        layer_list.append(nn.Linear(d_model, d_model))
        layer_list.append(get_activation_fn(lin_act))
        layer_list.append(nn.Linear(d_model, d_model//4))
        layer_list.append(get_activation_fn(lin_act))
        layer_list.append(nn.Linear(d_model//4, 1))
        self.lin_layers = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin_layers(x)

    def init_weights(self, initrange = 0.1):
        for layer in self.lin_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        return None
    ## END OF RegressionHead


class SignalConv1d(nn.Module):
    def __init__(self, seq_len, signal_stride, kmer_size, d_model, act = "gelu", dropout = 0.1):
        super().__init__()
        self.input_size = (seq_len + kmer_size - 1) * signal_stride
        self.out_dim = d_model

        layer_list = []

        layer_list.append(nn.Conv1d(in_channels = 1, out_channels = d_model, kernel_size = 13,
                                    padding = "same"))
        layer_list.append(get_activation_fn(act))

        layer_list.append(nn.Conv1d(in_channels = d_model, out_channels = d_model, kernel_size = 5,
                                    padding = "valid"))
        layer_list.append(get_activation_fn(act))

        layer_list.append(nn.Conv1d(in_channels = d_model, out_channels = d_model, kernel_size = 5,
                                    padding = "valid"))
        layer_list.append(get_activation_fn(act))

        layer_list.append(nn.Conv1d(in_channels = d_model, out_channels = d_model, kernel_size = 5,
                                    padding = "valid"))
        layer_list.append(get_activation_fn(act))

        layer_list.append(nn.Conv1d(in_channels = d_model, out_channels = d_model, kernel_size = 5,
                                    padding = "valid"))
        layer_list.append(get_activation_fn(act))

        layer_list.append(nn.Conv1d(in_channels = d_model, out_channels = d_model, kernel_size = 13,
                                    padding = "valid", stride = signal_stride))
        layer_list.append(get_activation_fn(act))


        self.conv_layers = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return x

    def init_weights(self, initrange = 0.1):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        return None

    ## END OF SignalConv1d


