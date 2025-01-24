import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.activations import get_activation_fn


class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, max_bq: int = 40, block_len = 17, seq_len: int = 200,
                 t_act : str = 'gelu', lin_act : str = 'relu',
                 lin_depth: int = 1, signal_stride = 6, **kwargs) -> None:

        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.signal_embedding = nn.Linear(signal_size+3, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout,
                                                            activation = t_act,batch_first=True)
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

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        return None

    def process_kmer(self, src_kmer: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        batch = src_kmer.shape[0]
        src_kmer = (src_kmer - 65).clip(None,8)%5 ## Convert ACGTU to 01233.
        src_kmer = src_kmer.unfold(1, self.kmer_size, 1)
        src_kmer = src_kmer * (4**torch.arange(self.kmer_size, device = src_kmer.device, dtype = torch.int)).unsqueeze(0).unsqueeze(0)
        src_kmer = src_kmer.sum(dim = -1) + 1
        src_kmer = torch.cat([src_kmer, torch.zeros(batch, 1, device = src_kmer.device, dtype = torch.int)], dim = 1)
        src_kmer = src_kmer.flatten()
        src_kmer = src_kmer.repeat_interleave(src_seg_len_flat)
        src_kmer = src_kmer.reshape(batch, self.seq_len)
        src_kmer = src_kmer.int()
        return src_kmer

    def process_signal(self, src_signal: Tensor) -> Tensor:
        src_signal = src_signal.unfold(1, self.signal_stride * self.kmer_size, self.signal_stride)
        return src_signal

    def flatten_seg_len(self, src_seg_len: Tensor) -> Tensor:
        src_seg_len_flat = torch.cat([src_seg_len, self.seq_len - src_seg_len.sum(dim = 1, keepdims=True)], dim = 1)
        src_seg_len_flat = src_seg_len_flat.flatten()
        return src_seg_len_flat

    def create_src_pad_mask(self, src_signal: Tensor, src_seg_len: Tensor) -> Tensor:
        batch = src_signal.shape[0]
        src_pad_mask = torch.arange(self.seq_len, device=src_signal.device)
        src_pad_mask = src_pad_mask.repeat(batch, 1)
        src_pad_mask = src_pad_mask >= src_seg_len.sum(dim=1, keepdim=True)
        return src_pad_mask

    def create_target_mask(self, src_seg_len: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        batch = src_seg_len.shape[0]
        width = src_seg_len.shape[1]
        target_mask = torch.arange(width+1,device=src_seg_len.device, dtype = torch.int)
        target_mask = target_mask ==self.block_len//2
        target_mask = target_mask.repeat(batch)
        target_mask = target_mask.repeat_interleave(src_seg_len_flat)
        target_mask = target_mask.reshape(batch, self.seq_len)
        target_mask = target_mask.int()
        return target_mask

    def process_dwell_bq(self, src_dwell_bq: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        batch = src_dwell_bq.shape[0]
        channel = src_dwell_bq.shape[2]
        src_dwell_bq = torch.cat([src_dwell_bq, torch.zeros(batch, 1, channel, device = src_dwell_bq.device, dtype = torch.float32)], dim = 1)
        src_dwell_bq = src_dwell_bq.flatten(end_dim=1)
        src_dwell_bq = src_dwell_bq.repeat_interleave(src_seg_len_flat,dim=0)
        src_dwell_bq = src_dwell_bq.reshape(batch, self.seq_len, channel)
        return src_dwell_bq


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_seg_len: Tensor, src_dwell_bq: Tensor) -> Tensor:

        with torch.no_grad():
            src_seg_len_flat = self.flatten_seg_len(src_seg_len)
            src_kmer = self.process_kmer(src_kmer, src_seg_len_flat)
            src_signal = self.process_signal(src_signal)
            src_dwell_bq = self.process_dwell_bq(src_dwell_bq, src_seg_len_flat)
            src_pad_mask = self.create_src_pad_mask(src_signal, src_seg_len)
            target_mask = self.create_target_mask(src_seg_len, src_seg_len_flat)

        src_signal = torch.cat([src_signal, src_dwell_bq], dim = -1)
        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        pos_encoding = self.pos_encoding(src_kmer.shape[0])

        ## add all embeddings and dropout
        final_embedding = torch.stack([kmer_embedding, signal_embedding, pos_encoding], dim = 0).sum(dim = 0)
        output = self.transformer_encoder(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)

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
