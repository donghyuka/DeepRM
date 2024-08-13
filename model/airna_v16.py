import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.activations import get_activation_fn
from utils.interp1d_v2 import Interp1d

## AIRNA_V4: From V1. Removed BQ and MOVE features. Added More Signal Embedding FFW Layers.

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 48, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1,
                 sig_emb_depth: int = 6, signal_stride = 6) -> None:

        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.pe_sig = PositionalEncoding(d_model, seq_len)
        self.pe_seq = PositionalEncoding(d_model, block_len)
        self.bq_embedding = nn.Embedding(max_bq+1, d_model)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                   batch_first=True)
        self.transformer_decoder_1 = nn.TransformerDecoder(decoder_layer, n_layers)
        self.transformer_decoder_2 = nn.TransformerDecoder(decoder_layer, n_layers)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, lin_act, lin_depth, lin_dropout, block_len)

        ## Weight Initialization
        self.init_weights()
        self.interp1d = Interp1d()
        self.kmer_size = kmer_size
        self.signal_stride = signal_stride
        self.unit_size = int((seq_len + kmer_size - 1) / block_len)
        self.target_start_idx = (block_len // 2) * self.unit_size - (kmer_size // 2)
        self.target_end_idx = self.target_start_idx + self.unit_size
        self.max_bq = max_bq

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_seg_len: Tensor, src_bq: Tensor) -> Tensor:
        with torch.no_grad():
            src_signal = self.interp1d(torch.arange(src_signal.shape[1], device=src_signal.device).unsqueeze(0).repeat(src_signal.shape[0],1), src_signal, (src_seg_len / self.unit_size).repeat_interleave(self.unit_size * self.signal_stride, dim=1).cumsum(dim=1)).unfold(1, self.signal_stride * self.kmer_size, self.signal_stride)
            src_kmer = (((src_kmer - 65).clip(None,8)%5).unfold(1, self.kmer_size, 1) * (4**torch.arange(self.kmer_size, device = src_kmer.device, dtype = torch.int)).unsqueeze(0).unsqueeze(0)).sum(dim = -1) + 1
            src_bq = src_bq.clip(0,self.max_bq)


        output = torch.stack([self.signal_embedding(src_signal), self.pe_sig(src_signal.size(0))], dim = 0).sum(dim = 0)
        output = self.transformer_encoder(src=output, mask = None, src_key_padding_mask = None)
        target = self.kmer_embedding(src_kmer) + self.pe_seq(src_kmer.size(0))
        target = self.transformer_decoder_1(tgt=target, memory=output, tgt_mask = None, memory_mask = None,
                                            tgt_key_padding_mask = None, memory_key_padding_mask = None)
        target = target + self.bq_embedding(src_bq)
        target = self.transformer_decoder_2(tgt=target, memory=output, tgt_mask = None, memory_mask = None,
                                            tgt_key_padding_mask = None, memory_key_padding_mask = None)

        output = self.regression_head(target)

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
        layer_list_1 =  []

        layer_list_1.append(nn.Linear(d_model, d_model))
        layer_list_1.append(get_activation_fn(lin_act))
        layer_list_1.append(nn.Dropout(lin_dropout))
        layer_list_1.append(nn.Linear(d_model, 128))
        layer_list_1.append(get_activation_fn(lin_act))
        layer_list_1.append(nn.Linear(128, 1))
        layer_list_1.append(get_activation_fn(lin_act))
        self.lin_layers_1 = nn.Sequential(*layer_list_1)

        layer_list_2 = []
        layer_list_2.append(nn.Linear(seq_length, 128))
        layer_list_2.append(get_activation_fn(lin_act))
        layer_list_2.append(nn.Linear(128, 32))
        layer_list_2.append(get_activation_fn(lin_act))
        layer_list_2.append(nn.Linear(32, 1))
        layer_list_2.append(nn.Sigmoid())
        self.lin_layers_2 = nn.Sequential(*layer_list_2)

    def forward(self, x: Tensor) -> Tensor:
        x =  self.lin_layers_1(x)
        x = x.squeeze(dim = 2)
        x = self.lin_layers_2(x)
        x = x.squeeze(dim = 1)
        return x

    def init_weights(self, initrange = 0.1):
        for layer in self.lin_layers_1:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        for layer in self.lin_layers_2:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        return None

    ## END OF RegressionHead
