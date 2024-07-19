import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.activations import get_activation_fn


## AIRNA_V4: From V1. Removed BQ and MOVE features. Added More Signal Embedding FFW Layers.

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 40, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1,
                 sig_emb_depth: int = 6) -> None:

        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, lin_act, lin_depth, lin_dropout, seq_len)

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.pos_encoding.pe.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor) -> Tensor:

        output = torch.stack([self.kmer_embedding(src_kmer), self.signal_embedding(src_signal), self.pos_encoding(src_kmer.size(0))], dim = 0).sum(dim = 0)
        output = self.transformer_encoder(src=output, mask = None, src_key_padding_mask = None)
        output = self.regression_head(output)

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

    def forward(self, batch_size:int) -> Tensor:
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

