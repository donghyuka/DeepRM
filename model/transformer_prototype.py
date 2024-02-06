import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 d_kmer_embedding: int, d_signal_embedding: int, d_spectrogram_embedding: int, d_bq_embedding: int,
                 d_pos_encoding: int, n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 5, spectrogram_size: int = 20,
                 t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1) -> None:
        super().__init__()
        self.model_type = 'Transformer'

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size, d_kmer_embedding)
        self.signal_embedding = nn.Linear(signal_size, d_signal_embedding)
        self.spectrogram_embedding = nn.Linear(spectrogram_size, d_spectrogram_embedding)
        self.bq_embedding = nn.Linear(1, d_bq_embedding)
        self.pos_encoding = PositionalEncoding(d_pos_encoding, encoder_dropout)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        ## Regression Head Initialization
        self.regerssion_head = self.RegressionHead(d_model, lin_act, lin_depth, lin_dropout)

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.spectrogram_embedding.weight.data.uniform_(-initrange, initrange)
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.regerssion_head.init_weights(initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_spectrogram: Tensor, src_bq: Tensor,
                src_pad_mask: Tensor = None, target_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        spectrogram_embedding = self.spectrogram_embedding(src_spectrogram)
        bq_embedding = self.bq_embedding(src_bq)
        final_embedding = torch.cat([kmer_embedding, signal_embedding, spectrogram_embedding, bq_embedding], dim=2)

        output = self.transformer_encoder(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)
        if target_mask is not None:
            output = output * target_mask

        output = self.regerssion_head(output)

        return output

    ## END OF TransformerModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    ## END OF PositionalEncoding


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, lin_act: str, lin_depth: int, lin_dropout: float):
        super().__init__()
        self.lin_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(lin_depth)])
        self.activation = getattr(nn, lin_act)
        self.dropout = nn.Dropout(lin_dropout)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.lin_layers:
            x = layer(x)
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, initrange = 0.1):
        for layer in self.lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()
        self.final_layer.weight.data.uniform_(-initrange, initrange)
        self.final_layer.bias.data.zero_()

    ## END OF RegressionHead

