import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_ff: int,
                 d_kmer_embedding: int, d_signal_embedding: int, d_spectrogram_embedding: int, d_bq_embedding: int,
                 d_pos_embedding: int,
                 nlayers: int, dropout: float = 0.1, kmer_size: int = 5, signal_size: int = 5, spectrogram_size: int = 20,
                 t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1) -> None:
        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size, d_kmer_embedding)
        self.signal_embedding = nn.Linear(signal_size, d_signal_embedding)
        self.spectrogram_embedding = nn.Linear(spectrogram_size, d_spectrogram_embedding)
        self.bq_embedding = nn.Linear(1, d_bq_embedding)


        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout = dropout, activation = t_act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.final_linear = self.build_linear(d_model, lin_act, lin_depth)

        self.init_weights()


    def cosine_pos_encoder(self, d_pos_embedding, d_model):
        encoder = torch.zeros(1000, d_model)
        position = torch.arange(0, 1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pos_embedding, 2) * -(math.log(10000.0) / d_pos_embedding))
        encoder[:, 0::2] = torch.sin(position * div_term)
        encoder[:, 1::2] = torch.cos(position * div_term)
        return encoder

    def build_linear(self, d_model, lin_act, lin_depth):
        if lin_act == 'relu':
            activation = nn.ReLU()
        elif lin_act == 'gelu':
            activation = nn.GELU()
        elif lin_act == 'tanh':
            activation = nn.Tanh()
        else:
            raise ValueError(f"Activation function {lin_act} not supported")
        layers = []
        for i in range(lin_depth):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(activation)
        layers.append(nn.Linear(d_model, 1))
        layers.append(nn.Sigmoid())
        layers = nn.Sequential(*layers)
        return layers

    def init_weights(self) -> None:
        initrange = 0.1
        self.bq_embedding.bias.data.zero_()
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

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
        output = self.linear(output)
        return output
