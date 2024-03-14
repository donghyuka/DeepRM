import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 40, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1) -> None:
        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model//4)
        self.signal_embedding = nn.Linear(signal_size, d_model//4)
        self.spectrogram_embedding = nn.Linear(spectrogram_size, d_model//4)
        self.bq_embedding = nn.Embedding(max_bq+1, d_model//4)
        self.pos_encoding = PositionalEncoding(d_model)
        self.move_embedding = nn.Embedding(block_len+1, d_model)
        self.embedding_dropout = nn.Dropout(encoder_dropout)

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

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.spectrogram_embedding.weight.data.uniform_(-initrange, initrange)
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.move_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.pos_encoding.pe.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_spectrogram: Tensor, src_bq: Tensor, src_move: Tensor,
                src_pad_mask: Tensor, target_mask: Tensor) -> Tensor:

        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        spectrogram_embedding = self.spectrogram_embedding(src_spectrogram)
        bq_embedding = self.bq_embedding(src_bq)
        pos_encoding = self.pos_encoding(torch.zeros_like(kmer_embedding))
        move_embedding = self.move_embedding(src_move)

        ## add all embeddings and dropout
        final_embedding = torch.concat([kmer_embedding, signal_embedding, spectrogram_embedding, bq_embedding], dim = -1)
        final_embedding = final_embedding + pos_encoding + move_embedding
        final_embedding = self.embedding_dropout(final_embedding)

        output = self.transformer_encoder(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)

        if target_mask is not None:
            ## match target_mask (l) to model output shape (l,d) by repeating the mask
            target_mask = target_mask.unsqueeze(-1).repeat(1,1,self.d_model)
            output = output * target_mask

        ## apply regression head to each token:
        output = self.regression_head(output)
        output = output.mean(dim = 1).squeeze(-1)

        return output

    ## END OF TransformerModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
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
        x = self.pe[:x.size(0)]
        return x

    ## END OF PositionalEncoding


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, lin_act: str, lin_depth: int, lin_dropout: float, seq_length: int):
        super().__init__()
        self.lin_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(lin_depth)])
        self.activation = self._get_activation_fn(lin_act)
        self.dropout = nn.Dropout(lin_dropout)
        self.batch_norm = nn.BatchNorm1d(seq_length)
        self.semi_final_layer = nn.Linear(d_model, d_model//4)
        self.final_layer = nn.Linear(d_model//4, 1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.lin_layers:
            x = layer(x)
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.semi_final_layer(x)
        x = self.activation(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, initrange = 0.1):
        for layer in self.lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()
        self.final_layer.weight.data.uniform_(-initrange, initrange)
        self.final_layer.bias.data.zero_()

    def _get_activation_fn(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    ## END OF RegressionHead

