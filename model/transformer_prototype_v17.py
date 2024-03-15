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
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.spectrogram_embedding = nn.Linear(spectrogram_size, d_model)
        self.bq_embedding = nn.Embedding(max_bq+1, d_model)
        self.move_embedding = nn.Embedding(block_len+1, d_model)
        self.embedding_dropout = nn.Dropout(encoder_dropout)
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
        move_embedding = self.move_embedding(src_move)
        pos_encoding = self.pos_encoding(kmer_embedding)

        ## add all embeddings and dropout
        final_embedding = torch.stack([kmer_embedding, signal_embedding, spectrogram_embedding, bq_embedding,
                                       pos_encoding, move_embedding], dim = 0).sum(dim = 0)
        final_embedding = nn.LayerNorm(final_embedding.size()[1:], elementwise_affine=False)(final_embedding)
        final_embedding = self.embedding_dropout(final_embedding)
        output = self.transformer_encoder(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)

        ## apply regression head to each token:
        output = self.regression_head(output)
        output = output.squeeze(-1)

        target_mask_sum = target_mask.sum(dim = 1)
        output = output * target_mask
        output = output.sum(dim = 1)
        output = output / target_mask_sum
        
        output = torch.clamp(output, 0, 1)

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

    def forward(self, x) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_size = x.size(0)
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
            layer_list.append(self._get_activation_fn(lin_act))
            layer_list.append(nn.Dropout(lin_dropout))

        layer_list.append(nn.Linear(d_model, d_model))
        layer_list.append(self._get_activation_fn(lin_act))
        layer_list.append(nn.Linear(d_model, d_model//4))
        layer_list.append(self._get_activation_fn(lin_act))
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

    def _get_activation_fn(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    ## END OF RegressionHead

