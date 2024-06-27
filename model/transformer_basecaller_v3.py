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
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        self.kmer_embedding = nn.Embedding(4**kmer_size+1, d_model)
        self.bq_embedding = nn.Embedding(max_bq+1, d_model)
        self.move_embedding = nn.Embedding(block_len+1, d_model)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                   batch_first=True)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, 5)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, 3)

        ## Regression Head Initialization
        self.basecall_regression_head = RegressionHead(d_model, 5, lin_act, lin_depth, lin_dropout, seq_len)
        self.modification_regression_head = RegressionHead(d_model, 1, lin_act, lin_depth, lin_dropout, seq_len)

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.pos_encoding.pe.data.uniform_(-initrange, initrange)
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.move_embedding.weight.data.uniform_(-initrange, initrange)
        self.basecall_regression_head.init_weights(initrange)
        self.modification_regression_head.init_weights(initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_bq: Tensor, src_move: Tensor,
                src_pad_mask: Tensor, src_target_mask: Tensor) -> Tuple[Tensor, Tensor]:
        ## Channel order: [m6A, U, G, C, cA, PAD]

        signal_embedding = self.signal_embedding(src_signal)
        pos_encoding = self.pos_encoding(signal_embedding)
        kmer_embedding = self.kmer_embedding(src_kmer)
        bq_embedding = self.bq_embedding(src_bq)
        move_embedding = self.move_embedding(src_move)

        ## add all embeddings and dropout
        final_embedding = torch.stack([signal_embedding, pos_encoding], dim = 0).sum(dim = 0)
        output_encoder = self.transformer_encoder_1(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)

        ## apply regression head to each token:
        output_basecalling = self.basecall_regression_head(output_encoder)
        output_basecalling = torch.permute(output_basecalling, (0, 2, 1))

        output_modification =  torch.stack([output_encoder, bq_embedding, kmer_embedding, move_embedding], dim = 0).sum(dim = 0)
        output_modification = self.transformer_encoder_2(src=output_modification, mask = None, src_key_padding_mask = src_pad_mask)
        output_modification = self.modification_regression_head(output_modification)
        output_modification = output_modification.squeeze(-1)
        target_mask_sum = src_target_mask.sum(dim = 1)
        output_modification = output_modification * src_target_mask
        output_modification = output_modification.sum(dim = 1)
        output_modification = output_modification / target_mask_sum
        output_modification = torch.sigmoid(output_modification)

        return output_basecalling, output_modification

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
    def __init__(self, d_model: int, d_out: int, lin_act: str, lin_depth: int, lin_dropout: float, seq_length: int):
        super().__init__()
        layer_list=  []

        for i in range(lin_depth-2):
            layer_list.append(nn.Linear(d_model, d_model))
            layer_list.append(self._get_activation_fn(lin_act))
            layer_list.append(nn.Dropout(lin_dropout))

        curr_dim = d_model
        for i in range(2):
            next_dim = curr_dim//4
            layer_list.append(nn.Linear(curr_dim, next_dim))
            layer_list.append(self._get_activation_fn(lin_act))
            curr_dim = next_dim

        layer_list.append(nn.Linear(curr_dim, d_out))

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

