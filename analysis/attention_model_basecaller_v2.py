import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from analysis.attention_model import TransformerEncoderWithWeight, TransformerEncoderLayerWithWeight

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 40, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1) -> None:
        super().__init__()

        ## Embedding Initialization
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = TransformerEncoderLayerWithWeight(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                             batch_first=True)
        self.transformer_encoder = TransformerEncoderWithWeight(encoder_layer, n_layers)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, 6, lin_act, lin_depth, lin_dropout, seq_len)
        self.regression_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.regression_head)

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.pos_encoding.pe.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_signal: Tensor, src_pad_mask: Tensor, src_target_mask: Tensor):

        signal_embedding = self.signal_embedding(src_signal)
        pos_encoding = self.pos_encoding(signal_embedding)

        ## add all embeddings and dropout
        final_embedding = torch.stack([signal_embedding, pos_encoding], dim = 0).sum(dim = 0)
        output, attn_list = self.transformer_encoder(src=final_embedding, mask = None, src_key_padding_mask = src_pad_mask)

        ## apply regression head to each token:
        output = self.regression_head(output)
        output_basecalling = torch.permute(output, (0, 2, 1))

        output_modification = torch.softmax(output, dim = -1)[:,:,-1]
        target_mask_sum = src_target_mask.sum(dim = 1)
        output_modification = output_modification * src_target_mask
        output_modification = output_modification.sum(dim = 1)
        output_modification = output_modification / target_mask_sum

        return output_modification, attn_list

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

