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
        self.bq_embedding = nn.Embedding(max_bq+1, d_model)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_bq: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        bq_embedding = self.bq_embedding(src_bq)

        return kmer_embedding, signal_embedding, bq_embedding

    ## END OF TransformerModel
