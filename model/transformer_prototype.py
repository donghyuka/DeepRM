import math
import os
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from conv_module import Conv1DModel, Conv2DModel


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 d_kmer_embedding: int, d_signal_embedding: int, d_spectrogram_embedding: int,
                 nlayers: int, dropout: float = 0.5, kmer_size: int = 5):
        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = KmerEmbedding(kmer_size=kmer_size, dropout=dropout, d_embedding=d_kmer_embedding)
        self.signal_embedding = Conv1DModel(d_model=d_signal_embedding, dropout=dropout,)
        self.spectrogram_embedding = Conv2DModel(d_model=d_spectrogram_embedding, dropout=dropout)
        self.bq_embedding = nn.Linear(1, d_model)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.bq_embedding.bias.data.zero_()
        self.bq_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_spectrogram: Tensor, src_bq: Tensor, src_mask: Tensor = None) -> Tensor:
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

        final_embedding = kmer_embedding + signal_embedding + spectrogram_embedding + bq_embedding




        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class KmerEmbedding(nn.Module):
    ## Get batches of RNA sequence, which are kmer list.
    ## Convert into kmer embedding.

    def __init__(self, kmer_size: int = 5, dropout: float = 0.1, max_len: int = 5000, d_embedding: int = 128):
        super().__init__()
        self.kmer_size = kmer_size
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(4**kmer_size, d_embedding)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``
        """
        x = self.embedding(x)
        x = self.dropout(x)
        return x

