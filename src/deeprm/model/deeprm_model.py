"""
Module: deeprm.model.deeprm_model
This module defines the DeepRM model architecture, including the ResNet block,
Transformer model, positional encoding, and regression head.
"""

import math

from deeprm.utils import check_deps

check_deps.check_torch_available()


import torch  # noqa
from torch import Tensor, nn  # noqa

from deeprm.utils.activations import get_activation_fn  # noqa


class ResNetBlock(nn.Module):
    """
    A 1D ResNet block for 1D DeepRM.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels. If None, set to out_channels. (optional)
        kernel_size (int): Kernel size for the middle convolutional layer. Default is 3. (optional)
        stride (int): Stride for the convolutional layers. Default is 1. (optional)
        activation (str): Activation function to use. Default is 'gelu'. (optional)
        dropout (float): Dropout rate. Default is 0.1. (optional)
        groups (int): Number of groups for grouped convolution. Default is 1.  (optional)

    Attributes:
        bn1 (torch.nn.BatchNorm1d): Batch normalization layer for the first convolution.
        activation (typing.Callable): Activation function.
        conv1 (torch.nn.Conv1d): First convolutional layer with kernel size 1.
        bn2 (torch.nn.BatchNorm1d): Batch normalization layer for the second convolution.
        conv2 (torch.nn.Conv1d): Second convolutional layer with specified kernel size and groups.
        bn3 (torch.nn.BatchNorm1d): Batch normalization layer for the third convolution.
        dropout (torch.nn.Dropout): Dropout layer.
        conv3 (torch.nn.Conv1d): Third convolutional layer with kernel size 1.
        shortcut (torch.nn.Module): Shortcut connection to match input and output dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = get_activation_fn(activation)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=stride, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, padding="same", groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=stride, padding="same")

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ResNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out += residual
        return out


class TransformerModel(nn.Module):
    """
    A Transformer model for DeepRM.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        n_layers (int): Number of encoder layers.
        encoder_dropout (float): Dropout rate for the encoder. Default is 0.1. (optional)
        lin_dropout (float): Dropout rate for the linear layers. Default is 0.1. (optional)
        kmer_size (int): Size of the k-mer. Default is 5. (optional)
        signal_size (int): Size of the signal input. Default is 25. (optional)
        block_len (int): Length of the block. Default is 17. (optional)
        seq_len (int): Length of the sequence. Default is 200. (optional)
        t_act (str): Activation function for the transformer. Default is 'gelu'. (optional)
        lin_act (str): Activation function for the linear layers. Default is 'relu'. (optional)
        lin_depth (int): Depth of the linear layers. Default is 1. (optional)
        signal_stride (int): Stride for the signal input. Default is 6. (optional)
        **kwargs: Additional keyword arguments.

    Attributes:
        kmer_embedding (torch.nn.Embedding): Embedding layer for k-mer sequences.
        signal_embedding (torch.nn.Linear): Linear layer for signal input.
        pos_encoding (PositionalEncoding): Positional encoding layer.
        cnn_encoder (torch.nn.Sequential): Sequential container for CNN encoder blocks.
        transformer_encoder (torch.nn.TransformerEncoder): Transformer encoder.
        regression_head (RegressionHead): Regression head for the model output.
        d_model (int): Dimension of the model.
        model_type (str): Type of the model, set to 'Transformer'.
        kmer_size (int): Size of the k-mer.
        signal_stride (int): Stride for the signal input.
        unit_size (int): Size of the unit for processing sequences.
        target_start_idx (int): Start index for the target in the sequence.
        target_end_idx (int): End index for the target in the sequence.
        seq_len (int): Length of the sequence.
        block_len (int): Length of the block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        encoder_dropout: float = 0.1,
        lin_dropout: float = 0.1,
        kmer_size: int = 5,
        signal_size: int = 25,
        block_len=17,
        seq_len: int = 200,
        t_act: str = "gelu",
        lin_act: str = "relu",
        lin_depth: int = 1,
        signal_stride=6,
        **kwargs,
    ) -> None:

        super().__init__()

        ## Embedding Initialization
        self.kmer_embedding = nn.Embedding(4**kmer_size + 1, d_model)
        self.signal_embedding = nn.Linear(signal_size + 3, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = "Transformer"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=encoder_dropout, activation=t_act, batch_first=True
        )

        self.cnn_encoder = nn.Sequential()
        self.cnn_encoder.add_module(
            "resnet_1",
            ResNetBlock(d_model, d_model, kernel_size=5, stride=1, groups=8, activation=t_act, dropout=encoder_dropout),
        )
        self.cnn_encoder.add_module(
            "resnet_2",
            ResNetBlock(
                d_model, d_model, kernel_size=15, stride=1, groups=8, activation=t_act, dropout=encoder_dropout
            ),
        )
        self.cnn_encoder.add_module(
            "resnet_3",
            ResNetBlock(d_model, d_model, kernel_size=5, stride=1, groups=8, activation=t_act, dropout=encoder_dropout),
        )
        self.cnn_encoder.add_module(
            "resnet_4",
            ResNetBlock(
                d_model, d_model, kernel_size=15, stride=1, groups=8, activation=t_act, dropout=encoder_dropout
            ),
        )

        self.cnn_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.cnn_encoder)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, lin_act, lin_depth, lin_dropout, seq_len)
        self.regression_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.regression_head)

        ## Weight Initialization
        self.init_weights()

        self.kmer_size = kmer_size
        self.signal_stride = signal_stride
        self.unit_size = int((seq_len + kmer_size - 1) / block_len)
        self.target_start_idx = (block_len // 2) * self.unit_size - (kmer_size // 2)
        self.target_end_idx = self.target_start_idx + self.unit_size
        self.seq_len = seq_len
        self.block_len = block_len

    def init_weights(self, initrange=0.1):
        """
        Initialize the weights of the model.

        Args:
            initrange (float): Range for uniform initialization of weights. Default is 0.1. (optional)

        Returns:
            None
        """
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        return None

    def process_kmer(self, src_kmer: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        """
        Process the k-mer input to convert nucleotide characters to numerical indices.

        Args:
            src_kmer (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing nucleotide characters.
            src_seg_len_flat (torch.Tensor): Flattened segment lengths for the input sequences.

        Returns:
            torch.Tensor: Processed k-mer tensor of shape (batch_size, seq_len) with numerical indices.
        """
        batch = src_kmer.shape[0]
        src_kmer = (src_kmer - 65).clip(None, 8) % 5  ## Convert ACGTU to 01233.
        src_kmer = src_kmer.unfold(1, self.kmer_size, 1)
        src_kmer = src_kmer * (4 ** torch.arange(self.kmer_size, device=src_kmer.device, dtype=torch.int)).unsqueeze(
            0
        ).unsqueeze(0)
        src_kmer = src_kmer.sum(dim=-1) + 1
        src_kmer = torch.cat([src_kmer, torch.zeros(batch, 1, device=src_kmer.device, dtype=torch.int)], dim=1)
        src_kmer = src_kmer.flatten()
        src_kmer = src_kmer.repeat_interleave(src_seg_len_flat)
        src_kmer = src_kmer.reshape(batch, self.seq_len)
        src_kmer = src_kmer.int()
        return src_kmer

    def process_signal(self, src_signal: Tensor) -> Tensor:
        """
        Process the signal input by unfolding it into segments based on the signal stride and k-mer size.

        Args:
            src_signal (torch.Tensor): Input tensor of shape (batch_size, seq_len, signal_size) containing signal data.

        Returns:
            torch.Tensor: Processed signal tensor of shape (batch_size, new_seq_len, signal_size) after unfolding.
        """
        src_signal = src_signal.unfold(1, self.signal_stride * self.kmer_size, self.signal_stride)
        return src_signal

    def flatten_seg_len(self, src_seg_len: Tensor) -> Tensor:
        """
        Flatten the segment lengths to create a single dimension for each sequence.

        Args:
            src_seg_len (torch.Tensor): Input tensor of shape (batch_size, num_segments) containing segment lengths.

        Returns:
            torch.Tensor: Flattened segment lengths of shape (batch_size, seq_len).
        """
        src_seg_len_flat = torch.cat([src_seg_len, self.seq_len - src_seg_len.sum(dim=1, keepdims=True)], dim=1)
        src_seg_len_flat = src_seg_len_flat.flatten()
        return src_seg_len_flat

    def create_src_pad_mask(self, src_signal: Tensor, src_seg_len: Tensor) -> Tensor:
        """
        Create a padding mask for the source signal to ignore padded values during processing.

        Args:
            src_signal (torch.Tensor): Input tensor of shape (batch_size, seq_len, signal_size) containing signal data.
            src_seg_len (torch.Tensor): Segment lengths tensor of shape (batch_size, num_segments).

        Returns:
            torch.Tensor: Padding mask of shape (batch_size, seq_len) where True indicates padded positions.
        """
        batch = src_signal.shape[0]
        src_pad_mask = torch.arange(self.seq_len, device=src_signal.device)
        src_pad_mask = src_pad_mask.repeat(batch, 1)
        src_pad_mask = src_pad_mask >= src_seg_len.sum(dim=1, keepdim=True)
        return src_pad_mask

    def create_target_mask(self, src_seg_len: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        """
        Create a target mask to identify the target positions in the sequence.

        Args:
            src_seg_len (torch.Tensor): Segment lengths tensor of shape (batch_size, num_segments).
            src_seg_len_flat (torch.Tensor): Flattened segment lengths tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Target mask of shape (batch_size, seq_len) where True indicates target positions.
        """
        batch = src_seg_len.shape[0]
        width = src_seg_len.shape[1]
        target_mask = torch.arange(width + 1, device=src_seg_len.device, dtype=torch.int)
        target_mask = target_mask == self.block_len // 2
        target_mask = target_mask.repeat(batch)
        target_mask = target_mask.repeat_interleave(src_seg_len_flat)
        target_mask = target_mask.reshape(batch, self.seq_len)
        target_mask = target_mask.int()
        return target_mask

    def process_dwell_bq(self, src_dwell_bq: Tensor, src_seg_len_flat: Tensor) -> Tensor:
        """
        Process the dwell time and base quality input by flattening and repeating it based on segment lengths.

        Args:
            src_dwell_bq (torch.Tensor): Input tensor of shape (batch_size, seq_len, channel)
                containing dwell time and base quality.
            src_seg_len_flat (torch.Tensor): Flattened segment lengths for the input sequences.

        Returns:
            torch.Tensor: Processed dwell time and base quality tensor of shape (batch_size, seq_len, channel).
        """
        batch = src_dwell_bq.shape[0]
        channel = src_dwell_bq.shape[2]
        src_dwell_bq = torch.cat(
            [src_dwell_bq, torch.zeros(batch, 1, channel, device=src_dwell_bq.device, dtype=torch.float32)], dim=1
        )
        src_dwell_bq = src_dwell_bq.flatten(end_dim=1)
        src_dwell_bq = src_dwell_bq.repeat_interleave(src_seg_len_flat, dim=0)
        src_dwell_bq = src_dwell_bq.reshape(batch, self.seq_len, channel)
        return src_dwell_bq

    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_seg_len: Tensor, src_dwell_bq: Tensor) -> Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            src_kmer (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing k-mer sequences.
            src_signal (torch.Tensor): Input tensor of shape (batch_size, seq_len, signal_size) containing signal data.
            src_seg_len (torch.Tensor): Segment lengths tensor of shape (batch_size, num_segments).
            src_dwell_bq (torch.Tensor): Input tensor of shape (batch_size, seq_len, channel)
                containing dwell time and base quality.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len) after processing through the model.
        """
        with torch.no_grad():
            src_seg_len_flat = self.flatten_seg_len(src_seg_len)
            src_kmer = self.process_kmer(src_kmer, src_seg_len_flat)
            src_signal = self.process_signal(src_signal)
            src_dwell_bq = self.process_dwell_bq(src_dwell_bq, src_seg_len_flat)
            src_pad_mask = self.create_src_pad_mask(src_signal, src_seg_len)
            target_mask = self.create_target_mask(src_seg_len, src_seg_len_flat)

        src_signal = torch.cat([src_signal, src_dwell_bq], dim=-1)
        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        pos_encoding = self.pos_encoding(src_kmer.shape[0])

        ## add all embeddings and dropout
        final_embedding = torch.stack([kmer_embedding, signal_embedding, pos_encoding], dim=0).sum(dim=0)
        final_embedding = final_embedding.permute(0, 2, 1)  # Change to (batch, feature, time)
        output = self.cnn_encoder(final_embedding)
        output = output.permute(0, 2, 1)  # Change back to (batch, time, feature)
        output = self.transformer_encoder(src=output, mask=None, src_key_padding_mask=src_pad_mask)

        ## apply regression head to each token:
        output = self.regression_head(output)
        output = output.squeeze(-1)

        target_mask_sum = target_mask.sum(dim=1)
        output = output * target_mask
        output = output.sum(dim=1)
        output = output / target_mask_sum

        output = torch.sigmoid(output)

        return output

    ## END OF TransformerModel


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.

    Args:
        d_model (int): Dimension of the model.
        seq_len (int): Length of the sequence.

    Attributes:
        pe (torch.Tensor): Positional encoding tensor of shape (1, seq_len, d_model).
    """

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, batch_size) -> Tensor:
        """
        Forward pass to repeat the positional encoding for the given batch size.

        Args:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Returns:
            Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = self.pe.repeat(batch_size, 1, 1)
        return pe

    ## END OF PositionalEncoding


class RegressionHead(nn.Module):
    """
    Regression head for the Transformer model.

    Args:
        d_model (int): Dimension of the model.
        lin_act (str): Activation function for the linear layers.
        lin_depth (int): Depth of the linear layers.
        lin_dropout (float): Dropout rate for the linear layers.
        seq_length (int): Length of the sequence.

    Attributes:
        lin_layers (torch.nn.Sequential): Sequential container for the linear layers.
    """

    def __init__(self, d_model: int, lin_act: str, lin_depth: int, lin_dropout: float, seq_length: int):
        super().__init__()
        layer_list = []
        for i in range(lin_depth - 1):
            layer_list.append(nn.Linear(d_model, d_model))
            layer_list.append(nn.BatchNorm1d(seq_length))
            layer_list.append(get_activation_fn(lin_act))
            layer_list.append(nn.Dropout(lin_dropout))

        layer_list.append(nn.Linear(d_model, d_model))
        layer_list.append(get_activation_fn(lin_act))
        layer_list.append(nn.Linear(d_model, d_model // 4))
        layer_list.append(get_activation_fn(lin_act))
        layer_list.append(nn.Linear(d_model // 4, 1))
        self.lin_layers = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the regression head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, 1) after processing through the linear layers.
        """
        return self.lin_layers(x)

    def init_weights(self, initrange=0.1):
        """
        Initialize the weights of the linear layers in the regression head.

        Args:
            initrange (float): Range for uniform initialization of weights. Default is 0.1.

        Returns:
            None
        """
        for layer in self.lin_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        return None

    ## END OF RegressionHead
