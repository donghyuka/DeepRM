import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch.nn import ModuleList
import copy
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

class TransformerEncoderLayerWithWeight(nn.TransformerEncoderLayer):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        super(TransformerEncoderLayerWithWeight, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                                                                layer_norm_eps, batch_first, norm_first, bias,
                                                                device, dtype)


    def _sa_block(self, x: Tensor,
                  attn_mask, key_padding_mask, is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        self.attention_weight = x[1]
        return self.dropout1(x[0])


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                    )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x, self.attention_weight

    ## END OF TransformerEncoderLayerWithWeight


class TransformerEncoderWithWeight(nn.TransformerEncoder):
    def __init__(self, encoder_layer: TransformerEncoderLayerWithWeight, num_layers: int, norm_first: bool = False) -> None:
        super().__init__(encoder_layer, num_layers)
        self.norm_first = norm_first

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tuple[Tensor, Any]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
              and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        attn_list = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            attn_list.append(attn)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_list



class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, encoder_dropout: float = 0.1, lin_dropout: float = 0.1,
                 kmer_size: int = 5, signal_size: int = 25, spectrogram_size: int = 21, max_bq: int = 40, block_len = 17,
                 seq_len: int = 200, t_act : str = 'gelu', lin_act : str = 'relu', lin_depth: int = 1) -> None:
        super().__init__()

        ## Embedding Initialization
        self.kmer_count = 4**kmer_size+1
        self.kmer_embedding = nn.Embedding(self.kmer_count*(max_bq+1), d_model)
        self.signal_embedding = nn.Linear(signal_size, d_model)
        self.embedding_dropout = nn.Dropout(encoder_dropout)
        self.pos_encoding = PositionalEncoding(d_model//2, seq_len)

        ## Encoder Initialization
        self.d_model = d_model
        self.model_type = 'Transformer'
        encoder_layer = TransformerEncoderLayerWithWeight(d_model, n_heads, d_ff, dropout = encoder_dropout, activation = t_act,
                                                            batch_first=True)

        self.transformer_encoder_1 = TransformerEncoderWithWeight(encoder_layer, n_layers)
        self.transformer_encoder_2 = TransformerEncoderWithWeight(encoder_layer, n_layers)
        self.cross_attention_layer = nn.MultiheadAttention(d_model, n_heads, dropout = encoder_dropout, batch_first = True)
        self.layer_norm = nn.LayerNorm(d_model)

        ## Regression Head Initialization
        self.regression_head = RegressionHead(d_model, lin_act, lin_depth, lin_dropout, seq_len)
        self.regression_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.regression_head)
        self.div_term = torch.exp(torch.arange(0, self.d_model // 2, 2) * (-math.log(10000.0) / self.d_model // 2))
        self.max_bq = max_bq

        self.max_bq = max_bq

        ## Weight Initialization
        self.init_weights()

    def init_weights(self, initrange = 0.1):
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)
        self.pos_encoding.pe.data.uniform_(-initrange, initrange)
        return None


    def forward(self, src_kmer: Tensor, src_signal: Tensor, src_move: Tensor,
                src_pad_mask: Tensor, target_mask: Tensor) -> Tuple[Tensor, Any]:

        kmer_embedding = self.kmer_embedding(src_kmer)
        signal_embedding = self.signal_embedding(src_signal)
        pos_encoding = self.pos_encoding(signal_embedding)
        div_term = self.div_term.to(src_kmer.device)

        src_move = src_move.unsqueeze(-1)
        mv_embedding = torch.zeros(src_move.shape[0], src_move.shape[1], self.d_model // 2).to(src_move.device)
        mv_embedding[:, :, 0::2] = torch.sin(src_move * div_term)
        mv_embedding[:, :, 1::2] = torch.cos(src_move * div_term)
        pos_encoding = torch.cat([pos_encoding, mv_embedding], dim = -1)

        ## add all embeddings and dropout
        attn_list_list = []

        signal_output = torch.add(signal_embedding, pos_encoding)
        signal_output, attn_list = self.transformer_encoder_1(src=signal_output, mask = None, src_key_padding_mask = src_pad_mask)
        attn_list_list.extend(attn_list)
        kmer_output = torch.add(kmer_embedding, pos_encoding)
        kmer_output, attn_list = self.transformer_encoder_2(src=kmer_output, mask = None, src_key_padding_mask = src_pad_mask)
        attn_list_list.extend(attn_list)
        cross_output, attn = self.cross_attention_layer(signal_output, kmer_output, kmer_output, key_padding_mask = src_pad_mask,
                                                             need_weights = True, is_causal = False)
        attn_list_list.append(attn)
        cross_output = cross_output[0] + signal_output
        cross_output = self.layer_norm(cross_output)
        cross_output, attn_list = self.transformer_encoder_3(src=cross_output, mask = None, src_key_padding_mask = src_pad_mask)
        attn_list_list.extend(attn_list)

        attn_list = attn_list_list

        ## apply regression head to each token:
        output = self.regression_head(cross_output)
        output = output.squeeze(-1)

        target_mask_sum = target_mask.sum(dim = 1)
        output = output * target_mask
        output = output.sum(dim = 1)
        output = output / target_mask_sum

        output = torch.sigmoid(output)

        return output, attn_list

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

