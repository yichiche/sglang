# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import logging
import os

import aiter
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)

_use_fp8_attn = os.environ.get("SGLANG_AITER_FP8_ATTN", "0") == "1"
_fp8_dtype = torch.float8_e4m3fn

if _use_fp8_attn:
    logger.info("DiT FP8 attention enabled via SGLANG_AITER_FP8_ATTN=1")


class AITerBackend(AttentionBackend):
    """
    Backend for AITemplate attention implementation.
    """

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER

    @staticmethod
    def get_impl_cls() -> type["AITerImpl"]:
        return AITerImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITer backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError("AITer backend does not have a metadata builder.")


class AITerImpl(AttentionImpl):
    """
    Implementation of attention using AITemplate.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITer backend does not support Grouped Query Attention yet."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using aiter.flash_attn_func (BF16) or
        aiter.flash_attn_fp8_pertensor_func (FP8).

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        if _use_fp8_attn:
            if query.dtype != _fp8_dtype:
                q_fp8, q_scale = aiter.per_tensor_quant(query, quant_dtype=_fp8_dtype)
                k_fp8, k_scale = aiter.per_tensor_quant(key, quant_dtype=_fp8_dtype)
                v_fp8, v_scale = aiter.per_tensor_quant(value, quant_dtype=_fp8_dtype)
            else:
                q_fp8, k_fp8, v_fp8 = query, key, value
                one_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
                q_scale = k_scale = v_scale = one_scale

            output = aiter.flash_attn_fp8_pertensor_func(
                q_fp8,
                k_fp8,
                v_fp8,
                q_descale=q_scale,
                k_descale=k_scale,
                v_descale=v_scale,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
            return output

        # BF16 path (default)
        output, _ = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p,
            causal=self.causal,
            return_attn_probs=False,
            return_lse=True,
        )
        return output
