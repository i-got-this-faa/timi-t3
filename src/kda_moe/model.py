"""Full KDA-MoE transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .attention import KDAAttention, GlobalGQA, RMSNorm
from .fused import fused_cross_entropy
from .moe import LatentMoE


def _activation(name: str):
    if name == "gelu":
        return F.gelu
    return F.silu


class DenseFFN(nn.Module):
    """Standard dense feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        self.act = _activation(activation)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """One transformer layer with pre-norm and optional gradient checkpointing.

    Layer schedule (plan.md §2.4):
      Layer 0:             KDA + dense FFN (warm-up)
      Layers 1,2,4-6,8-10,12-14:  KDA + MoE
      Layers 3,7,11,15:    global GQA + MoE
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # ── attention ──
        is_global = layer_idx in config.global_attn_layers
        eps = config.rms_norm_eps
        if config.use_kda and not is_global:
            self.attn = KDAAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                g_min=config.kda_g_min,
                chunk_size=config.kda_chunk_size,
                use_fla=config.kda_use_fla,
                eps=eps,
                std=config.initializer_std,
            )
        else:
            self.attn = GlobalGQA(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads if config.n_kv_heads else config.n_heads,
                head_dim=config.head_dim,
                eps=eps,
                use_sdpa=config.use_sdpa,
                std=config.initializer_std,
            )

        # ── FFN ──
        ffn_mult = config.ffn_multiplier
        self.has_ffn = True
        if not config.use_moe:
            # Dense baseline: all layers have dense FFN
            self.ffn: nn.Module = DenseFFN(
                config.d_model,
                int(config.d_model * ffn_mult),
                activation=config.activation,
            )
        elif layer_idx == 0:
            # Layer 0: dense warmup FFN
            self.ffn = DenseFFN(
                config.d_model,
                int(config.d_model * ffn_mult),
                activation=config.activation,
            )
        else:
            # All other layers (KDA + global) use the MoE FFN
            self.ffn = LatentMoE(
                d_model=config.d_model,
                latent_dim=config.latent_dim,
                n_experts=config.n_experts,
                top_k=config.top_k,
                expert_hidden=config.expert_hidden,
                shared_expert_hidden=config.shared_expert_hidden,
                aux_free_bias_update=config.aux_free_bias_update
                / max(1, config.grad_accum_steps),
                z_loss_coeff=config.z_loss_coeff,
                eps=eps,
                std=config.initializer_std,
            )

        # ── norms ──
        self.norm1 = RMSNorm(config.d_model, eps=eps, fused=config.fused_rmsnorm)
        self.norm2 = (
            RMSNorm(config.d_model, eps=eps, fused=config.fused_rmsnorm)
            if self.has_ffn
            else None
        )

        # ── dropout (0 by default) ──
        self.attn_drop = nn.Dropout(config.attention_dropout or config.dropout)
        self.ffn_drop = nn.Dropout(config.ffn_dropout or config.dropout)
        self.residual_drop = nn.Dropout(config.residual_dropout)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # Attention sublayer
        residual = x
        x = self.attn(self.norm1(x))
        x = self.attn_drop(x)
        x = residual + x
        x = self.residual_drop(x)

        # FFN sublayer
        if self.has_ffn:
            residual = x
            x = self.ffn(self.norm2(x))  # type: ignore[operator]
            x = self.ffn_drop(x)
            x = residual + x
            x = self.residual_drop(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.config.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class KDAMoEModel(nn.Module):
    """Full KDA-MoE transformer.

    - Token embedding (tied with LM head)
    - Stack of TransformerBlock layers
    - LM head (linear, tied weights with embedding)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)

        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps, fused=config.fused_rmsnorm)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) token indices

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

    def loss_fn(
        self,
        logits: Tensor,
        targets: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        """Cross-entropy loss with ignore_index, plus any MoE z-loss."""
        B, T, V = logits.shape
        if self.config.fused_cross_entropy:
            loss = fused_cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=ignore_index,
            )
        else:
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=ignore_index,
            )
        for layer in self.layers:
            ffn = getattr(layer, "ffn", None)
            z_loss = getattr(ffn, "last_z_loss", None)
            if z_loss is not None and ffn.z_loss_coeff:
                loss = loss + ffn.z_loss_coeff * z_loss
        return loss

    def accumulate_router_bias(self) -> None:
        """Accumulate this micro-batch's load for the aux-free bias update.

        Call after each micro-batch forward, outside the gradient-checkpointed
        region. No-op in eval mode.
        """
        if not self.training:
            return
        for layer in self.layers:
            ffn = getattr(layer, "ffn", None)
            if isinstance(ffn, LatentMoE):
                ffn.accumulate_bias()

    def apply_router_bias(self) -> None:
        """Apply accumulated aux-free bias updates; call once per optimizer step."""
        for layer in self.layers:
            ffn = getattr(layer, "ffn", None)
            if isinstance(ffn, LatentMoE):
                ffn.apply_bias()

    def get_num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
