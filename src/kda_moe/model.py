"""Full KDA-MoE transformer model."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .attention import KDAAttention, GlobalGQA, RMSNorm
from .moe import LatentMoE


class DenseFFN(nn.Module):
    """Standard dense feed-forward network with SwiGLU activation."""

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """One transformer layer with pre-norm and optional gradient checkpointing.

    Layer schedule (from plan.md §2.4):
      Layer 0:         KDA + dense FFN
      Layers 1-2,4-6,8-10,12-14:  KDA + MoE
      Layers 3,7,11,15:  global GQA (no FFN in MoE mode)
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # ── attention ──
        is_global = layer_idx in config.global_attn_layers
        if config.use_kda and not is_global:
            self.attn = KDAAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                g_min=config.kda_g_min,
                chunk_size=config.kda_chunk_size,
                use_fla=config.kda_use_fla,
            )
        else:
            self.attn = GlobalGQA(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads if config.n_kv_heads else config.n_heads,
                head_dim=config.head_dim,
            )

        # ── FFN ──
        self.has_ffn = True
        if not config.use_moe:
            # Dense baseline: all layers have dense FFN
            self.ffn: nn.Module = DenseFFN(config.d_model, config.d_model * 4)
        elif layer_idx == 0:
            # Layer 0: dense warmup FFN
            self.ffn = DenseFFN(config.d_model, config.d_model * 4)
        elif is_global:
            # Global attention layers: no FFN in MoE mode
            self.ffn = nn.Identity()
            self.has_ffn = False
        else:
            self.ffn = LatentMoE(
                d_model=config.d_model,
                latent_dim=config.latent_dim,
                n_experts=config.n_experts,
                top_k=config.top_k,
                expert_hidden=config.expert_hidden,
                shared_expert_hidden=config.shared_expert_hidden,
                aux_free_bias_update=config.aux_free_bias_update,
            )

        # ── norms ──
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model) if self.has_ffn else None

        # Dropout (only for dense baseline)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Attention sublayer
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x

        # FFN sublayer
        if self.has_ffn:
            residual = x
            x = self.norm2(x)  # type: ignore[operator]
            x = self.ffn(x)
            x = self.dropout(x)
            x = residual + x

        return x


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

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

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
        """Cross-entropy loss with ignore_index."""
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=ignore_index,
        )

    def get_num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
