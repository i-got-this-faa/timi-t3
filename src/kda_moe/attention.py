"""KDA attention layer and global grouped-query attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .kda import ShortConv, compute_kda_params, kda_chunked, kda_recurrent, kda_fla


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class KDAAttention(nn.Module):
    """KDA attention layer with persistent ShortConv projections.

    Pipeline:
        x → ShortConv → Swish (q,k) / SiLU (v) → Linear projection
        → L2-norm (q,k) → compute_kda_params → kda_chunked → output gate → out

    All convolutions and linear projections are persistent nn.Module
    parameters — no allocations in forward beyond activations.
    """

    def __init__(
        self,
        d_model: int = 640,
        n_heads: int = 10,
        head_dim: int = 64,
        g_min: float = -5.0,
        chunk_size: int = 64,
        use_fla: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.g_min = g_min
        self.chunk_size = chunk_size
        self.use_fla = use_fla

        inner_dim = n_heads * head_dim
        decay_inner = d_model // 4

        # ShortConv per projection (persistent, trainable)
        self.conv_q = ShortConv(d_model)
        self.conv_k = ShortConv(d_model)
        self.conv_v = ShortConv(d_model)

        # Linear projections
        self.W_q = nn.Linear(d_model, inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, inner_dim, bias=False)
        self.W_beta = nn.Linear(d_model, n_heads, bias=False)
        self.W_g = nn.Linear(d_model, n_heads, bias=False)

        # Decay projections
        self.W_ad = nn.Linear(d_model, decay_inner, bias=False)
        self.W_au = nn.Linear(decay_inner, inner_dim, bias=False)
        self.A_h = nn.Parameter(torch.empty(n_heads))

        # Output projection
        self.W_o = nn.Linear(inner_dim, d_model, bias=False)
        self.norm = RMSNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        for mod in [self.W_q, self.W_k, self.W_v, self.W_beta, self.W_g,
                     self.W_ad, self.W_au, self.W_o]:
            nn.init.normal_(mod.weight, std=std)
        nn.init.normal_(self.A_h, std=0.1)

    def forward(self, x: Tensor, use_recurrent: bool = False) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)
            use_recurrent: if True, use slow recurrent path for correctness checks

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        q, k, v, beta, log_decay, g = compute_kda_params(
            x,
            self.conv_q, self.conv_k, self.conv_v,
            self.W_q, self.W_k, self.W_v,
            self.W_beta, self.W_g,
            self.W_ad, self.W_au, self.A_h,
            self.n_heads, self.head_dim,
            g_min=self.g_min,
        )

        if use_recurrent:
            o, _ = kda_recurrent(q, k, v, beta, log_decay, g)
        elif self.use_fla:
            o = kda_fla(q, k, v, beta, log_decay, g)
        else:
            o = kda_chunked(q, k, v, beta, log_decay, g, chunk_size=self.chunk_size)

        # o: (B, H, T, D) → (B, T, inner_dim)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)

        # Output gate: mean over heads, sigmoid, scale
        g_out = g.transpose(1, 2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        o = torch.sigmoid(g_out) * self.norm(self.W_o(o))

        return o


class GlobalGQA(nn.Module):
    """Standard grouped-query attention with NoPE.

    No positional encoding — relies on KDA layers for positional information.
    """

    def __init__(
        self,
        d_model: int = 640,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        head_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // n_kv_heads

        inner_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        self.W_q = nn.Linear(d_model, inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, kv_dim, bias=False)
        self.W_v = nn.Linear(d_model, kv_dim, bias=False)
        self.W_o = nn.Linear(inner_dim, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for GQA
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, T, self.head_dim)
            k = k.reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, T, self.head_dim)
            v = v.reshape(B, self.n_heads, T, self.head_dim)

        # Scaled dot-product attention (NoPE, causal)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=attn.dtype), diagonal=1)
        attn = attn.masked_fill(causal.bool(), float("-inf"))
        attn_w = F.softmax(attn, dim=-1)

        o = torch.matmul(attn_w, v)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)

        return self.norm(self.W_o(o))
