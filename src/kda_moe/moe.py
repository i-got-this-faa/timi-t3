"""LatentMoE: mixture-of-experts with latent-space routing, SiTUGLU, sigmoid router."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def softcap(x: Tensor, cap: float) -> Tensor:
    return cap * torch.tanh(x / cap)


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.proj = nn.Linear(dim, hidden_dim * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        g, u = self.proj(x).chunk(2, dim=-1)
        return softcap(g, 4.0) * torch.sigmoid(g) * softcap(u, 25.0)


class SigmoidRouter(nn.Module):
    def __init__(
        self, d_model: int, n_experts: int, top_k: int = 4, aux_free_bias_update: float = 0.001
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_free_bias_update = aux_free_bias_update
        self.weight = nn.Parameter(torch.empty(d_model, n_experts))
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = x @ self.weight
        scores = torch.sigmoid(logits + self.expert_bias)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return topk_indices, topk_scores, logits

    def update_bias(self, expert_indices: Tensor):
        with torch.no_grad():
            N = expert_indices.numel()
            counts = torch.bincount(expert_indices.flatten(), minlength=self.n_experts).float()
            avg = N / self.n_experts
            self.expert_bias -= self.aux_free_bias_update * (counts - avg).sign()


class LatentMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 640,
        latent_dim: int = 320,
        n_experts: int = 96,
        top_k: int = 4,
        expert_hidden: int = 660,
        shared_expert_hidden: int = 640,
        aux_free_bias_update: float = 0.001,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.top_k = top_k

        half_h = expert_hidden // 2
        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.norm_down = RMSNorm(latent_dim)
        self.router = SigmoidRouter(latent_dim, n_experts, top_k, aux_free_bias_update)
        self.expert_gate = nn.Parameter(torch.empty(n_experts, latent_dim, half_h))
        self.expert_up = nn.Parameter(torch.empty(n_experts, latent_dim, half_h))
        self.expert_down = nn.Parameter(torch.empty(n_experts, half_h, latent_dim))
        self.shared_up = nn.Linear(latent_dim, shared_expert_hidden, bias=False)
        self.shared_down = nn.Linear(shared_expert_hidden, latent_dim, bias=False)
        self.norm_out = RMSNorm(latent_dim)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.expert_gate, std=0.02)
        nn.init.normal_(self.expert_up, std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        N = B * T
        h = self.norm_down(self.down_proj(x))
        h_flat = h.view(N, self.latent_dim)
        expert_idx, expert_w, _ = self.router(h_flat)
        self.router.update_bias(expert_idx)

        expert_out = torch.zeros(N, self.latent_dim, device=x.device, dtype=x.dtype)

        for k_idx in range(self.top_k):
            idx_k = expert_idx[:, k_idx]
            w_k = expert_w[:, k_idx].unsqueeze(-1)
            for eid in idx_k.unique():
                mask = idx_k == eid
                h_e = h_flat[mask]
                gate = h_e @ self.expert_gate[eid]
                up = h_e @ self.expert_up[eid]
                act = softcap(gate, 4.0) * torch.sigmoid(gate) * softcap(up, 25.0)
                out_e = act @ self.expert_down[eid]
                expert_out[mask] += w_k[mask] * out_e

        shared_out = self.shared_down(F.silu(self.shared_up(h_flat)))
        combined = self.norm_out(expert_out + shared_out)
        return self.up_proj(combined.view(B, T, self.latent_dim))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)
