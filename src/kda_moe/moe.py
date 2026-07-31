"""LatentMoE with grouped dispatch — sort tokens by expert, batch matmul per group."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def softcap(x: Tensor, cap: float) -> Tensor:
    return cap * torch.tanh(x / cap)


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
        self.register_buffer("_bias_updates", torch.zeros(n_experts))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = x @ self.weight
        scores = torch.sigmoid(logits + self.expert_bias)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return topk_indices, topk_scores, logits

    def accumulate_bias(self, expert_indices: Tensor):
        """Accumulate one micro-batch's load-balancing update (deferred to step end)."""
        with torch.no_grad():
            N = expert_indices.numel()
            counts = torch.bincount(expert_indices.flatten(), minlength=self.n_experts).float()
            avg = N / self.n_experts
            self._bias_updates += (counts - avg).sign()

    def apply_bias(self):
        """Apply accumulated updates to expert_bias; called once per optimizer step.

        expert_bias must stay constant across a step so that gradient-checkpoint
        recompute reproduces the same router decisions (in-place/step-end updates
        would otherwise produce different token->expert grouping on recompute).
        """
        with torch.no_grad():
            self.expert_bias -= self.aux_free_bias_update * self._bias_updates
            self._bias_updates.zero_()

    def router_entropy(self, logits: Tensor) -> float:
        """Compute entropy of router probability distribution."""
        probs = torch.sigmoid(logits)
        p_avg = probs.mean(dim=0)
        p_avg = p_avg.clamp_min(1e-10)
        return float(-(p_avg * p_avg.log()).sum().item())


class LatentMoE(nn.Module):
    """MoE with grouped expert dispatch — tokens sorted by expert, batched matmul per group."""

    def __init__(
        self,
        d_model: int = 640,
        latent_dim: int = 320,
        n_experts: int = 96,
        top_k: int = 4,
        expert_hidden: int = 660,
        shared_expert_hidden: int = 640,
        aux_free_bias_update: float = 0.001,
        z_loss_coeff: float = 0.0,
        eps: float = 1e-6,
        std: float = 0.02,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.half_h = expert_hidden // 2
        self.z_loss_coeff = z_loss_coeff
        self.register_buffer("last_router_logits", torch.zeros(0))
        self.last_z_loss: Tensor | None = None

        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.norm_down = RMSNorm(latent_dim, eps=eps)
        self.router = SigmoidRouter(latent_dim, n_experts, top_k, aux_free_bias_update)

        # Expert weights: (E, l, h/2) for gate/up, (E, h/2, l) for down
        self.expert_gate = nn.Parameter(torch.empty(n_experts, latent_dim, self.half_h))
        self.expert_up = nn.Parameter(torch.empty(n_experts, latent_dim, self.half_h))
        self.expert_down = nn.Parameter(torch.empty(n_experts, self.half_h, latent_dim))

        self.shared_up = nn.Linear(latent_dim, shared_expert_hidden, bias=False)
        self.shared_down = nn.Linear(shared_expert_hidden, latent_dim, bias=False)
        self.norm_out = RMSNorm(latent_dim, eps=eps)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.std = std
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.expert_gate, std=self.std)
        nn.init.normal_(self.expert_up, std=self.std)
        nn.init.normal_(self.expert_down, std=self.std)

    def _dispatch_one_slot(self, h_flat: Tensor, idx_k: Tensor, w_k: Tensor) -> Tensor:
        """Grouped dispatch for one top-k slot.

        Sorts tokens by expert, does one batched matmul per contiguous expert group.
        Returns (N, latent_dim) expert output for this slot.
        """
        N, l = h_flat.shape
        device = h_flat.device
        dtype = h_flat.dtype

        # Sort tokens by expert ID
        sort_order = idx_k.argsort()
        sorted_h = h_flat[sort_order]  # (N, l)
        sorted_w = w_k[sort_order]  # (N,)
        sorted_eid = idx_k[sort_order]  # (N,)

        # Pre-allocate output, scatter back later
        slot_out = torch.zeros(N, l, device=device, dtype=dtype)

        # Find boundaries between expert groups
        # Use a single GPU kernel: diff then cumsum
        boundaries = torch.where(sorted_eid[1:] != sorted_eid[:-1])[0] + 1
        starts = torch.cat([torch.tensor([0], device=device), boundaries])
        ends = torch.cat([boundaries, torch.tensor([N], device=device)])

        for g in range(len(starts)):
            s, e = starts[g].item(), ends[g].item()
            if s >= e:
                continue
            eid = sorted_eid[s].item()
            h_g = sorted_h[s:e]  # (n_g, l)
            w_g = sorted_w[s:e].unsqueeze(-1)  # (n_g, 1)

            # Expert FFN: gate + SiTUGLU + down
            gate = h_g @ self.expert_gate[eid]  # (n_g, h/2)
            up = h_g @ self.expert_up[eid]
            act = (softcap(gate, 4.0) * torch.sigmoid(gate) * softcap(up, 25.0)).to(dtype)
            out = (act.float() @ self.expert_down[eid]).to(dtype)  # (n_g, l)

            # Scatter back
            orig_pos = sort_order[s:e]
            slot_out[orig_pos] = (w_g * out).to(dtype)

        return slot_out

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        N = B * T
        device = x.device

        # Down-project
        h = self.norm_down(self.down_proj(x)).view(N, self.latent_dim)
        expert_idx, expert_w, router_logits = self.router(h)
        self.last_router_logits = router_logits.detach()
        self.last_expert_idx = expert_idx.detach()
        self.last_z_loss = (
            (torch.logsumexp(router_logits, dim=-1) ** 2).mean()
            if self.z_loss_coeff
            else None
        )

        # Dispatch each top-k slot
        out = torch.zeros(N, self.latent_dim, device=device, dtype=x.dtype)
        for k_idx in range(self.top_k):
            out += self._dispatch_one_slot(
                h,
                expert_idx[:, k_idx],
                expert_w[:, k_idx],
            )

        # Shared expert + output
        shared = self.shared_down(F.silu(self.shared_up(h)))
        return self.up_proj(self.norm_out(out + shared).view(B, T, self.latent_dim))

    def accumulate_bias(self) -> None:
        """Record the last micro-batch's load for the aux-free bias update.

        Called by the training loop after each micro-batch forward, outside the
        gradient-checkpointed region. The actual bias change is deferred to
        `apply_bias` at optimizer-step end so the checkpointed forward always
        sees a constant expert_bias (recompute-safe).
        """
        if not self.training or self.last_expert_idx is None:
            return
        if self.last_expert_idx.numel() == 0:
            return
        self.router.accumulate_bias(self.last_expert_idx)

    def apply_bias(self) -> None:
        """Apply the accumulated aux-free bias update (once per optimizer step)."""
        self.router.apply_bias()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)
