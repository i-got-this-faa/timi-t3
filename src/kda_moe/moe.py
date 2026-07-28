"""LatentMoE: mixture-of-experts with latent-space routing, SiTUGLU, sigmoid router."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def softcap(x: Tensor, cap: float) -> Tensor:
    """Soft capping: cap * tanh(x / cap)."""
    return cap * torch.tanh(x / cap)


class SiTUGLU(nn.Module):
    """SiTU-GLU activation: softcap(g, 4) * sigmoid(g) * softcap(u, 25)
    where (g, u) = split(proj(x)).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.proj = nn.Linear(dim, hidden_dim * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        g, u = self.proj(x).chunk(2, dim=-1)
        return softcap(g, 4.0) * torch.sigmoid(g) * softcap(u, 25.0)


class SigmoidRouter(nn.Module):
    """Projects d_model→n_experts, sigmoid, top-k.

    Uses auxiliary-loss-free load balancing via per-expert bias.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 4,
        aux_free_bias_update: float = 0.001,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_free_bias_update = aux_free_bias_update

        self.weight = nn.Parameter(torch.empty(d_model, n_experts))
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            x: (N, d_model)

        Returns:
            expert_indices: (N, top_k)
            expert_weights: (N, top_k)
            router_logits: (N, n_experts)
        """
        logits = x @ self.weight
        scores = torch.sigmoid(logits + self.expert_bias)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return topk_indices, topk_scores, logits

    def update_bias(self, expert_indices: Tensor):
        """Auxiliary-loss-free load balancing.

        Decrements bias of overloaded experts, increments underloaded ones.
        """
        with torch.no_grad():
            N = expert_indices.numel()  # total assignments = N_tokens * top_k
            counts = torch.bincount(expert_indices.flatten(), minlength=self.n_experts).float()
            avg = N / self.n_experts
            self.expert_bias -= self.aux_free_bias_update * (counts - avg).sign()

    def router_entropy(self, logits: Tensor) -> float:
        """Compute average assignment entropy for monitoring."""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            return entropy.item()


class LatentMoE(nn.Module):
    """MoE with latent-space routing.

    x → down-proj to latent (l) → sigmoid router top-k → per-expert FFN
    (l→hidden→l) → weighted sum + shared expert → up-proj to d_model.

    Memory-optimized: dispatches tokens per-expert instead of gathering
    all expert weight matrices for all tokens simultaneously.
    """

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
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_hidden = expert_hidden

        # Down-projection: d_model → latent_dim
        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.norm_down = RMSNorm(latent_dim)

        # Router in latent space
        self.router = SigmoidRouter(latent_dim, n_experts, top_k, aux_free_bias_update)

        # Per-expert FFNs: (l → hidden → l)
        # expert_gate/expert_up are (E, l, hidden/2) for SiTU-GLU
        half_hidden = expert_hidden // 2
        self.expert_gate = nn.Parameter(torch.empty(n_experts, latent_dim, half_hidden))
        self.expert_up = nn.Parameter(torch.empty(n_experts, latent_dim, half_hidden))
        self.expert_down = nn.Parameter(torch.empty(n_experts, half_hidden, latent_dim))

        # Shared expert in latent space
        self.shared_up = nn.Linear(latent_dim, shared_expert_hidden, bias=False)
        self.shared_down = nn.Linear(shared_expert_hidden, latent_dim, bias=False)

        # Output
        self.norm_out = RMSNorm(latent_dim)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.expert_gate, std=0.02)
        nn.init.normal_(self.expert_up, std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        N = B * T

        # Down-project to latent space
        h = self.norm_down(self.down_proj(x))  # (B, T, l)
        h_flat = h.view(N, self.latent_dim)     # (N, l)

        # Route
        expert_idx, expert_w, router_logits = self.router(h_flat)
        # expert_idx: (N, k), expert_w: (N, k)

        # Update load-balancing bias
        self.router.update_bias(expert_idx)

        # Compute per-expert outputs efficiently:
        # Instead of gathering all expert matrices at once, dispatch tokens
        # to their selected experts and batch per unique expert.
        expert_out = torch.zeros(N, self.latent_dim, device=x.device, dtype=x.dtype)

        # Process each of the k slots
        for k_idx in range(self.top_k):
            idx_k = expert_idx[:, k_idx]           # (N,) — expert IDs for this slot
            w_k = expert_w[:, k_idx].unsqueeze(-1)  # (N, 1)

            # For each unique expert, process its assigned tokens at once
            for eid in range(self.n_experts):
                mask = (idx_k == eid)
                if not mask.any():
                    continue
                h_e = h_flat[mask]  # (n_e, l)

                # Expert FFN: gate projection + SiTUGLU activation
                gate = h_e @ self.expert_gate[eid]  # (n_e, h/2)
                up = h_e @ self.expert_up[eid]       # (n_e, h/2)
                act = softcap(gate, 4.0) * torch.sigmoid(gate) * softcap(up, 25.0)
                out_e = act @ self.expert_down[eid]  # (n_e, l)

                expert_out[mask] += w_k[mask] * out_e

        # Shared expert
        shared_out = self.shared_down(F.silu(self.shared_up(h_flat)))

        # Combine
        combined = expert_out + shared_out
        combined = self.norm_out(combined)
        combined = combined.view(B, T, self.latent_dim)
        out = self.up_proj(combined)

        return out


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (local copy for moe.py independence)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)
