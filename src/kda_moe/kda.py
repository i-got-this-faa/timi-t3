"""Kolmogorov-Dirac Attention: ShortConv module, recurrence, chunked-parallel KDA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ShortConv(nn.Module):
    """Depthwise causal 1D convolution — persistent trainable parameters.

    Used as a per-projection module in KDAAttention, not created per-call.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=False
        )
        nn.init.normal_(self.conv.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal depthwise conv along sequence dim.

        Args:
            x: (B, T, C)
        Returns:
            (B, T, C) — causal, same shape
        """
        B, T, C = x.shape
        # Conv1d expects (N, C, L)
        x_t = x.transpose(1, 2)  # (B, C, T)
        out = self.conv(x_t)  # (B, C, T + pad)
        out = out[:, :, :T]  # causal: strip future padding
        return out.transpose(1, 2)  # (B, T, C)


def kda_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    log_decay: Tensor,
    g: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """KDA recurrence — vectorized via Sherman-Morrison identity.

    Uses (I - βkk^T) @ diag(α) @ S = α⊙S - β·k⊗(k^T@(α⊙S))
    avoiding (D,D)@(D,D) matmuls and diag_embed.
    State in fp32 internally.
    """
    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    beta_f = beta.float()
    alpha = torch.exp(log_decay.float())  # (B, H, T, D)

    S = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.float()

    outputs = []
    for t in range(T):
        q_t = q[:, :, t, :]  # (B, H, D)
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]

        # X = diag(α_t) @ S = α_t ⊙ S  (channel-wise via broadcast)
        X = alpha[:, :, t, :].unsqueeze(-1) * S  # (B, H, D, D)

        # k^T @ X  (vector-matrix → vector)
        kX = torch.einsum("bhd,bhde->bhe", k_t, X)

        # corr = k ⊗ kX,   kv = k ⊗ v
        corr = torch.einsum("bhd,bhe->bhde", k_t, kX)
        kv = torch.einsum("bhd,bhe->bhde", k_t, v_t)

        b = beta_f[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S = X - b * corr + b * kv

        o_t = torch.einsum("bhd,bhde->bhe", q_t, S)  # S^T q: o[v] = Σ_k q[k] S[k,v]
        outputs.append(o_t)

    output = torch.stack(outputs, dim=2)
    return output.to(dtype), S.to(dtype)


def kda_chunked(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    log_decay: Tensor,
    g: Tensor,
    chunk_size: int = 64,
) -> Tensor:
    """Chunked-parallel KDA: split sequence, run recurrence per chunk, carry state.

    Processes the sequence in chunks of `chunk_size`. Within each chunk,
    runs the recurrence step-by-step and carries the state S across chunk
    boundaries.  This is equivalent to kda_recurrent but with bounded memory.

    Unlike standard attention, the recurrence is NOT parallelizable within
    a chunk — the state S depends on all previous tokens. The chunking
    exists purely for memory management of intermediate einsum products.

    Args:
        q,k,v: (B, H, T, D)
        beta: (B, H, T)
        log_decay: (B, H, T, D)
        g: (B, H, T) — unused in recurrence (applied externally)
        chunk_size: chunk boundary size (default 64)

    Returns:
        output: (B, H, T, D)
    """
    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    beta_f32 = beta.float()
    decay = torch.exp(log_decay.float())

    S = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    I = torch.eye(D, device=device, dtype=torch.float32).view(1, 1, D, D)

    outputs = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        for t in range(start, end):
            q_t = q_f32[:, :, t, :]
            k_t = k_f32[:, :, t, :]
            v_t = v_f32[:, :, t, :]
            b_t = beta_f32[:, :, t].unsqueeze(-1).unsqueeze(-1)
            a_t = decay[:, :, t, :]
            a_diag = torch.diag_embed(a_t)

            kk = torch.einsum("bhd,bhe->bhde", k_t, k_t)
            kv = torch.einsum("bhd,bhe->bhde", k_t, v_t)

            S = (I - b_t * kk) @ a_diag @ S + b_t * kv

            o_t = torch.einsum("bhd,bhde->bhe", q_t, S)  # S^T q
            outputs.append(o_t)

    output = torch.stack(outputs, dim=2)
    return output.to(dtype)


def compute_kda_params(
    x: Tensor,
    conv_q: ShortConv,
    conv_k: ShortConv,
    conv_v: ShortConv,
    W_q: nn.Linear,
    W_k: nn.Linear,
    W_v: nn.Linear,
    W_beta: nn.Linear,
    W_g: nn.Linear,
    W_ad: nn.Linear,
    W_au: nn.Linear,
    A_h: Tensor,
    n_heads: int,
    head_dim: int,
    g_min: float = -5.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project input x into q, k, v, beta, log_decay, g.

    Pipeline:
        x → ShortConv → Swish/L2-norm (q,k) or SiLU (v)
        beta = sigmoid(W_beta @ x)
        g = sigmoid(W_g @ x)
        log_decay = g_min · sigmoid(exp(A_h) · W_au @ silu(W_ad @ x))

    Args:
        x: (B, T, d_model)
        conv_q/k/v: ShortConv modules (one per projection)
        W_q/k/v: Linear(d_model → n_heads*head_dim)
        W_beta, W_g: Linear(d_model → n_heads)
        W_ad: Linear(d_model → d_model//4), W_au: Linear(d_model//4 → d_model)
        A_h: per-head decay scale (n_heads,)

    Returns:
        q,k,v: (B, H, T, D)
        beta: (B, H, T)
        log_decay: (B, H, T, D)
        g: (B, H, T)
    """
    B, T, d_model = x.shape
    inner_dim = n_heads * head_dim

    # ShortConv + Swish for q, k
    x_q = conv_q(x)
    x_k = conv_k(x)
    x_v = conv_v(x)

    q = x_q * torch.sigmoid(x_q)  # Swish
    q = W_q(q).view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, H, T, D)
    q = F.normalize(q, p=2, dim=-1)

    k = x_k * torch.sigmoid(x_k)
    k = W_k(k).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = F.normalize(k, p=2, dim=-1)

    v = F.silu(x_v)
    v = W_v(v).view(B, T, n_heads, head_dim).transpose(1, 2)

    # Gates
    beta = torch.sigmoid(W_beta(x)).transpose(1, 2)  # (B, H, T)
    g = torch.sigmoid(W_g(x)).transpose(1, 2)

    # Log decay
    z = F.silu(W_ad(x))
    z_full = W_au(z).view(B, T, n_heads, head_dim).transpose(1, 2)
    A_h_exp = torch.exp(A_h).view(1, n_heads, 1, 1)
    log_decay = g_min * torch.sigmoid(A_h_exp * z_full)
    log_decay = log_decay.clamp(min=g_min + 1e-6, max=0.0)

    return q, k, v, beta, log_decay, g


def kda_fla(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    log_decay: Tensor,
    g: Tensor,
) -> Tensor:
    """FLA-backed KDA. Import fla.ops.kda lazily.

    Our q/k are L2-normalized, ``log_decay`` is already in log space, and
    ``beta`` is already post-sigmoid — matching fla's KDA reference
    formulation (KDA paper). So we pass the precomputed decay directly
    (``use_gate_in_kernel=False``, ``use_qk_l2norm_in_kernel=False``,
    ``use_beta_sigmoid_in_kernel=False``, ``scale=1.0``) and apply the
    output gate ``g`` at the call site as usual.

    fla wants ``(B, T, H, D)``; we convert from our ``(B, H, T, D)``.

    Raises ImportError with message 'FLA not available — use chunked PyTorch path'
    if the package is not installed.
    """
    try:
        from fla.ops.kda import fused_recurrent_kda  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("FLA not available — use chunked PyTorch path")

    out = fused_recurrent_kda(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        log_decay.transpose(1, 2).contiguous(),
        beta.transpose(1, 2).contiguous(),
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        output_final_state=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.transpose(1, 2)  # back to (B, H, T, D)
