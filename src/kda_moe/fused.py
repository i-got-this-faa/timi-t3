"""Optional fused kernels.

Each function attempts a Triton/flash kernel when the runtime libs are
available and otherwise falls back to the equivalent torch op, so training
works identically on CPU / no-GPU dev machines. Kernels are only used when
the caller opts in via config (fused_rmsnorm / fused_cross_entropy).
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

_trt = None
_trt_checked = False


def _triton() -> object | None:
    global _trt, _trt_checked
    if not _trt_checked:
        _trt_checked = True
        try:
            import triton  # type: ignore[import-not-found]

            _trt = triton
        except ImportError:
            _trt = None
    return _trt


@functools.cache
def _has_triton() -> bool:
    return _triton() is not None and torch.cuda.is_available()


@functools.cache
def _flash_attn() -> object | None:
    try:
        import flash_attn  # type: ignore[import-not-found]
    except ImportError:
        return None
    return flash_attn


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm; Triton kernel when available, else torch-native."""
    if _has_triton() and x.is_cuda:
        kernel = _build_rmsnorm_kernel()
        return kernel(x, weight, eps)
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with optional online softmax reduction (matches F.cross_entropy)."""
    if _has_triton() and logits.is_cuda and logits.size(-1) % 16 == 0:
        kernel = _build_ce_kernel()
        return kernel(logits, targets, ignore_index)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def _build_rmsnorm_kernel():
    triton = _triton()
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        offs = row * n_cols + cols
        x = tl.load(x_ptr + offs, mask=cols < n_cols, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        x = x * tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0)
        tl.store(y_ptr + offs, x * w, mask=cols < n_cols)

    def kernel(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        rows, n_cols = x.shape
        y = torch.empty_like(x)
        _rmsnorm_kernel[(rows,)](
            x, weight, y, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols)
        )
        return y

    return kernel


def _build_ce_kernel():
    triton = _triton()
    import triton.language as tl

    @triton.jit
    def _ce_kernel(
        logits_ptr, tgt_ptr, out_ptr, n_rows, n_cols, ignore, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        offs = row * n_cols + cols
        logits = tl.load(logits_ptr + offs, mask=cols < n_cols, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(logits, axis=0)
        logits = logits - m
        lse = tl.log(tl.sum(tl.exp(logits), axis=0)) + m
        tgt = tl.load(tgt_ptr + row)
        loss = lse - tl.sum(tl.where(cols == tgt, logits, 0.0), axis=0)
        tl.store(out_ptr + row, loss)

    def kernel(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        rows, n_cols = logits.shape
        out = torch.empty(rows, device=logits.device, dtype=torch.float32)
        _ce_kernel[(rows,)](
            logits, targets, out, rows, n_cols, ignore_index,
            BLOCK=triton.next_power_of_2(n_cols),
        )
        valid = targets != ignore_index
        return out[valid].mean() if valid.any() else out.mean()

    return kernel
