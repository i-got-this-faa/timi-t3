"""Optimizers (Lion, Muon) and EMA helper used by the training loop."""

from __future__ import annotations

import torch
from torch import nn


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration orthogonalizing the trailing two dims of g."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float()
    for _ in range(steps):
        xtx = x @ x.transpose(-2, -1)
        x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    return x.to(g.dtype)


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign(EMA) updates, no second moment."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-group["lr"])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan et al.): Newton-Schulz orthogonalized momentum for
    matrix params, NAdamW-style updates for vector/scalar params."""

    def __init__(self, params, lr=3e-4, betas=(0.95, 0.95, 0.95), eps=1e-8,
                 weight_decay=0.1, ns_steps=5):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                    "ns_steps": ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, _ = group["betas"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                momentum = state["momentum"]
                exp_avg_sq = state["exp_avg_sq"]
                momentum.lerp_(grad, 1 - beta1)
                if p.ndim >= 2:
                    update = _zeropower_via_newtonschulz5(momentum, group["ns_steps"])
                else:
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    update = momentum * torch.rsqrt(exp_avg_sq + group["eps"])
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


class EMA:
    """Exponential moving average of a model's float params. CPU-stored,
    copied into the model only when `swap_in()` is called (e.g. eval)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k] = v.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                shadow = self.shadow.get(k)
                if shadow is not None and v.is_floating_point():
                    shadow.mul_(self.decay).add_(v.detach().float().cpu(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in state.items()}
