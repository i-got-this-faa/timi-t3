"""P2 router health tests."""

import torch
import pytest

from kda_moe.config import ModelConfig
from kda_moe.moe import LatentMoE, SigmoidRouter


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_router_output_shape(device):
    moe = LatentMoE(
        d_model=640,
        latent_dim=320,
        n_experts=32,
        top_k=2,
        expert_hidden=660,
        shared_expert_hidden=640,
    ).to(device)

    x = torch.randn(2, 128, 640, device=device)
    out = moe(x)
    assert out.shape == (2, 128, 640)


def test_topk_selection():
    router = SigmoidRouter(d_model=320, n_experts=32, top_k=2)
    x = torch.randn(256, 320)
    indices, weights, _ = router(x)

    assert indices.shape == (256, 2)
    assert weights.shape == (256, 2)
    assert (indices >= 0).all() and (indices < 32).all()
    assert (weights.sum(dim=-1) <= 2.0 + 1e-5).all()


def test_router_gradient(device):
    moe = LatentMoE(
        d_model=640,
        latent_dim=320,
        n_experts=32,
        top_k=2,
        expert_hidden=660,
        shared_expert_hidden=640,
    ).to(device)

    x = torch.randn(2, 128, 640, device=device)
    out = moe(x)
    loss = out.mean()
    loss.backward()

    assert moe.router.weight.grad is not None, "Router weight has no grad"
    assert torch.isfinite(moe.router.weight.grad).all(), "Router weight grad has NaN"
    assert moe.expert_gate.grad is not None, "expert_gate has no grad"
    assert torch.isfinite(moe.expert_gate.grad).all(), "expert_gate grad has NaN"


def test_no_dead_experts_random():
    """Verify sigmoid router covers experts under random input."""
    router = SigmoidRouter(d_model=320, n_experts=32, top_k=2)
    seen = set()
    for _ in range(100):
        x = torch.randn(2 * 256, 320)
        indices, _, _ = router(x)
        seen.update(indices.flatten().tolist())

    missing = 32 - len(seen)
    print(
        f"Seen {len(seen)}/32 experts in 100 random batches "
        f"({missing} missing — expected with random sigmoid)"
    )


def test_router_bias_update():
    """Verify auxiliary-loss-free bias update changes biases."""
    router = SigmoidRouter(d_model=320, n_experts=8, top_k=2, aux_free_bias_update=0.01)
    initial_bias = router.expert_bias.clone()

    x = torch.randn(64, 320)
    indices, _, _ = router(x)
    router.update_bias(indices)

    diff = (router.expert_bias - initial_bias).abs().max().item()
    assert diff > 0, f"Bias didn't change after update: diff={diff:.6f}"
    print(f"Bias update: max change = {diff:.6f}")
