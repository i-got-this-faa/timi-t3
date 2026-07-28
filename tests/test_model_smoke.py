"""P2 gate: model forward/backward smoke tests."""
import torch
import pytest

from kda_moe.config import ModelConfig
from kda_moe.model import KDAMoEModel


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_forward_450m(device):
    config = ModelConfig.preset_450m()
    config.seq_len = 128
    model = KDAMoEModel(config).to(device)

    input_ids = torch.randint(0, config.vocab_size, (1, 128), device=device)
    targets = torch.randint(0, config.vocab_size, (1, 128), device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = model.loss_fn(logits, targets, ignore_index=-100)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    total, _ = model.get_num_params()
    print(f"450M model: {total/1e6:.1f}M params, loss={loss.item():.4f}")


def test_backward_small(device):
    """Forward + backward + gradient check on a tiny config."""
    config = ModelConfig(
        vocab_size=1000, n_layers=2, d_model=128, n_heads=2,
        n_kv_heads=1, head_dim=32, use_kda=True, use_moe=True,
        n_experts=8, top_k=2, latent_dim=64, expert_hidden=128,
        shared_expert_hidden=128, global_attn_layers=(1,),
        seq_len=32,
    )
    model = KDAMoEModel(config).to(device)

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    targets = torch.randint(0, config.vocab_size, (1, 32), device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = model.loss_fn(logits, targets, ignore_index=-100)

    loss.backward()

    grads = sum(1 for _, p in model.named_parameters() if p.grad is not None)
    nans = sum(1 for _, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all())
    assert grads > 0, "No parameters received gradients"
    assert nans == 0, f"{nans} parameters have NaN gradients"
    total, _ = model.get_num_params()
    print(f"Backward test: {total/1e3:.1f}K params, {grads} grads OK")


def test_forward_1b_cpu():
    config = ModelConfig.preset_1b()
    config.seq_len = 64
    model = KDAMoEModel(config).to("cpu")
    model.eval()  # disable gradient checkpointing for CPU test

    input_ids = torch.randint(0, config.vocab_size, (1, 64))
    targets = torch.randint(0, config.vocab_size, (1, 64))

    with torch.no_grad():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = model.loss_fn(logits, targets, ignore_index=-100)

    assert torch.isfinite(loss)
    total, _ = model.get_num_params()
    assert total > 350_000_000, f"Expected >350M params, got {total/1e6:.1f}M"
    print(f"1B model (CPU): {total/1e6:.1f}M params, loss={loss.item():.4f}")


def test_gradient_checkpointing(device):
    """Verify that model trains with checkpointing."""
    config = ModelConfig(
        vocab_size=1000, n_layers=4, d_model=128, n_heads=2,
        n_kv_heads=1, head_dim=32, use_kda=True, use_moe=False,
        n_experts=0, top_k=0, latent_dim=64, expert_hidden=128,
        shared_expert_hidden=128, global_attn_layers=(),
        seq_len=32,
    )
    model = KDAMoEModel(config).to(device)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    targets = torch.randint(0, config.vocab_size, (1, 32), device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = model.loss_fn(logits, targets, ignore_index=-100)

    loss.backward()

    for i, layer in enumerate(model.layers):
        has_grad = False
        for p in layer.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"Layer {i} has no gradients"
    print("All layers received gradients with checkpointing")
