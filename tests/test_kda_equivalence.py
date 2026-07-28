"""P3 gate: KDA equivalence and correctness tests."""
import torch
import pytest

from kda_moe.kda import (
    ShortConv,
    kda_recurrent,
    kda_chunked,
    compute_kda_params,
    kda_fla,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_kda_inputs(B, H, T, D, device):
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.float32))
    log_decay = -5.0 * torch.sigmoid(torch.randn(B, H, T, D, device=device, dtype=torch.float32))
    g = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.float32))
    return q, k, v, beta, log_decay, g


def test_recurrent_equals_chunked(device):
    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 256, 64, device)

    out_rec, _ = kda_recurrent(q, k, v, beta, log_decay, g)
    out_chunked = kda_chunked(q, k, v, beta, log_decay, g, chunk_size=64)

    diff = (out_rec - out_chunked).abs().max().item()
    print(f"Max abs diff recurrent vs chunked: {diff:.6f}")
    assert diff < 1e-6, f"Difference too large: {diff:.6f}"


def test_causality(device):
    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 256, 64, device)

    out_orig, _ = kda_recurrent(q, k, v, beta, log_decay, g)

    q_mod = q.clone()
    q_mod[:, :, 100:, :] = torch.randn_like(q[:, :, 100:, :])
    k_mod = k.clone()
    k_mod[:, :, 100:, :] = torch.randn_like(k[:, :, 100:, :])
    v_mod = v.clone()
    v_mod[:, :, 100:, :] = torch.randn_like(v[:, :, 100:, :])

    out_mod, _ = kda_recurrent(q_mod, k_mod, v_mod, beta, log_decay, g)

    diff_before = (out_orig[:, :, :100, :] - out_mod[:, :, :100, :]).abs().max().item()
    print(f"Max abs diff before modification point: {diff_before:.10f}")
    assert diff_before < 1e-6, f"Causality violated: diff {diff_before:.10f}"

    diff_after = (out_orig[:, :, 100:, :] - out_mod[:, :, 100:, :]).abs().max().item()
    assert diff_after > 1e-6, "Expected difference after modification point"


def test_no_nan_at_2048(device):
    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 2048, 64, device)

    out, final_state = kda_recurrent(q, k, v, beta, log_decay, g)

    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isnan(final_state).any(), "NaN in final state"
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()
    print(f"2048-token KDA: output range [{out.min().item():.4f}, {out.max().item():.4f}]")


def test_log_decay_bounds(device):
    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 512, 64, device)

    min_val = log_decay.min().item()
    max_val = log_decay.max().item()
    print(f"log_decay range: [{min_val:.6f}, {max_val:.6f}]")
    assert min_val >= -5.0 + 1e-6, f"log_decay below bound: {min_val}"
    assert max_val <= 0.0 + 1e-6, f"log_decay above bound: {max_val}"


def test_chunked_gradient(device):
    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 128, 64, device)

    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    beta = beta.clone().requires_grad_(True)
    log_decay = log_decay.clone().requires_grad_(True)

    out = kda_chunked(q, k, v, beta, log_decay, g, chunk_size=64)
    loss = out.mean()
    loss.backward()

    for name, tensor in [("q", q), ("k", k), ("v", v), ("beta", beta),
                          ("log_decay", log_decay)]:
        assert tensor.grad is not None, f"{name} has no grad"
        assert torch.isfinite(tensor.grad).all(), f"{name} grad has NaN/Inf"

    print("All chunked KDA grads are finite")


def test_fla_matches_recurrent(device):
    try:
        import fla  # noqa: F401
    except ImportError:
        pytest.skip("FLA not installed")

    torch.manual_seed(42)
    q, k, v, beta, log_decay, g = _make_kda_inputs(2, 4, 128, 64, device)

    out_rec, _ = kda_recurrent(q, k, v, beta, log_decay, g)

    try:
        out_fla = kda_fla(q, k, v, beta, log_decay, g)
    except (ImportError, AttributeError):
        pytest.skip("FLA kda_fla not functional")

    diff = (out_rec - out_fla).abs().max().item()
    print(f"Max abs diff recurrent vs FLA: {diff:.6f}")
    assert diff < 1e-3, f"FLA difference too large: {diff:.6f}"


def test_shortconv_persistent(device):
    """Verify ShortConv is a persistent nn.Module with trainable parameters."""
    conv = ShortConv(64, kernel_size=4).to(device)
    x = torch.randn(2, 128, 64, device=device)
    out = conv(x)
    assert out.shape == x.shape
    # Verify parameters exist and receive gradients
    loss = out.mean()
    loss.backward()
    assert conv.conv.weight.grad is not None, "ShortConv weight has no grad"
    print(f"ShortConv: in={x.shape} out={out.shape}")


def test_compute_kda_params_shape(device):
    """Verify compute_kda_params with persistent ShortConv modules."""
    B, T, d_model = 2, 64, 640
    n_heads, head_dim = 10, 64
    inner_dim = n_heads * head_dim
    decay_inner = d_model // 4  # 160

    x = torch.randn(B, T, d_model, device=device)

    conv_q = ShortConv(d_model).to(device)
    conv_k = ShortConv(d_model).to(device)
    conv_v = ShortConv(d_model).to(device)

    W_q = torch.nn.Linear(d_model, inner_dim, bias=False).to(device)
    W_k = torch.nn.Linear(d_model, inner_dim, bias=False).to(device)
    W_v = torch.nn.Linear(d_model, inner_dim, bias=False).to(device)
    W_beta = torch.nn.Linear(d_model, n_heads, bias=False).to(device)
    W_g = torch.nn.Linear(d_model, n_heads, bias=False).to(device)
    W_ad = torch.nn.Linear(d_model, decay_inner, bias=False).to(device)
    W_au = torch.nn.Linear(decay_inner, inner_dim, bias=False).to(device)
    A_h = torch.randn(n_heads, device=device)

    q, k, v, beta, log_decay, g = compute_kda_params(
        x, conv_q, conv_k, conv_v,
        W_q, W_k, W_v, W_beta, W_g, W_ad, W_au, A_h,
        n_heads, head_dim,
    )

    assert q.shape == (B, n_heads, T, head_dim)
    assert k.shape == (B, n_heads, T, head_dim)
    assert v.shape == (B, n_heads, T, head_dim)
    assert beta.shape == (B, n_heads, T)
    assert log_decay.shape == (B, n_heads, T, head_dim)
    assert g.shape == (B, n_heads, T)

    assert log_decay.min() >= -5.0 - 1e-5
    assert log_decay.max() <= 0.0 + 1e-5
