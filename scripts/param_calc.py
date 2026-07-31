"""Analytic parameter calculator + ground-truth count for a config TOML.

Usage:
    python scripts/param_calc.py configs/kda_moe_1b.toml
    python scripts/param_calc.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.model import KDAMoEModel


def analytic_params(cfg: ModelConfig) -> int:
    """Hand-counted parameter estimate, matching KDAMoEModel exactly."""
    h = cfg.d_model
    inner = cfg.n_heads * cfg.head_dim
    kv = cfg.n_kv_heads * cfg.head_dim
    di = h // 4
    ffn_hidden = int(h * cfg.ffn_multiplier)

    n_global = len(cfg.global_attn_layers)
    n_kda = cfg.n_layers - n_global
    n_dense = 1 if cfg.use_moe else cfg.n_layers
    n_moe = cfg.n_layers - n_dense

    # KDA attention layer
    kda = (
        3 * 4 * h  # ShortConv (kernel_size=4)
        + 3 * h * inner  # W_q/W_k/W_v
        + 2 * h * cfg.n_heads  # W_beta/W_g
        + h * di + di * inner  # W_ad/W_au
        + cfg.n_heads  # A_h
        + inner * h  # W_o
        + h  # norm
    )

    # Global GQA layer
    gqa = h * inner + 2 * h * kv + inner * h + h

    # Dense FFN (3-proj SwiGLU-ish)
    dense = 3 * h * ffn_hidden

    # LatentMoE
    half_h = cfg.expert_hidden // 2
    moe = (
        h * cfg.latent_dim  # down_proj
        + cfg.latent_dim  # norm_down
        + cfg.latent_dim * cfg.n_experts  # router
        + 3 * cfg.n_experts * cfg.latent_dim * half_h  # expert gate/up/down
        + 2 * cfg.latent_dim * cfg.shared_expert_hidden  # shared expert
        + cfg.latent_dim  # norm_out
        + cfg.latent_dim * h  # up_proj
    )

    emb = cfg.vocab_size * h  # tied embedding + lm_head
    final_norm = h

    return (
        n_kda * kda + n_global * gqa + n_dense * dense + n_moe * moe + emb + final_norm
    )


def count_config(path: Path) -> None:
    cfg = ModelConfig.from_toml(str(path))
    analytic = analytic_params(cfg)
    try:
        model = KDAMoEModel(cfg)
        actual, trainable = model.get_num_params()
        pct = 100 * actual / analytic if analytic else 0.0
        print(
            f"{path.name:<24} analytic={analytic/1e6:8.2f}M  actual={actual/1e6:8.2f}M  "
            f"trainable={trainable/1e6:8.2f}M  (match={pct:.1f}%)"
        )
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"{path.name:<24} analytic={analytic/1e6:8.2f}M  BUILD FAILED: {exc}")


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        targets = sorted(Path(CONFIG_DIR).glob("*.toml"))
    else:
        targets = [Path(a) for a in args if a != "--all"]
    for p in targets:
        if not p.exists():
            print(f"missing: {p}")
            continue
        count_config(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
