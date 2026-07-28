"""Model configuration dataclass with TOML I/O and presets."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Every hyperparameter for the KDA-MoE model and training loop.

    Fields match plan.md §2.1, §2.2, §6 (snake_case).

    Usage:
        cfg = ModelConfig.preset_1b()
        cfg = ModelConfig.from_toml("configs/kda_moe_1b.toml")
    """

    # ── vocabulary ──────────────────────────────────────────
    vocab_size: int = 32000
    pad_token_id: int = 0  # set after tokenizer training

    # ── architecture ────────────────────────────────────────
    n_layers: int = 16
    d_model: int = 640
    n_heads: int = 10
    n_kv_heads: int = 2  # GQA kv heads
    head_dim: int = 64

    # ── KDA ─────────────────────────────────────────────────
    use_kda: bool = True
    kda_kernel_size: int = 4  # ShortConv1D kernel
    kda_chunk_size: int = 64
    kda_g_min: float = -5.0
    kda_use_fla: bool = False  # FLA backend flag

    # ── MoE ─────────────────────────────────────────────────
    use_moe: bool = True
    n_experts: int = 32
    top_k: int = 2
    latent_dim: int = 320
    expert_hidden: int = 660
    shared_expert_hidden: int = 640
    aux_free_bias_update: float = 0.001  # per-step bias adjustment
    z_loss_coeff: float = 0.0  # 0 = off; 1e-4 if router saturating

    # ── global attention schedule ───────────────────────────
    global_attn_layers: tuple[int, ...] = (3, 7, 11, 15)

    # ── Block AttnRes (phase-2 flag) ────────────────────────
    use_attn_res: bool = False

    # ── sequence / curriculum ───────────────────────────────
    seq_len: int = 2048
    curriculum_start_seq: int = 512
    curriculum_milestones: list[tuple[int, int]] = field(default_factory=list)

    # ── optimizer ───────────────────────────────────────────
    lr: float = 1.5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 10000
    clip_grad_norm: float = 1.0
    use_8bit_adam: bool = True

    # ── batch ───────────────────────────────────────────────
    micro_batch_size: int = 1
    grad_accum_steps: int = 16

    # ── checkpointing / logging ─────────────────────────────
    checkpoint_interval: int = 500
    full_checkpoint_interval: int = 2000
    log_interval: int = 10

    # ── data ────────────────────────────────────────────────
    data_mix: dict[str, float] = field(default_factory=dict)
    data_caps_gb: dict[str, float] = field(default_factory=dict)

    # ── dropout (not used in v1; kept for dense baseline) ──
    dropout: float = 0.0

    @classmethod
    def from_toml(cls, path: str) -> ModelConfig:
        """Load config from a TOML file. Unknown keys warn but don't error."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        def _pop_section(section: str) -> dict[str, Any]:
            return {k: v for k, v in raw.pop(section, {}).items()}

        arch = _pop_section("architecture")
        kda = _pop_section("kda")
        moe = _pop_section("moe")
        training = _pop_section("training")
        data_section = _pop_section("data")

        # flatten
        merged: dict[str, Any] = {}
        merged.update(arch)
        merged.update(kda)
        merged.update(moe)
        merged.update(training)
        merged.update(data_section)
        merged.update(raw)  # top-level overrides

        # handle tuple fields that come from TOML as lists
        for key in ("global_attn_layers", "betas"):
            if key in merged and isinstance(merged[key], list):
                merged[key] = tuple(merged[key])

        # handle curriculum_milestones: list of [step, seq_len] → list of tuples
        if "curriculum_milestones" in merged and isinstance(merged["curriculum_milestones"], list):
            merged["curriculum_milestones"] = [
                tuple(pair) for pair in merged["curriculum_milestones"]
            ]

        return cls(**{k: v for k, v in merged.items() if k in cls.__dataclass_fields__})

    # ── presets ─────────────────────────────────────────────

    @classmethod
    def preset_1b(cls) -> ModelConfig:
        """Primary 1B config (Colab T4).  plan.md §2.1."""
        return cls(
            n_layers=16,
            d_model=640,
            n_heads=10,
            n_kv_heads=2,
            head_dim=64,
            use_kda=True,
            use_moe=True,
            n_experts=32,
            top_k=2,
            latent_dim=320,
            expert_hidden=660,
            shared_expert_hidden=640,
            global_attn_layers=(3, 7, 11, 15),
            seq_len=2048,
            curriculum_start_seq=512,
            curriculum_milestones=[(2000, 1024), (5000, 2048)],
            lr=1.5e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=10,
            total_steps=200,
            micro_batch_size=1,
            grad_accum_steps=4,
            data_mix={
                "dclm": 0.45,
                "fineweb": 0.20,
                "code": 0.20,
                "math": 0.10,
                "tinystories": 0.03,
                "markdown": 0.02,
            },
            data_caps_gb={
                "dclm": 2.5,
                "fineweb": 2.0,
                "code": 2.5,
                "math": 1.0,
                "tinystories": 0.5,
                "markdown": 0.5,
            },
        )

    @classmethod
    def preset_450m(cls) -> ModelConfig:
        """Smoke config for local RTX 4050.  plan.md §2.2."""
        return cls(
            n_layers=16,
            d_model=640,
            n_heads=10,
            n_kv_heads=2,
            head_dim=64,
            use_kda=True,
            use_moe=True,
            n_experts=32,
            top_k=2,
            latent_dim=320,
            expert_hidden=768,
            shared_expert_hidden=1280,
            global_attn_layers=(3, 7, 11, 15),
            seq_len=2048,
            curriculum_start_seq=512,
            curriculum_milestones=[(2000, 1024), (5000, 2048)],
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=100,
            total_steps=10000,
            micro_batch_size=1,
            grad_accum_steps=4,
            data_mix={
                "dclm": 0.45,
                "fineweb": 0.20,
                "code": 0.20,
                "math": 0.10,
                "tinystories": 0.03,
                "markdown": 0.02,
            },
            data_caps_gb={
                "dclm": 2.5,
                "fineweb": 2.0,
                "code": 2.5,
                "math": 1.0,
                "tinystories": 0.5,
                "markdown": 0.5,
            },
        )

    @classmethod
    def preset_dense_100m(cls) -> ModelConfig:
        """Dense 100M control: 16-layer GQA+NoPE, no KDA, no MoE.  plan.md §2.3."""
        return cls(
            n_layers=16,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            head_dim=64,
            use_kda=False,
            use_moe=False,
            n_experts=0,
            top_k=0,
            latent_dim=512,
            expert_hidden=2048,  # unused when use_moe=False, fallback FFN size
            shared_expert_hidden=2048,
            global_attn_layers=tuple(range(16)),  # all layers use GlobalGQA
            seq_len=2048,
            curriculum_start_seq=512,
            curriculum_milestones=[(2000, 1024), (5000, 2048)],
            lr=1.5e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=200,
            total_steps=10000,
            micro_batch_size=1,
            grad_accum_steps=16,
            dropout=0.0,
        )

    @classmethod
    def preset_24m(cls) -> ModelConfig:
        """24M scaling-law variant: 4 layers, tiny dims, ultra-fast smoke."""
        return cls(
            n_layers=4,
            d_model=256,
            n_heads=4,
            n_kv_heads=1,
            head_dim=64,
            use_kda=True,
            use_moe=True,
            n_experts=8,
            top_k=2,
            latent_dim=128,
            expert_hidden=256,
            shared_expert_hidden=256,
            global_attn_layers=(3,),
            seq_len=512,
            curriculum_start_seq=512,
            curriculum_milestones=[],
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=20,
            total_steps=500,
            micro_batch_size=1,
            grad_accum_steps=4,
            checkpoint_interval=100,
            log_interval=10,
            data_mix={
                "dclm": 0.45,
                "fineweb": 0.20,
                "code": 0.20,
                "math": 0.10,
                "tinystories": 0.03,
                "markdown": 0.02,
            },
            data_caps_gb={
                "dclm": 2.5,
                "fineweb": 2.0,
                "code": 2.5,
                "math": 1.0,
                "tinystories": 0.5,
                "markdown": 0.5,
            },
        )

    @classmethod
    def preset_80m(cls) -> ModelConfig:
        """80M scaling-law variant: 8 layers, moderate dims."""
        return cls(
            n_layers=8,
            d_model=384,
            n_heads=6,
            n_kv_heads=1,
            head_dim=64,
            use_kda=True,
            use_moe=True,
            n_experts=16,
            top_k=2,
            latent_dim=192,
            expert_hidden=384,
            shared_expert_hidden=384,
            global_attn_layers=(3, 7),
            seq_len=512,
            curriculum_start_seq=512,
            curriculum_milestones=[],
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=20,
            total_steps=500,
            micro_batch_size=1,
            grad_accum_steps=4,
            checkpoint_interval=100,
            log_interval=10,
            data_mix={
                "dclm": 0.45,
                "fineweb": 0.20,
                "code": 0.20,
                "math": 0.10,
                "tinystories": 0.03,
                "markdown": 0.02,
            },
            data_caps_gb={
                "dclm": 2.5,
                "fineweb": 2.0,
                "code": 2.5,
                "math": 1.0,
                "tinystories": 0.5,
                "markdown": 0.5,
            },
        )

    @classmethod
    def preset_180m(cls) -> ModelConfig:
        """180M scaling-law variant: 12 layers, mid-scale dims."""
        return cls(
            n_layers=12,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            head_dim=64,
            use_kda=True,
            use_moe=True,
            n_experts=24,
            top_k=2,
            latent_dim=256,
            expert_hidden=512,
            shared_expert_hidden=512,
            global_attn_layers=(3, 7, 11),
            seq_len=512,
            curriculum_start_seq=512,
            curriculum_milestones=[],
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup_steps=20,
            total_steps=500,
            micro_batch_size=1,
            grad_accum_steps=4,
            checkpoint_interval=100,
            log_interval=10,
            data_mix={
                "dclm": 0.45,
                "fineweb": 0.20,
                "code": 0.20,
                "math": 0.10,
                "tinystories": 0.03,
                "markdown": 0.02,
            },
            data_caps_gb={
                "dclm": 2.5,
                "fineweb": 2.0,
                "code": 2.5,
                "math": 1.0,
                "tinystories": 0.5,
                "markdown": 0.5,
            },
        )

    @classmethod
    def preset_smoke(cls) -> ModelConfig:
        """Ultra-small smoke for fast local iteration.  450M with seq=128."""
        cfg = cls.preset_450m()
        cfg.seq_len = 128
        cfg.curriculum_start_seq = 128
        cfg.curriculum_milestones = []
        cfg.micro_batch_size = 1
        cfg.grad_accum_steps = 4
        cfg.total_steps = 500
        cfg.warmup_steps = 10
        cfg.checkpoint_interval = 100
        cfg.log_interval = 5
        return cfg
