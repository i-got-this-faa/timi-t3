"""Model configuration dataclass with TOML I/O and presets.

Configs live in ``configs/*.toml``; the preset classmethods below are thin
wrappers that load from those files so there is a single source of truth.
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

# Shared pretraining corpus mix (tokens) and disk caps (GB). Kept in one place
# so rebalancing the dataset is a one-line edit instead of a scavenger hunt
# through every config file. Per-run configs may override via [data].
DEFAULT_DATA_MIX: dict[str, float] = {
    "dclm": 0.45,
    "fineweb": 0.20,
    "code": 0.20,
    "math": 0.10,
    "tinystories": 0.03,
    "markdown": 0.02,
}

DEFAULT_DATA_CAPS_GB: dict[str, float] = {
    "dclm": 2.5,
    "fineweb": 2.0,
    "code": 2.5,
    "math": 1.0,
    "tinystories": 0.5,
    "markdown": 0.5,
}


def _to_toml(value: Any) -> Any:
    """Recursively convert Python types to TOML-compatible types."""
    if isinstance(value, tuple):
        return [_to_toml(v) for v in value]
    if isinstance(value, list):
        return [_to_toml(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_toml(v) for k, v in value.items()}
    return value


@dataclass
class ModelConfig:
    """Every hyperparameter for the KDA-MoE model and training loop.

    Fields match plan.md §2.1, §2.2, §6 (snake_case).

    Usage:
        cfg = ModelConfig.preset_1b()
        cfg = ModelConfig.from_toml("configs/kda_moe_1b.toml")
        cfg.to_toml("configs/my_run.toml")
    """

    # ── vocabulary ──────────────────────────────────────────
    vocab_size: int = 32000
    pad_token_id: int = 0  # set after tokenizer training

    # ── architecture ────────────────────────────────────────
    n_layers: int = 16
    d_model: int = 640
    n_heads: int = 10
    n_kv_heads: int = 5  # GQA kv heads (2:1 for 10 heads)
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    activation: str = "silu"  # dense FFN activation: silu | gelu
    ffn_multiplier: float = 4.0  # dense-FFN hidden = d_model * this
    use_sdpa: bool = True  # torch SDPA for the global GQA layers

    # ── KDA ─────────────────────────────────────────────────
    use_kda: bool = True
    kda_kernel_size: int = 4  # ShortConv1D kernel
    kda_chunk_size: int = 64
    kda_g_min: float = -5.0
    kda_use_fla: bool = False  # FLA backend flag

    # ── MoE ─────────────────────────────────────────────────
    use_moe: bool = True
    n_experts: int = 96
    top_k: int = 4
    latent_dim: int = 320
    expert_hidden: int = 1320
    shared_expert_hidden: int = 640
    aux_free_bias_update: float = 0.001  # per optimizer-step bias adjustment
    z_loss_coeff: float = 1e-4  # router-collapse insurance; 0 = off

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
    warmup_steps: int = 2000
    total_steps: int = 10000
    clip_grad_norm: float = 1.0
    optimizer: str = "adamw8bit"  # adamw | adamw8bit | lion | muon

    # ── precision / perf ────────────────────────────────────
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False

    # ── EMA ─────────────────────────────────────────────────
    ema: bool = False
    ema_decay: float = 0.999

    # ── fused kernels (GPU-only; graceful fallback) ─────────
    fused_rmsnorm: bool = False
    fused_cross_entropy: bool = False

    # ── batch ───────────────────────────────────────────────
    micro_batch_size: int = 1
    grad_accum_steps: int = 16

    # ── checkpointing / logging ─────────────────────────────
    checkpoint_interval: int = 500
    full_checkpoint_interval: int = 2000
    log_interval: int = 10

    # ── data (defaults shared across configs; override per run) ──
    data_mix: dict[str, float] = field(default_factory=lambda: DEFAULT_DATA_MIX.copy())
    data_caps_gb: dict[str, float] = field(default_factory=lambda: DEFAULT_DATA_CAPS_GB.copy())

    # ── dropout (0 by default; finetuning presets set these) ──
    dropout: float = 0.0  # legacy global dropout (dense baseline)
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    residual_dropout: float = 0.0

    # TOML section layout used by to_toml.
    _SECTIONS: ClassVar[dict[str, list[str]]] = {
        "architecture": [
            "vocab_size",
            "pad_token_id",
            "n_layers",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "rms_norm_eps",
            "initializer_std",
            "activation",
            "ffn_multiplier",
            "use_sdpa",
            "use_attn_res",
            "dropout",
            "attention_dropout",
            "ffn_dropout",
            "residual_dropout",
        ],
        "kda": [
            "use_kda",
            "kda_kernel_size",
            "kda_chunk_size",
            "kda_g_min",
            "kda_use_fla",
        ],
        "moe": [
            "use_moe",
            "n_experts",
            "top_k",
            "latent_dim",
            "expert_hidden",
            "shared_expert_hidden",
            "aux_free_bias_update",
            "z_loss_coeff",
            "global_attn_layers",
        ],
        "training": [
            "seq_len",
            "curriculum_start_seq",
            "curriculum_milestones",
            "lr",
            "betas",
            "weight_decay",
            "warmup_steps",
            "total_steps",
            "clip_grad_norm",
            "optimizer",
            "bf16",
            "tf32",
            "gradient_checkpointing",
            "compile_model",
            "ema",
            "ema_decay",
            "fused_rmsnorm",
            "fused_cross_entropy",
            "micro_batch_size",
            "grad_accum_steps",
            "checkpoint_interval",
            "full_checkpoint_interval",
            "log_interval",
        ],
        "data": ["data_mix", "data_caps_gb"],
    }

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelConfig:
        """Load config from a TOML file. Unknown keys warn but don't error."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        def _pop_section(section: str) -> dict[str, Any]:
            return {k: v for k, v in raw.pop(section, {}).items()}

        # flatten sections in order; leftover top-level keys act as overrides
        merged: dict[str, Any] = {}
        for section in ("architecture", "kda", "moe", "training", "data"):
            merged.update(_pop_section(section))
        merged.update(raw)

        # handle tuple fields that come from TOML as lists
        for key in ("global_attn_layers", "betas"):
            if key in merged and isinstance(merged[key], list):
                merged[key] = tuple(merged[key])

        # handle curriculum_milestones: list of [step, seq_len] → list of tuples
        if "curriculum_milestones" in merged and isinstance(merged["curriculum_milestones"], list):
            merged["curriculum_milestones"] = [
                tuple(pair) for pair in merged["curriculum_milestones"]
            ]

        # migrate legacy use_8bit_adam → optimizer
        if "use_8bit_adam" in merged and "optimizer" not in merged:
            merged["optimizer"] = "adamw8bit" if merged.pop("use_8bit_adam") else "adamw"
        else:
            merged.pop("use_8bit_adam", None)

        unknown = [k for k in merged if k not in cls.__dataclass_fields__]
        for k in unknown:
            warnings.warn(f"Unknown config key '{k}' in {path}; ignoring", stacklevel=2)

        return cls(**{k: v for k, v in merged.items() if k in cls.__dataclass_fields__})

    def to_toml(self, path: str | Path) -> None:
        """Serialize this config to a TOML file in the standard section layout."""
        import tomli_w

        out = {
            section: {k: _to_toml(getattr(self, k)) for k in keys}
            for section, keys in self._SECTIONS.items()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(out, f)

    # ── presets ─────────────────────────────────────────────
    # These load from configs/*.toml; the TOML files are the source of truth.

    @classmethod
    def preset_1b(cls) -> ModelConfig:
        """Primary 1B config (Colab T4).  plan.md §2.1."""
        return cls.from_toml(CONFIG_DIR / "kda_moe_1b.toml")

    @classmethod
    def preset_450m(cls) -> ModelConfig:
        """Smoke config for local RTX 4050.  plan.md §2.2."""
        return cls.from_toml(CONFIG_DIR / "kda_moe_450m.toml")

    @classmethod
    def preset_dense_100m(cls) -> ModelConfig:
        """Dense 100M control: 16-layer GQA+NoPE, no KDA, no MoE.  plan.md §2.3."""
        return cls.from_toml(CONFIG_DIR / "dense_100m.toml")

    @classmethod
    def preset_24m(cls) -> ModelConfig:
        """24M scaling-law variant: 4 layers, tiny dims, ultra-fast smoke."""
        return cls.from_toml(CONFIG_DIR / "kda_moe_24m.toml")

    @classmethod
    def preset_80m(cls) -> ModelConfig:
        """80M scaling-law variant: 8 layers, moderate dims."""
        return cls.from_toml(CONFIG_DIR / "kda_moe_80m.toml")

    @classmethod
    def preset_180m(cls) -> ModelConfig:
        """180M scaling-law variant: 12 layers, mid-scale dims."""
        return cls.from_toml(CONFIG_DIR / "kda_moe_180m.toml")

    @classmethod
    def preset_smoke(cls) -> ModelConfig:
        """Ultra-small smoke for fast local iteration.  450M with seq=128."""
        return cls.from_toml(CONFIG_DIR / "smoke.toml")
