"""Unified training entry point: pretrain / smoke / dense control / SFT.

Usage:
    python scripts/train.py --config configs/kda_moe_24m.toml          # local smoke (real data if present)
    python scripts/train.py --config configs/kda_moe_80m.toml          # small run on the 1GB language mix
    python scripts/train.py --config configs/dense_100m.toml           # dense control
    python scripts/train.py --config configs/kda_moe_1b.toml --colab   # 1B pilot (Colab: drive + OOM retry)
    python scripts/train.py --config configs/kda_moe_1b.toml --sft     # FABLE.5 SFT

Data:
    Reads shards from --data-dir/shards (default artifacts/data_small, written
    by prepare_data_small.py). Falls back to synthetic random tokens when no
    shards are present. --tokenizer is used for the post-training generation
    check. Checkpoints/logs default to artifacts/{checkpoints,logs}/<config stem>.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch
from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.data import PretrainingDataset
from kda_moe.eval import generate
from kda_moe.fable import fable_pipeline, tokenize_sft_dataset
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import Trainer, load_checkpoint, save_checkpoint

GEN_PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "The theory of relativity",
]


class SFTTrainer(Trainer):
    """Trainer subclass that consumes pre-tokenized SFT data with loss masks."""

    def __init__(self, *args, sft_data: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_data = sft_data

    def train(self):
        if self.sft_data is None:
            return super().train()

        cfg = self.config
        model = self.model
        model.train()

        input_ids_all = self.sft_data["input_ids"]
        labels_all = self.sft_data["labels"]
        loss_mask_all = self.sft_data["loss_mask"]
        n = input_ids_all.shape[0]

        print(f"SFT: {n} examples, {cfg.total_steps} steps")
        accum_loss = 0.0
        step = 0
        for step in range(cfg.total_steps):
            lr = self._get_lr(step)
            self._set_lr(lr)

            accum_loss = 0.0
            self.optimizer.zero_grad()

            for micro_step in range(cfg.grad_accum_steps):
                idx = (step * cfg.grad_accum_steps + micro_step) % n
                input_ids = input_ids_all[idx : idx + 1].to(self.device)
                labels = labels_all[idx : idx + 1].to(self.device)
                mask = loss_mask_all[idx : idx + 1].to(self.device)

                if input_ids.shape[1] < 2:
                    continue

                inputs = input_ids[:, :-1]
                targets = labels[:, 1:]
                mask = mask[:, 1:]

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = model(inputs)
                    b, t, v = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(b * t, v),
                        targets.reshape(b * t),
                        ignore_index=-100,
                        reduction="none",
                    )
                    loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)
                    loss = loss / cfg.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                print(f"step {step:5d}/{cfg.total_steps} | loss {accum_loss:.4f} | lr {lr:.2e}")

            if step % cfg.checkpoint_interval == 0 and step > 0:
                save_checkpoint(
                    model,
                    self.optimizer,
                    step,
                    str(self.checkpoint_dir / f"step_{step}.pt"),
                    full=(step % cfg.full_checkpoint_interval == 0),
                    ema=self.ema,
                )

        return {
            "final_loss": accum_loss,
            "total_steps": step + 1,
            "tokens_processed": 0,
            "tok_per_sec": 0,
        }


def build_datasets(config: ModelConfig, data_dir: str, tokenizer_path: str):
    """Load tokenizer + shards from data_dir, or return (None, None, None) for synthetic."""
    shard_dir = Path(data_dir) / "shards"
    val_dir = Path(data_dir) / "val"

    tokenizer = None
    try:
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"Tokenizer: {tokenizer_path} ({tokenizer.get_vocab_size()} vocab)")
    except (FileNotFoundError, OSError):
        print(f"No tokenizer at {tokenizer_path}")

    shards = sorted(shard_dir.glob("*.jsonl")) if tokenizer else []
    if shards:
        seq = config.curriculum_start_seq
        train_ds = PretrainingDataset(
            str(shard_dir), tokenizer, seq_len=seq, data_mix=config.data_mix
        )
        val_ds = None
        val_shards = sorted(val_dir.glob("*.jsonl"))
        if val_shards:
            val_ds = PretrainingDataset(
                str(val_dir), tokenizer, seq_len=seq, data_mix=config.data_mix
            )
        print(f"Real data: {len(shards)} shards from {shard_dir}")
        return train_ds, val_ds, tokenizer

    print("No real shards found — training on synthetic random tokens")
    return None, None, tokenizer


def run_generation(model, tokenizer, device, max_new_tokens: int = 30):
    print("\n--- Generation check ---")
    if tokenizer is None:
        print("  (no tokenizer — skipped)")
        return
    for prompt in GEN_PROMPTS:
        completion = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device)
        print(f"  {prompt!r} -> {completion!r}")


def run_pretrain(config: ModelConfig, args) -> None:
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print(
        f"Config: {config.n_layers} layers, d={config.d_model}, "
        f"E={config.n_experts}, k={config.top_k}, seq={config.seq_len}, "
        f"steps={config.total_steps}"
    )

    print("Building model...")
    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.1f}M params")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_ds, val_ds, tokenizer = build_datasets(config, args.data_dir, args.tokenizer)

    print(f"\nStarting training ({config.total_steps} steps)...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=args.ckpt_dir,
        log_dir=args.log_dir,
    )

    try:
        stats = trainer.train()
    except RuntimeError as exc:
        if "OOM" not in str(exc):
            raise
        if args.colab:
            print("OOM! Halving sequence length and retrying...")
            torch.cuda.empty_cache()
            config.seq_len = max(128, config.seq_len // 2)
            config.curriculum_start_seq = min(config.curriculum_start_seq, config.seq_len)
            config.curriculum_milestones = [
                (m, min(s, config.seq_len)) for m, s in config.curriculum_milestones
            ]
            trainer = Trainer(
                model=model,
                config=config,
                train_dataset=train_ds,
                val_dataset=val_ds,
                checkpoint_dir=args.ckpt_dir,
                log_dir=args.log_dir,
            )
            stats = trainer.train()
        else:
            print(
                f"OOM! Reduce micro_batch_size/seq_len in the config, or use --colab "
                f"to auto-halve seq_len.\n{str(exc)[:500]}"
            )
            sys.exit(1)

    print(
        f"\nTraining complete: {stats['total_steps']} steps, "
        f"final_loss={stats['final_loss']:.4f}, tok/s={stats['tok_per_sec']:.0f}"
    )

    if not args.no_gen:
        run_generation(model, tokenizer, device)


def run_sft(config: ModelConfig, args) -> None:
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Building model...")
    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.1f}M params")

    base_ckpt = args.base_ckpt or find_latest_ckpt(args.ckpt_dir)
    if base_ckpt and Path(base_ckpt).exists():
        load_checkpoint(model, None, base_ckpt, device=device)
        print(f"Loaded base model from {base_ckpt}")
    else:
        print(f"Warning: base checkpoint not found at {base_ckpt}, starting from scratch")

    print("Loading FABLE traces...")
    traces = fable_pipeline(max_rows=args.sft_rows)
    print(f"Got {len(traces)} usable traces")

    tokenizer = load_tokenizer(args.tokenizer)
    sft_data = tokenize_sft_dataset(traces, tokenizer, max_seq_len=2048)
    print(f"Tokenized: {sft_data['input_ids'].shape[0]} sequences")

    sft_config = replace(
        config,
        lr=1e-5,
        total_steps=args.steps or 1000,
        warmup_steps=50,
        grad_accum_steps=4,
        checkpoint_interval=200,
        log_interval=20,
    )

    trainer = SFTTrainer(
        model=model,
        config=sft_config,
        checkpoint_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        sft_data=sft_data,
    )

    stats = trainer.train()
    print(f"\nSFT complete: loss={stats['final_loss']:.4f}")


def find_latest_ckpt(ckpt_dir: str) -> str | None:
    d = Path(ckpt_dir)
    lf = d / "latest.txt"
    if lf.exists():
        step = lf.read_text().strip()
        cp = d / f"step_{step}.pt"
        if cp.exists():
            return str(cp)
    pts = sorted(d.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pts[0]) if pts else None


def main():
    parser = argparse.ArgumentParser(description="Unified KDA-MoE training")
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "kda_moe_24m.toml"),
        help="Path to TOML config",
    )
    parser.add_argument("--data-dir", default="artifacts/data_small", help="dir with shards/ + val/")
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer_small/tokenizer.json", help="tokenizer.json path"
    )
    parser.add_argument("--ckpt-dir", default=None, help="default: artifacts/checkpoints/<config stem>")
    parser.add_argument("--log-dir", default=None, help="default: artifacts/logs/<config stem>")
    parser.add_argument("--steps", type=int, default=None, help="override total_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--no-gen", action="store_true", help="skip post-training generation check")
    parser.add_argument("--colab", action="store_true", help="Colab mode: mount Drive, CUDA assert, OOM retry")
    parser.add_argument("--sft", action="store_true", help="FABLE.5 SFT fine-tune instead of pretrain")
    parser.add_argument("--base-ckpt", default=None, help="base checkpoint for SFT (default: latest in ckpt-dir)")
    parser.add_argument("--sft-rows", type=int, default=5000, help="max FABLE traces for SFT")
    args = parser.parse_args()

    apply_triton_patch()
    torch.manual_seed(args.seed)

    config = ModelConfig.from_toml(args.config)
    if args.steps is not None:
        config.total_steps = args.steps

    stem = Path(args.config).stem
    args.ckpt_dir = args.ckpt_dir or f"artifacts/checkpoints/{stem}"
    args.log_dir = args.log_dir or f"artifacts/logs/{stem}"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.colab:
        if os.path.exists("/content"):
            try:
                from google.colab import drive

                drive.mount("/content/drive")
                args.ckpt_dir = f"/content/drive/MyDrive/kda-poc/{Path(args.ckpt_dir).name}"
            except Exception as exc:  # noqa: BLE001 - non-fatal in shell mode
                print(f"  (drive mount skipped: {exc})")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --colab training")

    if args.sft:
        run_sft(config, args)
    else:
        run_pretrain(config, args)


if __name__ == "__main__":
    main()
