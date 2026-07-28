"""P5 gate: Colab training entry point for the 1B KDA-MoE model.

Usage (Colab):
    %cd /content/drive/MyDrive/kda-poc
    !python scripts/train_colab.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import ModelConfig
from kda_moe.data import PretrainingDataset, build_pretraining_mix
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import Trainer


def main():
    # Environment
    repo_dir = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parent.parent))
    ckpt_dir = Path(os.environ.get("CKPT_DIR", "artifacts/checkpoints_1b"))
    data_dir = Path(os.environ.get("DATA_DIR", "data"))

    print(f"Repo: {repo_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Data: {data_dir}")

    # On Colab, mount Drive if needed
    if os.path.exists("/content") and not ckpt_dir.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            ckpt_dir = Path("/content/drive/MyDrive/kda-poc/artifacts/checkpoints_1b")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for 1B training"
    print(f"Device: {device} — {torch.cuda.get_device_name(0)}")

    # Config
    config = ModelConfig.preset_1b()
    print(f"Config: {config.n_layers} layers, d={config.d_model}, "
          f"E={config.n_experts}, k={config.top_k}")

    # Data
    train_dir = data_dir / "shards"
    val_dir = data_dir / "val"

    if not list(train_dir.glob("*.jsonl")):
        print("No shards found. Run scripts/prepare_data.py first.")
        print("Using synthetic data for smoke test...")
        train_dataset = None
        val_dataset = None
    else:
        tokenizer = load_tokenizer("artifacts/tokenizer/tokenizer.json")
        train_dataset = PretrainingDataset(
            str(train_dir), tokenizer, seq_len=config.curriculum_start_seq,
        )
        val_dataset = PretrainingDataset(
            str(val_dir), tokenizer, seq_len=config.curriculum_start_seq,
        )

    # Model
    print("Building model...")
    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total/1e6:.1f}M params")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Train
    print(f"\nStarting 1B pilot training ({config.total_steps} steps)...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(ckpt_dir.parent / "logs_1b"),
    )

    try:
        stats = trainer.train()
        print(f"\nTraining complete: {stats['total_steps']} steps, "
              f"final_loss={stats['final_loss']:.4f}")
    except RuntimeError as e:
        if "OOM" in str(e):
            print("OOM! Reducing sequence length and retrying...")
            torch.cuda.empty_cache()
            config.seq_len = 1024
            config.curriculum_milestones = [
                (m[0], min(m[1], 1024)) for m in config.curriculum_milestones
            ]
            trainer = Trainer(
                model=model, config=config,
                train_dataset=train_dataset, val_dataset=val_dataset,
                checkpoint_dir=str(ckpt_dir),
                log_dir=str(ckpt_dir.parent / "logs_1b"),
            )
            stats = trainer.train()
        else:
            raise


if __name__ == "__main__":
    main()
