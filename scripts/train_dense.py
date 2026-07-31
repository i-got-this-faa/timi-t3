"""P6 gate: train dense-100m control baseline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.data import PretrainingDataset
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "dense_100m.toml"),
        help="Path to TOML config",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ModelConfig.from_toml(args.config)
    print(
        f"Dense control: {config.n_layers} layers, d={config.d_model}, "
        f"KDA={config.use_kda}, MoE={config.use_moe}"
    )

    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.1f}M params")

    # Load data if available
    data_dir = Path("artifacts/data")
    train_dir = data_dir / "shards"
    if list(train_dir.glob("*.jsonl")):
        tokenizer = load_tokenizer("artifacts/tokenizer/tokenizer.json")
        train_dataset = PretrainingDataset(
            str(train_dir),
            tokenizer,
            seq_len=config.curriculum_start_seq,
        )
    else:
        print("No shards found — using synthetic data")
        train_dataset = None

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        checkpoint_dir="artifacts/checkpoints_dense",
        log_dir="artifacts/logs_dense",
    )

    stats = trainer.train()
    print(
        f"\nDense control complete: loss={stats['final_loss']:.4f}, "
        f"tok/s={stats['tok_per_sec']:.0f}"
    )


if __name__ == "__main__":
    main()
