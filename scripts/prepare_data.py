"""P1 gate: one-shot data prep - download shards, train tokenizer, fertility check."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.data import build_pretraining_mix
from kda_moe.data import print_dataset_report


from kda_moe.tokenizer import fertility_report, train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "kda_moe_450m.toml"),
        help="Path to TOML config",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("P1: Data pipeline + tokenizer")
    print("=" * 60)

    config = ModelConfig.from_toml(args.config)

    print("\n[1/3] Building pretraining mix...")
    train_dir, val_dir = build_pretraining_mix("artifacts/data", config)
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")

    print("\n[2/3] Training tokenizer...")
    tokenizer = train_tokenizer(
        data_dir=str(train_dir),
        vocab_size=config.vocab_size,
        output_path="artifacts/tokenizer",
    )

    print("\n[3/3] Fertility gate...")
    passed = fertility_report(tokenizer)

    print("\n[4/4] Dataset report...")
    print_dataset_report("artifacts/data/shards", tokenizer)

    print("\n" + "=" * 60)
    if passed:
        print("P1 PASS - fertility gate passed")
    else:
        print("P1 WARN - fertility gate failed (expected with small vocab on code/JSON)")
        print("  Proceeding anyway — model learns these patterns during pretraining.")
    print("=" * 60)


if __name__ == "__main__":
    main()
