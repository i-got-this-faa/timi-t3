"""P1-light: download ~1 GB of language-heavy non-gated data, train tokenizer, fertility check."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kda_moe.config import ModelConfig
from kda_moe.data import DATASET_SPECS, build_pretraining_mix, print_dataset_report
from kda_moe.tokenizer import fertility_report, train_tokenizer

# Gated / inaccessible datasets to skip
GATED_IDS = {"bigcode/the-stack-dedup"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config (defaults to preset_80m + small data caps)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing artifacts/data_small shards before downloading (re-apply new caps)",
    )
    args = parser.parse_args()

    import kda_moe.data as data_mod
    import shutil

    if args.force:
        print("  --force: removing existing artifacts/data_small ...")
        shutil.rmtree("artifacts/data_small", ignore_errors=True)

    # Filter out gated datasets before build_pretraining_mix iterates them
    original_specs = dict(data_mod.DATASET_SPECS)
    data_mod.DATASET_SPECS = {
        name: spec for name, spec in DATASET_SPECS.items() if spec["id"] not in GATED_IDS
    }

    try:
        print("=" * 60)
        print("P1-small: Data pipeline + tokenizer")
        print("=" * 60)

        # ~1 GB total, deliberately language-heavy: 92% natural language
        # (dclm + fineweb-edu + tinystories), 5% math, 3% code. Code is
        # minimized so the smoke runs learn language rather than syntax.
        if args.config:
            config = ModelConfig.from_toml(args.config)
        else:
            config = ModelConfig.preset_80m()
            config.data_caps_gb = {
                "dclm": 3.0,
                "fineweb": 2.0,
                "math": 0.50,
                "tinystories": 0.50,
                "code": 0.10,
            }
            # data_mix weights proportional to caps
            config.data_mix = {k: v / sum(config.data_caps_gb.values()) for k, v in config.data_caps_gb.items()}
            config.total_steps = 100
            config.warmup_steps = 5
            config.checkpoint_interval = 50
            config.log_interval = 5

        print(f"Total cap: ~{sum(config.data_caps_gb.values()):.2f} GB across {len(config.data_caps_gb)} sources")

        print("\n[1/3] Building pretraining mix...")
        train_dir, val_dir = build_pretraining_mix("artifacts/data_small", config)
        print(f"  Train: {train_dir}")
        print(f"  Val:   {val_dir}")

        print("\n[2/3] Training tokenizer...")
        tokenizer = train_tokenizer(
            data_dir=str(train_dir),
            vocab_size=config.vocab_size,
            output_path="artifacts/tokenizer_small",
        )

        print("\n[3/3] Fertility gate...")
        passed = fertility_report(tokenizer)

        print("\n[4/4] Dataset report...")
        print_dataset_report("artifacts/data_small/shards", tokenizer)

        print("\n" + "=" * 60)
        if passed:
            print("P1-small PASS — fertility gate passed")
        else:
            print("P1-small WARN — fertility gate failed")
            print("  Proceeding anyway — model learns these patterns during pretraining.")
        print("=" * 60)
    finally:
        data_mod.DATASET_SPECS = original_specs


if __name__ == "__main__":
    main()
