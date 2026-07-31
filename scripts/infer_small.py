"""Interactive inference for the small 80M model.

Usage:
    uv run python scripts/infer_small.py              # auto-detect latest ckpt
    uv run python scripts/infer_small.py --cpu         # force CPU

Type 'quit' to exit, 'reload' to reload the latest checkpoint.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import ModelConfig
from kda_moe.eval import generate
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer

CKPT_DIR = "artifacts/checkpoints_small"
TOKENIZER_PATH = "artifacts/tokenizer_small/tokenizer.json"


def find_latest(checkpoint_dir: str) -> Path | None:
    d = Path(checkpoint_dir)
    lf = d / "latest.txt"
    if lf.exists():
        step = lf.read_text().strip()
        cp = d / f"step_{step}.pt"
        if cp.exists():
            return cp
    pts = sorted(d.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def load_model(ckpt_path: Path, config, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    step = state.get("step", 0)
    model_state = state.get("model", state)

    model = KDAMoEModel(config).to(device)

    # Filter out training-time buffers that have shape mismatches
    # (last_log_decay, last_router_logits — initialized as zeros(0) but saved
    #  with full shapes from the training step that produced the checkpoint)
    model_keys = dict(model.state_dict())
    filtered = {}
    skipped = 0
    for k, v in model_state.items():
        if k in model_keys and model_keys[k].shape != v.shape:
            skipped += 1
            continue
        filtered[k] = v
    if skipped:
        print(f"  Skipped {skipped} training-only buffer(s) with shape mismatch")

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (auto-detect if omitted)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = ModelConfig.preset_80m()
    config.seq_len = 512
    # Match training data_mix for completeness (only matters for weight init structure)
    total_cap = sum(config.data_caps_gb.values()) if config.data_caps_gb else 1.0
    config.data_mix = {k: v / total_cap for k, v in config.data_caps_gb.items()}

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"Tokenizer: {TOKENIZER_PATH} ({tokenizer.get_vocab_size()} vocab)")

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest(CKPT_DIR)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"No checkpoint found in {CKPT_DIR}. Run train_small.py first.")
        sys.exit(1)

    model, step = load_model(ckpt_path, config, device)
    total, _ = model.get_num_params()
    print(f"Loaded: {ckpt_path.name}  (step {step}, {total / 1e6:.1f}M params, {device})")
    print(f"Temp: {args.temp}  Top-K: {args.top_k}  Max tokens: {args.max_tokens}")
    print("Commands: 'reload' | 'quit'")
    print("-" * 50)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "reload":
            latest = find_latest(CKPT_DIR)
            if latest:
                model, step = load_model(latest, config, device)
                print(f"Reloaded: {latest.name} (step {step})")
            else:
                print("No new checkpoint.")
            continue

        t0 = time.time()
        completion = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t0
        print(completion)
        print(f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
