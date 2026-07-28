"""Interactive prompt loop for testing KDA-MoE checkpoints.

Usage:
    python scripts/infer.py                           # auto-detect latest ckpt
    python scripts/infer.py --ckpt step_40.pt         # specific checkpoint
    python scripts/infer.py --cpu                     # force CPU (for low-VRAM)

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
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.eval import generate


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


def load_model(ckpt_path: Path, config: ModelConfig, device: torch.device):
    """Load checkpoint, build model on CPU first to avoid OOM on small GPUs."""
    model = KDAMoEModel(config)  # CPU
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    step = ckpt_path.stem.replace("step_", "")
    return model, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = ModelConfig.preset_1b()
    if args.config and Path(args.config).exists():
        config = ModelConfig.from_toml(args.config)
    config.seq_len = 2048

    try:
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Tokenizer: {args.tokenizer}")
    except FileNotFoundError:
        print(f"No tokenizer at {args.tokenizer}")
        tokenizer = None

    ckpt_dir = "artifacts/checkpoints_1b"
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest(ckpt_dir)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"No checkpoint found")
        sys.exit(1)

    model, step = load_model(ckpt_path, config, device)
    total, _ = model.get_num_params()
    print(f"Loaded: {ckpt_path.name} (step {step}, {total / 1e6:.1f}M params, {device})")
    print(f"Temp: {args.temp}, Top-K: {args.top_k}, Max tokens: {args.max_tokens}")
    print("Commands: 'reload', 'quit'")
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
            latest = find_latest(ckpt_dir)
            if latest:
                model, step = load_model(latest, config, device)
                print(f"Reloaded: {latest.name} (step {step})")
            else:
                print("No new checkpoint.")
            continue

        t0 = time.time()
        if tokenizer:
            completion = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                device=device,
            )
        else:
            completion = "[no tokenizer]"
        elapsed = time.time() - t0
        print(completion)
        print(f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
