"""Interactive inference for any KDA-MoE checkpoint.

Usage:
    python scripts/infer.py                                      # latest ckpt under artifacts/checkpoints
    python scripts/infer.py --config configs/kda_moe_24m.toml    # matching config
    python scripts/infer.py --ckpt artifacts/checkpoints/kda_moe_24m/step_200.pt
    python scripts/infer.py --tokenizer artifacts/tokenizer_small/tokenizer.json
    python scripts/infer.py --cpu                                # force CPU (low VRAM)
    python scripts/infer.py --prompt "Once upon a time," --max-tokens 64   # one-shot, no REPL

REPL commands: 'quit' to exit, 'reload' to reload the latest checkpoint.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch
from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.eval import generate
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer


def find_latest(checkpoint_dir: str) -> Path | None:
    """Newest step_*.pt in checkpoint_dir, recursing into subdirs if needed."""
    d = Path(checkpoint_dir)
    if d.exists():
        lf = d / "latest.txt"
        if lf.exists():
            step = lf.read_text().strip()
            cp = d / f"step_{step}.pt"
            if cp.exists():
                return cp
        pts = sorted(d.rglob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
    return None


def load_model(ckpt_path: Path, config: ModelConfig, device: torch.device):
    """Build on CPU, filter training-only buffers, then move to device."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    step = 0
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        step = ckpt.get("step", 0)
        ckpt = ckpt["model_state"]

    model = KDAMoEModel(config)  # CPU
    model_keys = dict(model.state_dict())
    filtered = {}
    skipped = 0
    for k, v in ckpt.items():
        if k in model_keys and model_keys[k].shape != v.shape:
            skipped += 1
            continue
        filtered[k] = v
    if skipped:
        print(f"  Skipped {skipped} training-only buffer(s) with shape mismatch")

    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    if not step:
        try:
            step = int(ckpt_path.stem.replace("step_", "").split("_")[0])
        except ValueError:
            step = 0
    return model, step


def main():
    parser = argparse.ArgumentParser(description="KDA-MoE interactive inference")
    parser.add_argument("--config", default=str(CONFIG_DIR / "kda_moe_24m.toml"))
    parser.add_argument("--ckpt", default=None, help="explicit checkpoint path (default: latest)")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints", help="search dir for latest ckpt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer_small/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt", default=None, help="one-shot generation, then exit")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    apply_triton_patch()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = ModelConfig.from_toml(args.config)

    try:
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Tokenizer: {args.tokenizer} ({tokenizer.get_vocab_size()} vocab)")
    except (FileNotFoundError, OSError):
        print(f"No tokenizer at {args.tokenizer} — prompts will be echoed only")
        tokenizer = None

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest(args.ckpt_dir)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"No checkpoint found (searched {args.ckpt_dir}). Run scripts/train.py first.")
        sys.exit(1)

    model, step = load_model(ckpt_path, config, device)
    total, _ = model.get_num_params()
    print(f"Loaded: {ckpt_path.name} (step {step}, {total / 1e6:.1f}M params, {device})")
    print(f"Temp: {args.temp}, Top-K: {args.top_k}, Max tokens: {args.max_tokens}")
    print("-" * 50)

    if args.prompt:
        t0 = time.time()
        completion = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device,
        )
        elapsed = time.time() - t0
        print(completion)
        print(f"({elapsed:.1f}s)")
        return

    print("Commands: 'reload' | 'quit'")
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
            latest = find_latest(args.ckpt_dir)
            if latest:
                model, step = load_model(latest, config, device)
                print(f"Reloaded: {latest.name} (step {step})")
            else:
                print("No new checkpoint.")
            continue

        t0 = time.time()
        if tokenizer:
            completion = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device,
            )
        else:
            completion = "[no tokenizer]"
        elapsed = time.time() - t0
        print(completion)
        print(f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
