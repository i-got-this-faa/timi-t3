"""Interactive prompt loop for testing KDA-MoE checkpoints.

Usage:
    python scripts/infer.py                           # auto-detect latest ckpt
    python scripts/infer.py --ckpt step_20.pt         # specific checkpoint
    python scripts/infer.py --watch                   # auto-reload on new ckpts

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
    # fallback: find newest .pt
    pts = sorted(d.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def load_model(ckpt_path: Path, config: ModelConfig, device: torch.device):
    model = KDAMoEModel(config).to(device)
    model.eval()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
        step = ckpt.get("step", "?")
    else:
        step = ckpt_path.stem.replace("step_", "")
    model.load_state_dict(ckpt, strict=False)
    return model, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default="configs/kda_moe_1b.toml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--watch", action="store_true", help="Monitor checkpoints dir, auto-reload on new"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = (
        ModelConfig.from_toml(args.config)
        if Path(args.config).exists()
        else ModelConfig.preset_1b()
    )
    config.seq_len = 2048  # override for eval

    # Tokenizer
    try:
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Tokenizer loaded: {args.tokenizer}")
    except FileNotFoundError:
        print(f"No tokenizer at {args.tokenizer} — using dummy (random tokens only)")
        print("Train a tokenizer first: python scripts/prepare_data.py")
        tokenizer = None

    # Default to preset_1b (96 experts). Only use TOML if explicitly passed.
    if args.config != "configs/kda_moe_1b.toml" and Path(args.config).exists():
        config = ModelConfig.from_toml(args.config)
    else:
        config = ModelConfig.preset_1b()
    config.seq_len = 2048

    ckpt_dir = "artifacts/checkpoints_1b"
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest(ckpt_dir)
    if ckpt_path is None:
        print(f"No checkpoint found in {ckpt_dir}")
        sys.exit(1)

    model, step = load_model(ckpt_path, config, device)
    total, _ = model.get_num_params()
    print(f"Loaded: {ckpt_path.name} (step {step}, {total / 1e6:.1f}M params)")

    last_ckpt = ckpt_path
    last_mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0

    print(f"Device: {device}")
    print(f"Temp: {args.temp}, Top-K: {args.top_k}, Max tokens: {args.max_tokens}")
    print("Commands: 'reload' (latest ckpt), 'watch' (auto mode), 'quit'")
    print("-" * 50)

    watch_mode = args.watch

    while True:
        if watch_mode:
            # Check for new checkpoint
            latest = find_latest(ckpt_dir)
            if latest and latest.stat().st_mtime > last_mtime:
                print(f"\n[watch] New checkpoint: {latest.name}")
                model, step = load_model(latest, config, device)
                last_mtime = latest.stat().st_mtime
                print(f"[watch] Reloaded step {step}")
            time.sleep(2)
            continue

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
            if latest and latest != last_ckpt:
                model, step = load_model(latest, config, device)
                last_ckpt = latest
                print(f"Reloaded: {latest.name} (step {step})")
            else:
                print("No new checkpoint.")
            continue
        if prompt.lower() == "watch":
            watch_mode = True
            print("[watch] Monitoring for new checkpoints... (Ctrl+C to stop)")
            continue

        # Generate
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
            # Dummy generation: just forward the prompt
            encoded = torch.randint(0, config.vocab_size, (1, min(len(prompt), 512)), device=device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(encoded)
            completion = "[no tokenizer — logits shape: {}]".format(logits.shape)

        elapsed = time.time() - t0
        print(completion)
        print(f"({elapsed:.1f}s, {len(completion.split())} tokens)")


if __name__ == "__main__":
    main()
