"""Quick 80M model training on ~150 MB of real data — for local inference testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import ModelConfig
from kda_moe.data import PretrainingDataset
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ──────────────────────────────────────────────
    config = ModelConfig.preset_80m()
    # Match data_caps_gb from prepare_data_small.py for weighted sampling
    config.data_caps_gb = {
        "dclm": 0.05,
        "fineweb": 0.04,
        "code": 0.04,
        "math": 0.05,
        "tinystories": 0.03,
    }
    total_cap = sum(config.data_caps_gb.values())
    config.data_mix = {k: v / total_cap for k, v in config.data_caps_gb.items()}
    config.total_steps = 200
    config.warmup_steps = 20
    config.checkpoint_interval = 100
    config.log_interval = 10
    config.seq_len = 512
    config.curriculum_start_seq = 512

    print(
        f"Config: {config.n_layers} layers, d={config.d_model}, "
        f"E={config.n_experts}, k={config.top_k}, seq={config.seq_len}, "
        f"steps={config.total_steps}"
    )

    # ── Data ────────────────────────────────────────────────
    tokenizer = load_tokenizer("artifacts/tokenizer_small/tokenizer.json")
    print(f"Tokenizer: {tokenizer.get_vocab_size()} vocab")

    train_ds = PretrainingDataset(
        "artifacts/data_small/shards",
        tokenizer,
        seq_len=config.seq_len,
        data_mix=config.data_mix,
    )

    # ── Model ───────────────────────────────────────────────
    print("Building model...")
    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.1f}M params")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Train ───────────────────────────────────────────────
    print(f"\nStarting training ({config.total_steps} steps)...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        checkpoint_dir="artifacts/checkpoints_small",
        log_dir="artifacts/logs_small",
    )

    try:
        stats = trainer.train()
    except RuntimeError as e:
        if "OOM" in str(e):
            print(f"OOM! Try reducing micro_batch_size or seq_len.\n{str(e)[:500]}")
            sys.exit(1)
        raise

    print(
        f"\nTraining complete: {stats['total_steps']} steps, "
        f"final_loss={stats['final_loss']:.4f}, "
        f"tok/s={stats['tok_per_sec']:.0f}"
    )

    # ── Generate ────────────────────────────────────────────
    print("\n--- Generation test ---")
    from kda_moe.eval import generate

    prompts = [
        "The capital of France is",
        "Once upon a time,",
        "def fibonacci(n):",
        "The theory of relativity",
    ]
    for prompt in prompts:
        completion = generate(model, tokenizer, prompt, max_new_tokens=30)
        print(f"  Prompt: {prompt}")
        print(f"  Output: {completion}")
        print()

    print("Done — model ready at artifacts/checkpoints_small/")


if __name__ == "__main__":
    main()
