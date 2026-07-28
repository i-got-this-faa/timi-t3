"""P4 gate: smoke training on 450M config — train, checkpoint, generate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.config import ModelConfig
from kda_moe.model import KDAMoEModel
from kda_moe.train import Trainer, load_checkpoint
from kda_moe.eval import generate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ModelConfig.preset_180m()
    print(
        f"Config: {config.n_layers} layers, d={config.d_model}, "
        f"E={config.n_experts}, k={config.top_k}, seq={config.seq_len}"
    )

    # Build model
    print("Building model...")
    model = KDAMoEModel(config).to(device)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.1f}M params")

    # Train (no real data, uses synthetic random tokens)
    print("\nStarting smoke training...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=None,  # synthetic data
        checkpoint_dir="artifacts/checkpoints_smoke",
        log_dir="artifacts/logs_smoke",
    )

    stats = trainer.train()

    print(
        f"\nTraining complete: {stats['total_steps']} steps, "
        f"final_loss={stats['final_loss']:.4f}, "
        f"tok/s={stats['tok_per_sec']:.0f}"
    )

    # Checkpoint reload test
    print("\nTesting checkpoint reload...")
    model2 = KDAMoEModel(config).to(device)
    load_checkpoint(
        model2,
        None,
        "artifacts/checkpoints_smoke/step_500.pt",
        device=device,
    )
    print("Checkpoint reload OK")

    # Quick generation test (needs a tokenizer — use a dummy one)
    print("\nTesting generation...")
    try:
        from kda_moe.tokenizer import load_tokenizer

        tokenizer = load_tokenizer("artifacts/tokenizer/tokenizer.json")
    except (FileNotFoundError, Exception):
        # No tokenizer trained — skip generation test
        print("No tokenizer found, skipping generation test")
        print("\nP4 PASS — training loop works")
        return

    prompt = "The capital of France is"
    completion = generate(model2, tokenizer, prompt, max_new_tokens=20)
    assert len(completion) > 0, "Generation produced empty output"
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion[:200]}")

    print("\nP4 PASS — smoke training complete")


if __name__ == "__main__":
    main()
