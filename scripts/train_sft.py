"""P7: SFT fine-tuning on FABLE.5 traces."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import CONFIG_DIR, ModelConfig
from kda_moe.fable import fable_pipeline, tokenize_sft_dataset
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import Trainer, load_checkpoint


class SFTTrainer(Trainer):
    """Subclass of Trainer that accepts pre-tokenized SFT dataset with loss masks."""

    def __init__(self, *args, sft_data: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_data = sft_data

    def train(self):
        """Override to use masked loss on SFT data."""
        if self.sft_data is None:
            return super().train()

        cfg = self.config
        model = self.model
        model.train()

        input_ids_all = self.sft_data["input_ids"]
        labels_all = self.sft_data["labels"]
        loss_mask_all = self.sft_data["loss_mask"]
        N = input_ids_all.shape[0]

        print(f"SFT: {N} examples, {cfg.total_steps} steps")
        total, _ = model.get_num_params()
        print(f"Model: {total/1e6:.1f}M params")

        for step in range(cfg.total_steps):
            self.step = step
            lr = self._get_lr(step)
            self._set_lr(lr)

            accum_loss = 0.0
            self.optimizer.zero_grad()

            for micro_step in range(cfg.grad_accum_steps):
                idx = (step * cfg.grad_accum_steps + micro_step) % N
                input_ids = input_ids_all[idx:idx+1].to(self.device)
                labels = labels_all[idx:idx+1].to(self.device)
                mask = loss_mask_all[idx:idx+1].to(self.device)

                if input_ids.shape[1] < 2:
                    continue

                targets = labels[:, 1:]
                inputs = input_ids[:, :-1]
                mask = mask[:, 1:]

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = model(inputs)
                    B, T, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(B * T, V),
                        targets.reshape(B * T),
                        ignore_index=-100,
                        reduction="none",
                    )
                    loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)
                    loss = loss / cfg.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                print(f"step {step:5d}/{cfg.total_steps} | loss {accum_loss:.4f} | lr {lr:.2e}")

            if step % cfg.checkpoint_interval == 0 and step > 0:
                save_checkpoint(
                    model, self.optimizer, step,
                    str(self.checkpoint_dir / f"step_{step}.pt"),
                    full=(step % cfg.full_checkpoint_interval == 0),
                )

        return {"final_loss": accum_loss, "total_steps": step, "tokens_processed": 0, "tok_per_sec": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "kda_moe_1b.toml"),
        help="Path to TOML config for the base model",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model
    config = ModelConfig.from_toml(args.config)
    base_ckpt = "artifacts/checkpoints_1b/step_10000.pt"
    model = KDAMoEModel(config).to(device)

    if Path(base_ckpt).exists():
        load_checkpoint(model, None, base_ckpt, device=device)
        print(f"Loaded base model from {base_ckpt}")
    else:
        print(f"Warning: base checkpoint not found at {base_ckpt}, starting from scratch")

    # Load and tokenize FABLE traces
    print("Loading FABLE traces...")
    traces = fable_pipeline(max_rows=5000)
    print(f"Got {len(traces)} usable traces")

    tokenizer = load_tokenizer("artifacts/tokenizer/tokenizer.json")
    sft_data = tokenize_sft_dataset(traces, tokenizer, max_seq_len=2048)
    print(f"Tokenized: {sft_data['input_ids'].shape[0]} sequences")

    # SFT config (model shape from base config; SFT-specific schedule)
    sft_config = replace(
        config,
        lr=1e-5,
        total_steps=1000,
        warmup_steps=50,
        grad_accum_steps=4,
        checkpoint_interval=200,
        log_interval=20,
    )

    trainer = SFTTrainer(
        model=model,
        config=sft_config,
        checkpoint_dir="artifacts/checkpoints_sft",
        log_dir="artifacts/logs_sft",
        sft_data=sft_data,
    )

    stats = trainer.train()
    print(f"\nSFT complete: loss={stats['final_loss']:.4f}")


if __name__ == "__main__":
    main()
