"""Training loop with curriculum, checkpointing, monitoring, resume support."""

from __future__ import annotations

from .compat import apply_triton_patch

apply_triton_patch()

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import ModelConfig
from .model import KDAMoEModel


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    path: str,
    full: bool = False,
) -> None:
    """Save model state_dict. Optionally save optimizer state."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out.with_suffix(".pt")))

    if full and optimizer is not None:
        full_path = out.parent / f"{out.stem}_full.pt"
        torch.save(
            {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            str(full_path),
        )

    (out.parent / "latest.txt").write_text(str(step))


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device | None = None,
) -> int:
    """Load model + optimizer state. Return step number."""
    p = Path(path)
    for ext in ("", ".pt", ".safetensors"):
        candidate = p.with_suffix(ext) if ext else p
        if candidate.exists():
            p = candidate
            break
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(p), map_location=device or "cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt.get("step", 0)
    model.load_state_dict(ckpt, strict=False)
    return 0


def _find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str] | None:
    d = Path(checkpoint_dir)
    lf = d / "latest.txt"
    if lf.exists():
        try:
            step = int(lf.read_text().strip())
            cp = d / f"step_{step}.pt"
            if cp.exists():
                return step, str(cp)
        except (ValueError, OSError):
            pass
    best = -1
    best_path = ""
    for f in sorted(d.glob("step_*.pt")):
        try:
            s = int(f.stem.replace("step_", "").split("_")[0])
            if s > best:
                best = s
                best_path = str(f)
        except ValueError:
            continue
    return (best, best_path) if best >= 0 else None


class Trainer:
    """Single-GPU training loop with curriculum, checkpointing, monitoring."""

    def __init__(
        self,
        model: KDAMoEModel,
        config: ModelConfig,
        train_dataset: torch.utils.data.IterableDataset | None = None,
        val_dataset: torch.utils.data.IterableDataset | None = None,
        checkpoint_dir: str = "artifacts/checkpoints",
        log_dir: str = "artifacts/logs",
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = next(model.parameters()).device
        self.writer = SummaryWriter(str(log_dir))
        self.step = 0
        self.current_seq_len = config.curriculum_start_seq

        self._setup_optimizer()
        self._try_resume()

        self.scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need it
        self.tokens_processed = 0
        self.start_time = time.time()
        self.router_entropy_history: list[float] = []

    def _setup_optimizer(self):
        cfg = self.config
        try:
            import bitsandbytes as bnb

            self.optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
            self.use_8bit = True
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
            self.use_8bit = False

    def _try_resume(self):
        latest = _find_latest_checkpoint(str(self.checkpoint_dir))
        if latest is None:
            return
        resume_step, ckpt_path = latest
        print(f"Resuming from step {resume_step}: {ckpt_path}")
        loaded = load_checkpoint(self.model, self.optimizer, ckpt_path, device=self.device)
        self.step = max(resume_step, loaded)
        for ms, sl in self.config.curriculum_milestones:
            if self.step >= ms:
                self.current_seq_len = sl

    def _update_curriculum(self):
        for ms, sl in self.config.curriculum_milestones:
            if self.step == ms:
                self.current_seq_len = sl
                print(f"  Curriculum: seq_len -> {sl}")

    def _get_lr(self, step: int) -> float:
        cfg = self.config
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _collect_router_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for i, layer in enumerate(self.model.layers):
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "router"):
                r = ffn.router
                bias = r.expert_bias
                stats[f"bias_mean_{i}"] = bias.mean().item()
                stats[f"bias_min_{i}"] = bias.min().item()
                stats[f"bias_max_{i}"] = bias.max().item()
                dead = (bias < -1.0).sum().item()
                stats[f"dead_{i}"] = dead

                # Router entropy from last logits
                if ffn.last_router_logits.numel() > 0:
                    entropy = r.router_entropy(ffn.last_router_logits)
                    stats[f"router_entropy_{i}"] = entropy

                # Expert load: fraction of tokens routed to each expert
                if ffn.last_router_logits.numel() > 0:
                    scores = torch.sigmoid(ffn.last_router_logits + bias)
                    _, top_idx = torch.topk(scores, r.top_k, dim=-1)
                    counts = torch.bincount(top_idx.flatten(), minlength=r.n_experts).float()
                    total = counts.sum()
                    if total > 0:
                        load = counts / total
                        stats[f"expert_load_min_{i}"] = load.min().item()
                        stats[f"expert_load_max_{i}"] = load.max().item()
                        stats[f"expert_load_std_{i}"] = load.std().item()

            # KDA decay statistics
            attn = getattr(layer, "attn", None)
            if (
                attn is not None
                and hasattr(attn, "last_log_decay")
                and attn.last_log_decay.numel() > 0
            ):
                ld = attn.last_log_decay
                stats[f"kda_decay_min_{i}"] = ld.min().item()
                stats[f"kda_decay_max_{i}"] = ld.max().item()
                stats[f"kda_decay_mean_{i}"] = ld.mean().item()

        return stats

    def train(self) -> dict[str, Any]:
        cfg = self.config
        model = self.model
        model.train()

        from .display import StatusDisplay

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        vram_total = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if self.device.type == "cuda"
            else 0
        )

        total, _ = model.get_num_params()
        print(
            f"KDA-MoE 1B  |  {total / 1e6:.1f}M params  |  {cfg.total_steps} steps  |  "
            f"8bit Adam={'on' if self.use_8bit else 'off'}",
            flush=True,
        )

        display = StatusDisplay(cfg.total_steps, gpu_name, vram_total)
        train_iter = iter(self.train_dataset) if self.train_dataset else None
        losses: list[float] = []

        while self.step < cfg.total_steps:
            self._update_curriculum()
            lr = self._get_lr(self.step)
            self._set_lr(lr)

            accum_loss = 0.0
            self.optimizer.zero_grad()
            step_t0 = time.time()

            for _ in range(cfg.grad_accum_steps):
                if train_iter is None:
                    input_ids = torch.randint(
                        0,
                        cfg.vocab_size,
                        (cfg.micro_batch_size, self.current_seq_len),
                        device=self.device,
                    )
                else:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataset)
                        batch = next(train_iter)
                    input_ids = batch.to(self.device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)

                targets = input_ids[:, 1:]
                inputs = input_ids[:, :-1]
                if inputs.shape[1] < 1:
                    continue

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = model(inputs)
                    loss = model.loss_fn(logits, targets, ignore_index=-100)
                    loss = loss / cfg.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            self.optimizer.step()

            self.step += 1
            tokens = cfg.micro_batch_size * self.current_seq_len * cfg.grad_accum_steps
            self.tokens_processed += tokens

            step_time = time.time() - step_t0
            elapsed = max(time.time() - self.start_time, 0.01)
            tok_per_sec = self.tokens_processed / elapsed
            grad_norm = (
                sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
                ** 0.5
            )

            rs = self._collect_router_stats()
            dead = sum(v for k, v in rs.items() if k.startswith("dead_"))
            active = cfg.n_experts - dead

            # Average router entropy across MoE layers for display
            entropies = [v for k, v in rs.items() if k.startswith("router_entropy_")]
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

            vram_used = (
                torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0
            )
            vram_peak = (
                torch.cuda.max_memory_allocated(self.device) / 1e9
                if self.device.type == "cuda"
                else 0
            )
            epoch = self.tokens_processed / 2.5e9

            display.update(
                step=self.step,
                loss=accum_loss,
                lr=lr,
                grad_norm=grad_norm,
                tok_per_sec=tok_per_sec,
                step_time=step_time,
                seq_len=self.current_seq_len,
                vram_used=vram_used,
                vram_peak=vram_peak,
                active_experts=active,
                total_experts=cfg.n_experts,
                router_entropy=avg_entropy,
                epoch=epoch,
            )

            if self.step % cfg.log_interval == 0:
                self.writer.add_scalar("train/loss", accum_loss, self.step)
                self.writer.add_scalar("train/lr", lr, self.step)
                self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
                self.writer.add_scalar("train/tok_per_sec", tok_per_sec, self.step)
                for k, v in rs.items():
                    self.writer.add_scalar(f"router/{k}", v, self.step)
                losses.append(accum_loss)

            if self.step % cfg.checkpoint_interval == 0 and self.step > 0:
                save_checkpoint(
                    model,
                    self.optimizer,
                    self.step,
                    str(self.checkpoint_dir / f"step_{self.step}.pt"),
                    full=(self.step % cfg.full_checkpoint_interval == 0),
                )

            if not torch.isfinite(torch.tensor(accum_loss)):
                raise RuntimeError(f"NaN loss at step {self.step}")

        save_checkpoint(
            model,
            self.optimizer,
            self.step,
            str(self.checkpoint_dir / f"step_{self.step}.pt"),
            full=True,
        )
        self.writer.close()

        return {
            "final_loss": losses[-1] if losses else float("nan"),
            "total_steps": self.step,
            "tokens_processed": self.tokens_processed,
            "tok_per_sec": self.tokens_processed / max(time.time() - self.start_time, 0.001),
        }
