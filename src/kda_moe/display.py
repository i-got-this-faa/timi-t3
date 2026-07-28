"""Rich in-place status display — overdraws same block each step via ANSI codes."""

from __future__ import annotations

import shutil
import time
from typing import Any


class StatusDisplay:
    """Renders a dashboard that overwrites itself each update (no scroll)."""

    def __init__(self, total_steps: int, gpu_name: str = "", vram_total: float = 0):
        self.total_steps = total_steps
        self.gpu_name = gpu_name
        self.vram_total = vram_total
        self.start_time = time.time()

    def update(
        self,
        step: int = 0,
        loss: float = 0,
        lr: float = 0,
        grad_norm: float = 0,
        tok_per_sec: float = 0,
        step_time: float = 0,
        seq_len: int = 0,
        vram_used: float = 0,
        vram_peak: float = 0,
        active_experts: int = 0,
        total_experts: int = 0,
        router_entropy: float = 0,
        epoch: float = 0,
        val_loss: float | None = None,
        best_val: float | None = None,
        **_: Any,
    ):
        w = min(shutil.get_terminal_size().columns, 100)
        bar = "─" * (w - 2)

        # GPU/torch info
        torch_ver = ""
        try:
            import torch

            tv = torch.__version__.split("+")[0]
            cv = torch.version.cuda or "?"
            torch_ver = f"{tv} CUDA {cv}"
        except Exception:
            pass

        eta_s = (self.total_steps - step) * max(step_time, 0.01)
        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)

        tk = f"{tok_per_sec / 1000:.1f}k" if tok_per_sec > 1000 else f"{tok_per_sec:.0f}"

        loss_str = f"loss     {loss:.4f}"
        if val_loss is not None:
            loss_str += f"  val {val_loss:.4f}"
            if best_val is not None:
                loss_str += f" (best {best_val:.4f})"

        lines = f"""\033[H\033[1m┌{bar}┐\033[0m
\033[1m│\033[0m  GPU      {self.gpu_name} ({self.vram_total:.1f} GB)  Torch {torch_ver}{" " * max(0, w - 45 - len(self.gpu_name) - len(torch_ver))}\033[1m│\033[0m
\033[1m├{bar}┤\033[0m
\033[1m│\033[0m  step     {step}/{self.total_steps}{" " * max(0, w - 22 - len(str(step)) - len(str(self.total_steps)))}tok/s  {tk}  \033[1m│\033[0m
\033[1m│\033[0m  {loss_str}{" " * max(0, w - len(loss_str) - 5)}\033[1m│\033[0m
\033[1m│\033[0m  lr       {lr:.2e}  grad {grad_norm:.2f}{" " * max(0, w - 33)}\033[1m│\033[0m
\033[1m├{bar}┤\033[0m
\033[1m│\033[0m  VRAM     {vram_used:.1f} / {self.vram_total:.1f} GB{" " * max(0, w - 32)}peak  {vram_peak:.1f} GB  \033[1m│\033[0m
\033[1m│\033[0m  experts  {active_experts} / {total_experts} active{" " * max(0, w - 33)}entropy  {router_entropy:.2f}  \033[1m│\033[0m
\033[1m├{bar}┤\033[0m
\033[1m│\033[0m  ETA      {eta_h}h {eta_m}m{" " * max(0, w - 26)}seq  {seq_len}{" " * max(0, w - 36)}s/step  {step_time:.2f}s  \033[1m│\033[0m
\033[1m└{bar}┘\033[0m"""
        print(lines, end="", flush=True)
