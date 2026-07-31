"""Rich-powered live status display with proper TUI layout."""

from __future__ import annotations

import time
from typing import Any

from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table


class StatusDisplay:
    """Renders a live-updating dashboard via rich. No ANSI hackery."""

    def __init__(self, total_steps: int, gpu_name: str = "", vram_total: float = 0):
        self.total_steps = total_steps
        self.gpu_name = gpu_name
        self.vram_total = vram_total
        self.start_time = time.time()

        torch_ver = ""
        try:
            import torch

            tv = torch.__version__.split("+")[0]
            cv = torch.version.cuda or "?"
            torch_ver = f"Torch {tv}  CUDA {cv}"
        except Exception:
            pass

        self._header = f"[bold]{self.gpu_name}[/] ({self.vram_total:.1f} GB)   {torch_ver}"

        self._progress = Progress(
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("• {task.fields[detail]}"),
            expand=True,
        )
        self._progress_task = self._progress.add_task(
            "[bold]Training", total=total_steps, detail=""
        )

        # Root layout
        self._layout = Layout()
        self._layout.split(
            Layout(name="header", size=3),
            Layout(name="metrics", size=7),
            Layout(name="progress", size=3),
            Layout(name="resources", size=5),
        )

        self._live = Live(self._layout, refresh_per_second=4, transient=True)
        self._live.start()

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
        self._render_header()
        self._render_metrics(
            step,
            loss,
            lr,
            grad_norm,
            tok_per_sec,
            step_time,
            seq_len,
            val_loss,
            best_val,
        )
        self._render_progress(step)
        self._render_resources(
            vram_used,
            vram_peak,
            active_experts,
            total_experts,
            router_entropy,
        )
        self._live.update(self._layout)

    def _render_header(self):
        self._layout["header"].update(
            Panel(Align.center(self._header, vertical="middle"), style="bold blue")
        )

    def _render_metrics(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tok_per_sec: float,
        step_time: float,
        seq_len: int,
        val_loss: float | None,
        best_val: float | None,
    ):
        tk = f"{tok_per_sec / 1000:.1f}k" if tok_per_sec > 1000 else f"{tok_per_sec:.0f}"

        eta_s = (self.total_steps - step) * max(step_time, 0.01)
        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)

        loss_text = f"{loss:.4f}"
        if val_loss is not None:
            loss_text += f"  [dim]val {val_loss:.4f}[/]"
            if best_val is not None:
                loss_text += f" [dim](best {best_val:.4f})[/]"

        table = Table.grid(padding=(0, 4))
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="left", ratio=1)

        table.add_row(
            f"[bold]Step[/]  {step}/{self.total_steps}",
            f"[bold]tok/s[/]  {tk}",
        )
        table.add_row(
            f"[bold]Loss[/]  {loss_text}",
            f"[bold]seq[/]    {seq_len}",
        )
        table.add_row(
            f"[bold]LR[/]    {lr:.2e}  [bold]grad[/] {grad_norm:.2f}",
            f"[bold]ETA[/]   {eta_h}h {eta_m}m  [bold]s/step[/] {step_time:.2f}s",
        )

        self._layout["metrics"].update(Panel(table, title="Training", border_style="cyan"))

    def _render_progress(self, step: int):
        pct = step / max(self.total_steps, 1) * 100
        self._progress.update(
            self._progress_task,
            completed=step,
            detail=f"{pct:.1f}%  step {step}/{self.total_steps}",
        )
        self._layout["progress"].update(Panel(self._progress, border_style="green"))

    def _render_resources(
        self,
        vram_used: float,
        vram_peak: float,
        active_experts: int,
        total_experts: int,
        router_entropy: float,
    ):
        table = Table.grid(padding=(0, 4))
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="left", ratio=1)

        expert_color = "green" if active_experts == total_experts else "yellow"
        table.add_row(
            f"[bold]VRAM[/]     {vram_used:.1f} / {self.vram_total:.1f} GB",
            f"[bold]peak[/]    {vram_peak:.1f} GB",
        )
        table.add_row(
            f"[bold]Experts[/]  [{expert_color}]{active_experts} / {total_experts}[/] active",
            f"[bold]entropy[/] {router_entropy:.2f}",
        )

        self._layout["resources"].update(Panel(table, title="Resources", border_style="magenta"))

    def stop(self):
        """Stop the live display (call when training is done)."""
        self._live.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._live.stop()
