"""Rich-powered live status display with proper TUI layout."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

# Fixed bar-track length for the expert-load grid (keeps column widths constant).
EXPERT_BAR_MAX = 8


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

        self._header = f"[bold]{self.gpu_name}[/]   {torch_ver}"

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

        # Time-series history for trend graphs
        self._hist: dict[str, deque[float]] = {
            "loss": deque(maxlen=120),
            "tok/s": deque(maxlen=120),
            "entropy": deque(maxlen=120),
            "grad": deque(maxlen=120),
            "val_loss": deque(maxlen=120),
        }
        # (panel label, history key, last-value format)
        self._spark_metrics = [
            ("loss", "loss", "{:.3f}"),
            ("tok/s", "tok/s", "{:.0f}"),
            ("entropy", "entropy", "{:.2f}"),
            ("grad", "grad", "{:.2f}"),
            ("val", "val_loss", "{:.3f}"),
        ]

        # Root layout
        self._layout = Layout()
        self._layout.split(
            Layout(name="header", size=3),
            Layout(name="metrics", size=6),
            Layout(name="progress", size=3),
            Layout(name="trends", size=12),
            Layout(name="experts", ratio=1),
            Layout(name="resources", size=5),
        )

        self._live = Live(self._layout, refresh_per_second=4, transient=True)
        self._live.start()

        # Sparkline width is fixed for the whole run so nothing shifts as data grows.
        live_width = self._live.console.size.width or 0
        try:
            term_width = live_width or os.get_terminal_size().columns
        except OSError:
            term_width = 100
        self._spark_width = max(30, min(term_width - 24, 200))

    def update(
        self,
        step: int = 0,
        loss: float = 0,
        lr: float = 0,
        grad_norm: float = 0,
        tok_per_sec: float = 0,
        step_time: float = 0,
        seq_len: int = 0,
        tokens: int = 0,
        vram_used: float = 0,
        vram_peak: float = 0,
        active_experts: int = 0,
        total_experts: int = 0,
        router_entropy: float = 0,
        expert_load: list[float] | None = None,
        expert_active: list[bool] | None = None,
        epoch: float = 0,
        val_loss: float | None = None,
        best_val: float | None = None,
        **_: Any,
    ):
        if loss:
            self._hist["loss"].append(loss)
        if tok_per_sec:
            self._hist["tok/s"].append(tok_per_sec)
        if router_entropy:
            self._hist["entropy"].append(router_entropy)
        if grad_norm:
            self._hist["grad"].append(grad_norm)
        if val_loss is not None:
            self._hist["val_loss"].append(val_loss)

        self._render_header()
        self._render_metrics(
            step,
            lr,
            step_time,
            seq_len,
            tokens,
            epoch,
        )
        self._render_progress(step)
        self._render_trends()
        self._render_experts(expert_load, expert_active)
        self._render_resources(vram_used, vram_peak, active_experts, total_experts)
        self._live.update(self._layout)

    def _render_header(self):
        self._layout["header"].update(
            Panel(Align.center(self._header, vertical="middle"), style="bold blue")
        )

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(int(n))

    def _render_metrics(
        self,
        step: int,
        lr: float,
        step_time: float,
        seq_len: int,
        tokens: int,
        epoch: float,
    ):
        eta_s = (self.total_steps - step) * max(step_time, 0.01)
        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)

        table = Table.grid(padding=(0, 4))
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="left", ratio=1)

        table.add_row(
            f"[bold]Step[/]   {step}",
            f"[bold]seq[/]    {seq_len}",
        )
        table.add_row(
            f"[bold]Tokens[/] {self._fmt_tokens(tokens)}",
            f"[bold]ETA[/]   {eta_h}h {eta_m}m  [bold]s/step[/] {step_time:.2f}s",
        )
        table.add_row(
            f"[bold]Epochs[/] {epoch:.2f}",
            f"[bold]LR[/]    {lr:.2e}",
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

    @staticmethod
    def _sparkline(values: list[float], width: int) -> str:
        """Render a unicode block sparkline (▁▂▃▄▅▆▇█) for a numeric series."""
        if not values:
            return ""
        if len(values) > width:
            values = values[-width:]
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1e-9
        levels = "▁▂▃▄▅▆▇█"
        return "".join(levels[int((v - lo) / (hi - lo) * 7 + 0.5)] for v in values)

    def _render_trends(self):
        lines = []
        for label, key, fmt in self._spark_metrics:
            series = self._hist[key]
            if series:
                sp = self._sparkline(list(series), self._spark_width).ljust(self._spark_width)
                lines.append(
                    f"[bold]{label:>6}[/] {sp}  {fmt.format(series[-1]):>8}"
                )
            else:
                lines.append(f"[dim]{label:>6} {'·' * self._spark_width}  {'—':>8}[/]")
        self._layout["trends"].update(Panel("\n\n".join(lines), title="Trends", border_style="cyan"))

    def _render_experts(
        self,
        expert_load: list[float] | None,
        expert_active: list[bool] | None,
    ):
        n = len(expert_load) if expert_load else 0
        if n == 0:
            self._layout["experts"].update(
                Panel("[dim]waiting for router data…[/]", title="Expert load")
            )
            return

        cols = min(6, max(1, -(-n // 8)))
        rows = -(-n // cols)
        table = Table.grid(padding=(0, 3))
        for _ in range(cols):
            table.add_column(justify="left")

        for r in range(rows):
            row = []
            for c in range(cols):
                i = c * rows + r
                if i >= n:
                    row.append("")
                    continue
                pct = expert_load[i] * 100
                n_filled = min(EXPERT_BAR_MAX, int(round(pct)))
                track = "█" * n_filled + "░" * (EXPERT_BAR_MAX - n_filled)
                cell = f"E{i:02d} {track} {pct:4.1f}%"
                cell = f"[bold]{cell}[/]" if expert_active[i] else f"[dim]{cell}[/]"
                row.append(cell)
            table.add_row(*row)

        self._layout["experts"].update(Panel(table, title="Expert load", border_style="green"))

    def _render_resources(
        self,
        vram_used: float,
        vram_peak: float,
        active_experts: int,
        total_experts: int,
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
            "",
        )

        self._layout["resources"].update(Panel(table, title="Resources", border_style="magenta"))

    def stop(self):
        """Stop the live display (call when training is done)."""
        self._live.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._live.stop()
