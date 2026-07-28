"""Dataset loading, shard download, pretraining mix builder."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset

from .config import ModelConfig

# Dataset IDs and their configs (None = no config needed)
DATASET_SPECS: dict[str, dict] = {
    "dclm": {"id": "mlfoundations/dclm-baseline-1.0", "config": None},
    "fineweb": {"id": "HuggingFaceFW/fineweb-edu", "config": None},
    "code": {"id": "code-search-net/code_search_net", "config": "python"},
    "math_owm": {"id": "aklein4/proof-pile-2-fixed", "config": "open-web-math"},
    "math_alg": {"id": "aklein4/proof-pile-2-fixed", "config": "algebraic-stack"},
}

TEXT_FIELDS: dict[str, str] = {
    "mlfoundations/dclm-baseline-1.0": "text",
    "HuggingFaceFW/fineweb-edu": "text",
    "code-search-net/code_search_net": "whole_func_string",
    "aklein4/proof-pile-2-fixed": "text",
}


def stream_examples(
    dataset_id: str,
    split: str = "train",
    max_examples: int | None = None,
    config: str | None = None,
) -> Iterator[dict]:
    """Stream examples from a HuggingFace dataset. Returns raw dict rows."""
    kwargs = {"split": split, "streaming": True, "trust_remote_code": True}
    if config:
        ds = load_dataset(dataset_id, config, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)

    text_field = TEXT_FIELDS.get(dataset_id, "text")
    count = 0
    for row in ds:
        text = row.get(text_field, "") or ""
        if not text or not isinstance(text, str) or len(text.strip()) < 20:
            continue
        yield {"text": text, "source": f"{dataset_id}/{config}" if config else dataset_id}
        count += 1
        if max_examples is not None and count >= max_examples:
            break


def download_shard(
    dataset_id: str,
    output_dir: str,
    max_gb: float,
    split: str = "train",
    config: str | None = None,
) -> Path:
    """Download a fixed-size subset from a HF dataset, save as jsonl shards."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label = f"{dataset_id}/{config}" if config else dataset_id
    shard_path = out / f"{label.replace('/', '_')}.jsonl"
    if shard_path.exists():
        print(f"  Shard already exists: {shard_path}")
        return out

    bytes_written = 0
    max_bytes = int(max_gb * 1e9)
    written = 0

    with open(shard_path, "w", encoding="utf-8") as f:
        for row in stream_examples(dataset_id, split=split, config=config):
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            bytes_written += len(line.encode("utf-8"))
            written += 1
            if bytes_written >= max_bytes:
                break

    size_mb = bytes_written / 1e6
    print(f"  Downloaded {written} examples ({size_mb:.1f} MB) -> {shard_path}")
    return out


def build_pretraining_mix(
    data_dir: str,
    model_config: ModelConfig,
) -> tuple[Path, Path]:
    """Download all pretraining sources per plan.md §5.1 caps.

    Carves every 10th document into val_dir.
    Returns (train_dir, val_dir).
    """
    base = Path(data_dir)
    train_dir = base / "shards"
    val_dir = base / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in DATASET_SPECS.items():
        dataset_id = spec["id"]
        ds_config = spec["config"]
        cap_gb = model_config.data_caps_gb.get(name, model_config.data_caps_gb.get(dataset_id, 1.0))
        # Split math cap between owm and algstack
        if name.startswith("math_"):
            cap_gb = model_config.data_caps_gb.get("math", 1.0) / 2
        split_cap = cap_gb * 0.9
        label = f"{name} ({dataset_id}/{ds_config})" if ds_config else f"{name} ({dataset_id})"
        print(f"Downloading {label} - cap {cap_gb:.1f} GB")
        download_shard(dataset_id, str(train_dir), split_cap, split="train", config=ds_config)
    # Carve val: every 10th document from train shards
    for shard_file in sorted(train_dir.glob("*.jsonl")):
        val_file = val_dir / shard_file.name
        train_lines: list[str] = []
        val_lines: list[str] = []
        with open(shard_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % 10 == 0:
                    val_lines.append(line)
                else:
                    train_lines.append(line)
        with open(shard_file, "w", encoding="utf-8") as f:
            f.writelines(train_lines)
        with open(val_file, "w", encoding="utf-8") as f:
            f.writelines(val_lines)
        print(
            f"  Carved {len(val_lines)} val lines from {shard_file.name} ({len(train_lines)} train)"
        )

    return train_dir, val_dir


class PretrainingDataset(torch.utils.data.IterableDataset):
    """Iterable dataset streaming from jsonl shards, yielding tokenized sequences."""

    def __init__(
        self,
        shard_dir: str,
        tokenizer,
        seq_len: int = 2048,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.shard_files = sorted(self.shard_dir.glob("*.jsonl"))
        if not self.shard_files:
            raise FileNotFoundError(f"No .jsonl files in {shard_dir}")

    def __iter__(self):
        files = list(self.shard_files)
        if self.shuffle_shards:
            random.shuffle(files)
        buffer: list[int] = []
        for shard_file in files:
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text", "")
                    if not text:
                        continue
                    encoded = self.tokenizer.encode(text)
                    if encoded is None:
                        continue
                    ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len:
                        chunk = buffer[: self.seq_len]
                        buffer = buffer[self.seq_len :]
                        yield torch.tensor(chunk, dtype=torch.long)
        if len(buffer) >= self.seq_len // 2:
            padded = buffer + [0] * (self.seq_len - len(buffer))
            yield torch.tensor(padded, dtype=torch.long)


def load_fable_traces(max_rows: int | None = None) -> list[dict]:
    """Load FABLE.5 traces from Crownelius/Complete-FABLE.5-traces-2M."""
    ds = load_dataset(
        "Crownelius/Complete-FABLE.5-traces-2M",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    rows: list[dict] = []
    for row in ds:
        rows.append(dict(row))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows
