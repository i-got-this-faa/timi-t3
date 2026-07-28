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
    "code_csn": {"id": "code-search-net/code_search_net", "config": "python"},
    "math_owm": {"id": "aklein4/proof-pile-2-fixed", "config": "open-web-math"},
    "math_alg": {"id": "aklein4/proof-pile-2-fixed", "config": "algebraic-stack"},
    "tinystories": {"id": "roneneldan/TinyStories", "config": None},
    "markdown": {"id": "bigcode/the-stack-dedup", "config": "markdown"},
    "code_stack": {"id": "bigcode/the-stack-dedup", "config": "python"},
    "code_flytech": {"id": "flytech/python-codes-25k", "config": None},
}

TEXT_FIELDS: dict[str, str] = {
    "mlfoundations/dclm-baseline-1.0": "text",
    "HuggingFaceFW/fineweb-edu": "text",
    "code-search-net/code_search_net": "whole_func_string",
    "aklein4/proof-pile-2-fixed": "text",
    "roneneldan/TinyStories": "text",
    "bigcode/the-stack-dedup": "content",
    "flytech/python-codes-25k": "text",
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


def download_shard(
    dataset_id: str,
    output_dir: str,
    max_gb: float,
    split: str = "train",
    config: str | None = None,
) -> tuple[Path, int, int]:
    """Download a fixed-size subset from a HF dataset, save as jsonl shards.

    Returns (output_dir_path, char_count, doc_count).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label = f"{dataset_id}/{config}" if config else dataset_id
    shard_path = out / f"{label.replace('/', '_')}.jsonl"
    if shard_path.exists():
        char_count = sum(
            len(line.encode("utf-8"))
            for line in shard_path.read_text(encoding="utf-8").splitlines(True)
        )
        doc_count = sum(1 for _ in shard_path.read_text(encoding="utf-8").splitlines())
        print(f"  Shard already exists: {shard_path}")
        return out, char_count, doc_count

    bytes_written = 0
    max_bytes = int(max_gb * 1e9)
    written = 0
    char_count = 0

    with open(shard_path, "w", encoding="utf-8") as f:
        for row in stream_examples(dataset_id, split=split, config=config):
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            char_count += len(line)
            bytes_written += len(line.encode("utf-8"))
            written += 1
            if bytes_written >= max_bytes:
                break

    size_mb = bytes_written / 1e6
    print(f"  Downloaded {written} examples ({size_mb:.1f} MB) -> {shard_path}")
    return out, char_count, written


def _write_manifest(shard_dir: Path, stats: dict[str, dict]) -> None:
    """Write manifest.json tracking char_count and doc_count per dataset."""
    manifest = {}
    for name, info in sorted(stats.items()):
        manifest[name] = {
            "shards": info["shards"],
            "char_count": info["char_count"],
            "doc_count": info["doc_count"],
        }
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote manifest: {manifest_path}")


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

    # Count code sub-sources for cap splitting
    code_sources = [k for k in DATASET_SPECS if k.startswith("code_")]
    n_code = len(code_sources)

    per_dataset_stats: dict[str, dict] = {}

    for name, spec in DATASET_SPECS.items():
        dataset_id = spec["id"]
        ds_config = spec["config"]
        cap_gb = model_config.data_caps_gb.get(name, model_config.data_caps_gb.get(dataset_id, 1.0))
        # Split math cap between owm and algstack
        if name.startswith("math_"):
            cap_gb = model_config.data_caps_gb.get("math", 1.0) / 2
        # Split code cap across sub-sources
        if name.startswith("code_"):
            cap_gb = model_config.data_caps_gb.get("code", 1.0) / n_code
        split_cap = cap_gb * 0.9
        label = f"{name} ({dataset_id}/{ds_config})" if ds_config else f"{name} ({dataset_id})"
        print(f"Downloading {label} - cap {cap_gb:.1f} GB")
        _, char_count, doc_count = download_shard(
            dataset_id, str(train_dir), split_cap, split="train", config=ds_config
        )
        # Map to canonical data_mix key for manifest (strip code_/math_ prefixes)
        if name.startswith("code_"):
            canon = "code"
        elif name.startswith("math_"):
            canon = "math"
        else:
            canon = name
        if canon not in per_dataset_stats:
            per_dataset_stats[canon] = {"shards": [], "char_count": 0, "doc_count": 0}
        # Determine shard filename
        label_key = f"{dataset_id}/{ds_config}" if ds_config else dataset_id
        shard_filename = f"{label_key.replace('/', '_')}.jsonl"
        per_dataset_stats[canon]["shards"].append(shard_filename)
        per_dataset_stats[canon]["char_count"] += char_count
        per_dataset_stats[canon]["doc_count"] += doc_count

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

    _write_manifest(train_dir, per_dataset_stats)
    return train_dir, val_dir


class PretrainingDataset(torch.utils.data.IterableDataset):
    """Iterable dataset streaming from jsonl shards, yielding tokenized sequences.

    Supports weighted sampling via data_mix: when provided, shards are sampled
    with replacement according to the declared proportions, using manifest.json
    for shard-to-dataset mapping. Falls back to uniform streaming otherwise.
    """

    def __init__(
        self,
        shard_dir: str,
        tokenizer,
        seq_len: int = 2048,
        data_mix: dict[str, float] | None = None,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_mix = data_mix
        self.shuffle_shards = shuffle_shards
        self.shard_files = sorted(self.shard_dir.glob("*.jsonl"))
        if not self.shard_files:
            raise FileNotFoundError(f"No .jsonl files in {shard_dir}")

        # Load manifest for weighted sampling
        self._manifest: dict | None = None
        self._shard_weights: list[tuple[Path, float]] | None = None
        if self.data_mix:
            manifest_path = self.shard_dir / "manifest.json"
            if manifest_path.exists():
                self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self._shard_weights = self._build_shard_weights()
                if self._shard_weights:
                    print(
                        f"  Weighted sampling: {len(self._shard_weights)} shards "
                        f"across {len(self._manifest)} datasets"
                    )
                else:
                    print("  Weighted sampling: no matching shards, falling back to uniform")
                    self._shard_weights = None
            else:
                print("  Weighted sampling: manifest.json missing, falling back to uniform")
                self._shard_weights = None

    def _build_shard_weights(self) -> list[tuple[Path, float]] | None:
        """Build (shard_path, weight) pairs from manifest and data_mix."""
        assert self._manifest is not None
        assert self.data_mix is not None

        # Check for datasets in mix that have no shards — redistribute weight
        working_mix = dict(self.data_mix)
        missing = [d for d in working_mix if d not in self._manifest]
        for d in missing:
            print(f"  Warning: dataset '{d}' in data_mix has no shards — redistributing")
            del working_mix[d]
        if not working_mix:
            return None

        # Redistribute weight from missing datasets proportionally
        if missing:
            total_remain = sum(working_mix.values())
            if total_remain > 0:
                scale = 1.0 / total_remain
                working_mix = {k: v * scale for k, v in working_mix.items()}

        pairs: list[tuple[Path, float]] = []
        shard_map: dict[str, list[str]] = {}  # shard filename -> canonical dataset name
        for dname, info in self._manifest.items():
            for shard_name in info.get("shards", []):
                shard_map[shard_name] = dname

        for sf in self.shard_files:
            dname = shard_map.get(sf.name)
            if dname is None:
                continue
            weight = working_mix.get(dname, 0.0)
            if weight <= 0:
                continue
            # Divide dataset weight equally among its shards
            n_shards = len(self._manifest[dname].get("shards", [1]))
            pairs.append((sf, weight / n_shards))

        if not pairs:
            return None

        # Normalize weights to sum to 1
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return None
        return [(p, w / total_w) for p, w in pairs]

    def __iter__(self):
        # Weighted sampling mode
        if self._shard_weights:
            return self._iter_weighted()
        # Uniform mode (original behavior)
        return self._iter_uniform()

    def _iter_weighted(self):
        """Sample shards with replacement according to data_mix weights."""
        assert self._shard_weights is not None
        shard_files, weights = zip(*self._shard_weights)
        buffer: list[int] = []
        while True:
            chosen = random.choices(shard_files, weights=weights, k=1)[0]
            yield from self._stream_shard(chosen, buffer)

    def _iter_uniform(self):
        """Original behavior: stream all shards, optionally shuffled."""
        files = list(self.shard_files)
        if self.shuffle_shards:
            random.shuffle(files)
        buffer: list[int] = []
        for shard_file in files:
            yield from self._stream_shard(shard_file, buffer)
        if len(buffer) >= self.seq_len // 2:
            padded = buffer + [0] * (self.seq_len - len(buffer))
            yield torch.tensor(padded, dtype=torch.long)

    def _stream_shard(self, shard_file: Path, buffer: list[int]):
        """Stream tokenized sequences from a single shard into the buffer."""
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
                    buffer[:] = buffer[self.seq_len :]
                    yield torch.tensor(chunk, dtype=torch.long)


def print_dataset_report(shard_dir: str, tokenizer) -> None:
    """Print a formatted dataset report from manifest.json with real token counts.

    Samples up to 1000 documents per shard, tokenizes them, and scales
    to estimate total token counts. Updates manifest.json with real counts.
    """

    sd = Path(shard_dir)
    manifest_path = sd / "manifest.json"
    if not manifest_path.exists():
        print("Dataset report: manifest.json not found — run prepare_data.py first.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    SAMPLE_LIMIT = 1000
    label_map = {
        "dclm": "DCLM",
        "fineweb": "FineWeb-Edu",
        "code": "Code (CSN + The Stack + flytech)",
        "math": "Proof-Pile-2",
        "tinystories": "TinyStories",
        "markdown": "Markdown (The Stack)",
    }

    total_docs = 0
    total_tokens = 0
    total_chars_est = 0

    print()
    print("\u2500" * 40)
    print("Dataset Report")
    print("\u2500" * 40)

    updated_manifest: dict = {}

    for dname in sorted(manifest.keys()):
        info = manifest[dname]
        shard_names = info.get("shards", [])
        total_chars = info.get("char_count", 0)
        total_doc_count = info.get("doc_count", 0)

        sampled_chars = 0
        sampled_tokens = 0
        sampled_docs = 0

        for shard_name in shard_names:
            shard_path = sd / shard_name
            if not shard_path.exists():
                continue
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    if sampled_docs >= SAMPLE_LIMIT:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text", "")
                    if not text:
                        continue
                    encoded = tokenizer.encode(text)
                    if encoded is None:
                        continue
                    ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)
                    sampled_chars += len(text)
                    sampled_tokens += len(ids)
                    sampled_docs += 1
                if sampled_docs >= SAMPLE_LIMIT:
                    break

        # Scale: tokens_per_char * total_chars
        if sampled_chars > 0 and total_chars > 0:
            est_tokens = int((sampled_tokens / sampled_chars) * total_chars)
        else:
            est_tokens = total_chars // 4  # rough fallback

        est_gb = total_chars / 1e9

        label = label_map.get(dname, dname)
        print(f"{label}")
        print(
            f"  docs        {total_doc_count / 1e6:.1f}M"
            if total_doc_count >= 1e6
            else f"  docs        {total_doc_count / 1e3:.0f}k"
        )
        tokens_str = f"{est_tokens / 1e9:.2f}B" if est_tokens >= 1e9 else f"{est_tokens / 1e6:.0f}M"
        print(f"  tokens      {tokens_str}")
        print(f"  est bytes   {est_gb:.1f} GB")
        print()

        total_docs += total_doc_count
        total_tokens += est_tokens
        total_chars_est += total_chars

        # Store real token count in updated manifest
        updated_manifest[dname] = dict(info)
        updated_manifest[dname]["token_count"] = est_tokens

    avg_len = total_tokens / total_docs if total_docs > 0 else 0

    print("\u2500" * 40)
    print("TOTAL")
    docs_str = f"{total_docs / 1e6:.1f}M" if total_docs >= 1e6 else f"{total_docs / 1e3:.0f}k"
    tokens_total_str = (
        f"{total_tokens / 1e9:.2f}B" if total_tokens >= 1e9 else f"{total_tokens / 1e6:.0f}M"
    )
    print(f"  docs        {docs_str}")
    print(f"  tokens      {tokens_total_str}")
    print(f"  avg len     {avg_len:.0f}")
    print(f"  est bytes   {total_chars_est / 1e9:.1f} GB")
    print("\u2500" * 40)

    # Write updated manifest with real token counts
    manifest_path.write_text(
        json.dumps(updated_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("  Updated manifest.json with real token counts.")
    print()


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
