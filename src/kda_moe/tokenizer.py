"""BPE tokenizer training, loading, and fertility measurement."""
from __future__ import annotations

import json
import random
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|tool_result|>", "<|reasoning|>", "<|final|>", "<|end|>",
    "<|tools|>", "<|function_call|>", "<|tool_json|>", "<|tool_name|>", "<|tool_arguments|>",
    "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
]


def train_tokenizer(
    data_dir: str,
    vocab_size: int = 32000,
    output_path: str = "artifacts/tokenizer",
    sample_size: int = 10_000_000,
) -> HFTokenizer:
    """Train a BPE tokenizer on a bounded sample of pretraining shards."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = Path(data_dir)
    shard_files = sorted(shard_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl files in {data_dir}")

    random.shuffle(shard_files)
    texts: list[str] = []
    bytes_collected = 0

    for shard_file in shard_files:
        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if text and len(text) > 100:
                    texts.append(text)
                    bytes_collected += len(text.encode("utf-8"))
                    if bytes_collected >= sample_size:
                        break
        if bytes_collected >= sample_size:
            break

    print(f"Training tokenizer on {len(texts)} texts ({bytes_collected / 1e6:.1f} MB)")

    tokenizer = HFTokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    save_path = out_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}")

    return tokenizer


def load_tokenizer(path: str) -> HFTokenizer:
    """Load a trained tokenizer from disk."""
    return HFTokenizer.from_file(path)


def measure_fertility(
    tokenizer: HFTokenizer, samples: dict[str, list[str]],
) -> dict[str, float]:
    """Measure tokens per word on text samples.

    Returns dict with fertility scores per category.
    """
    results: dict[str, float] = {}
    for category, texts in samples.items():
        fertilities = []
        for text in texts:
            words = len(text.split())
            if words == 0:
                continue
            encoded = tokenizer.encode(text)
            if encoded is None:
                continue
            tokens = len(encoded.ids)
            fertilities.append(tokens / words)
        if fertilities:
            results[category] = sum(fertilities) / len(fertilities)
    return results


def fertility_report(tokenizer: HFTokenizer) -> bool:
    """Run fertility gate on built-in samples. Returns True if gate passes."""
    web_samples = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is a subfield of artificial intelligence that focuses on "
        "building systems that learn from data. " * 5,
        "The capital of France is Paris, a city known for its culture, cuisine, and "
        "historical landmarks. " * 5,
    ]

    code_samples = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n"
        "        super().__init__()\n        self.linear = nn.Linear(512, 256)\n"
        "    def forward(self, x):\n        return self.linear(x)",
        "for i in range(100):\n    if i % 15 == 0:\n        print('FizzBuzz')\n"
        "    elif i % 3 == 0:\n        print('Fizz')\n    elif i % 5 == 0:\n"
        "        print('Buzz')\n    else:\n        print(i)",
    ]

    json_samples = [
        '{"type":"function_call","name":"read","arguments":{"path":"src/main.py"}}',
        '{"type":"function_call","name":"write","arguments":{"path":"hello.py",'
        '"content":"print(\\"hello\\")"}}',
        '{"messages":[{"role":"user","content":"What is 2+2?"},'
        '{"role":"assistant","content":"4"}]}',
    ]

    samples: dict[str, list[str]] = {
        "web": web_samples,
        "code": code_samples,
        "json": json_samples,
    }

    results = measure_fertility(tokenizer, samples)

    gates = {"web": 2.0, "code": 2.0, "json": 5.0}
    all_pass = True

    for category, fertility in results.items():
        threshold = gates.get(category, 2.0)
        status = "PASS" if fertility < threshold else "FAIL"
        if fertility >= threshold:
            all_pass = False
        print(f"  {category}: {fertility:.2f} tokens/word (threshold < {threshold}) -> {status}")

    return all_pass
