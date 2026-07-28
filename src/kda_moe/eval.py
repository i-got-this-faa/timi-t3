"""Evaluation utilities: loss computation and autoregressive generation."""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: torch.utils.data.IterableDataset | None,
    tokenizer,
    max_batches: int = 10,
    device: torch.device | None = None,
) -> float:
    """Compute average cross-entropy loss on validation set."""
    if dataset is None:
        return float("nan")

    device = device or next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch in dataset:
        input_ids = batch.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        targets = input_ids[:, 1:]
        inputs = input_ids[:, :-1]

        if inputs.shape[1] < 1:
            continue

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(inputs)
            loss = model.loss_fn(logits, targets, ignore_index=-100)

        total_loss += loss.item()
        batches += 1
        if batches >= max_batches:
            break

    model.train()
    return total_loss / max(batches, 1)


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device | None = None,
) -> str:
    """Autoregressive generation. Greedy if temperature=0."""
    device = device or next(model.parameters()).device
    model.eval()

    encoded = tokenizer.encode(prompt)
    if encoded is None:
        return ""
    input_ids = torch.tensor(
        encoded.ids if hasattr(encoded, "ids") else list(encoded),
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    generated: list[int] = []

    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)  # (1, T, V)
        next_logits = logits[0, -1, :]  # (V,)

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(0, topk_idx, topk_vals)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = next_logits.argmax(dim=-1).item()

        generated.append(next_token)
        # Append to input
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device, dtype=torch.long)],
            dim=1,
        )

        # Stop at EOS if we have one
        if hasattr(tokenizer, "token_to_id"):
            eos_id = tokenizer.token_to_id("<|eos|>")
            if eos_id is not None and next_token == eos_id:
                break

    new_tokens = tokenizer.decode(generated) if generated else ""
    model.train()
    return new_tokens


def run_smoke_prompts(
    model: nn.Module,
    tokenizer,
    prompt_file: str,
    device: torch.device | None = None,
) -> dict[str, str]:
    """Run a list of fixed prompts from a file. Return dict of prompt->completion."""
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)

    results: dict[str, str] = {}
    for prompt in prompts:
        completion = generate(model, tokenizer, prompt, device=device)
        results[prompt] = completion
        print(f"Prompt: {prompt[:60]}...")
        print(f"  -> {completion[:120]}")
    return results
