"""P7: FABLE.5 trace normalization pipeline.

Normalizes FABLE.5 traces from diverse sources into OpenAI Responses-style JSON,
filters for quality, and tokenizes for SFT with loss masking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from datasets import load_dataset
from torch import Tensor


@dataclass
class NormalizedTrace:
    """Internal representation after normalization."""
    messages: list[dict]  # [{"role": "user"|"assistant", "content": str}, ...]
    tool_calls: list[dict]  # [{"name": str, "arguments": dict, "result": str | None}, ...]
    reasoning_spans: list[str]


@dataclass
class ResponsesTrace:
    """OpenAI Responses-style JSON."""
    system: str
    messages: list[dict]  # {"role": "user"|"assistant", "content": str}
    tool_calls: list[dict]  # {"type": "function_call", "name": str, "arguments": dict}
    reasoning: str | None


# ── source-specific normalizers ────────────────────────────


def _normalize_generic(row_json: dict) -> NormalizedTrace | None:
    """Generic normalizer: extract messages, tool_calls, reasoning from common JSON shapes."""
    try:
        if isinstance(row_json, str):
            row_json = json.loads(row_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(row_json, dict):
        return None

    messages: list[dict] = []
    tool_calls: list[dict] = []
    reasoning_spans: list[str] = []

    # Try common message formats
    if "messages" in row_json and isinstance(row_json["messages"], list):
        for msg in row_json["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": str(content)})
                # Check for tool calls in message
                if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            tool_calls.append({
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", {}),
                                "result": tc.get("result"),
                            })

    # Try conversations format
    if not messages and "conversations" in row_json:
        for turn in row_json["conversations"]:
            if isinstance(turn, dict):
                messages.append({
                    "role": turn.get("from", "user"),
                    "content": str(turn.get("value", "")),
                })

    # Extract reasoning
    for key in ("reasoning", "chain_of_thought", "cot", "thinking"):
        if key in row_json and row_json[key]:
            reasoning_spans.append(str(row_json[key]))

    if not messages:
        return None

    return NormalizedTrace(
        messages=messages,
        tool_calls=tool_calls,
        reasoning_spans=reasoning_spans,
    )


SOURCE_SCHEMAS: dict[str, Callable] = {
    # All sources use the generic normalizer; extend for source-specific formats
}


def normalize_row(source: str, row_json: dict) -> NormalizedTrace | None:
    """Dispatch to the correct normalizer for this source."""
    normalizer = SOURCE_SCHEMAS.get(source, _normalize_generic)
    return normalizer(row_json)


def to_responses_format(trace: NormalizedTrace) -> ResponsesTrace:
    """Convert normalized trace to OpenAI Responses-style JSON."""
    # Extract system message if first message has role "system"
    system = ""
    msgs = list(trace.messages)
    if msgs and msgs[0].get("role") == "system":
        system = msgs[0].get("content", "")
        msgs = msgs[1:]

    # Convert tool calls
    responses_calls = []
    for tc in trace.tool_calls:
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        responses_calls.append({
            "type": "function_call",
            "name": tc.get("name", ""),
            "arguments": args,
        })

    reasoning = " ".join(trace.reasoning_spans) if trace.reasoning_spans else None

    return ResponsesTrace(
        system=system,
        messages=msgs,
        tool_calls=responses_calls,
        reasoning=reasoning,
    )


def clean_reasoning(text: str) -> str:
    """Strip envelope noise / UI metadata from reasoning text."""
    if not text:
        return text
    # Remove common noise patterns
    for prefix in ("Let me think", "I need to", "First,", "Okay,"):
        idx = text.lower().find(prefix.lower())
        if idx >= 0 and idx < 50:
            text = text[idx:]
            break
    return text.strip()


def filter_trace(trace: NormalizedTrace) -> bool:
    """Return True if trace has real user intent + parseable tool spans + substantive response."""
    if not trace.messages:
        return False
    # Must have at least one user and one assistant message
    has_user = any(m.get("role") == "user" for m in trace.messages)
    has_assistant = any(m.get("role") == "assistant" for m in trace.messages)
    if not (has_user and has_assistant):
        return False
    # All messages must have non-empty content
    if any(not m.get("content", "").strip() for m in trace.messages):
        return False
    # Response must be substantive (>20 chars)
    for m in reversed(trace.messages):
        if m.get("role") == "assistant":
            if len(m.get("content", "")) < 20:
                return False
            break
    return True


def fable_pipeline(max_rows: int | None = None) -> list[ResponsesTrace]:
    """Full pipeline: load -> normalize -> filter -> convert -> clean."""
    ds = load_dataset(
        "Crownelius/Complete-FABLE.5-traces-2M",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    results: list[ResponsesTrace] = []
    normalized = 0
    filtered = 0

    for row in ds:
        row_dict = dict(row)
        source = row_dict.get("source", "unknown")
        row_json = row_dict.get("row_json")

        if row_json is None:
            continue

        if isinstance(row_json, str):
            try:
                row_json = json.loads(row_json)
            except json.JSONDecodeError:
                continue

        trace = normalize_row(source, row_json)
        normalized += 1

        if trace is None or not filter_trace(trace):
            filtered += 1
            continue

        rt = to_responses_format(trace)
        if rt.reasoning:
            rt.reasoning = clean_reasoning(rt.reasoning)
        results.append(rt)

        if max_rows is not None and len(results) >= max_rows:
            break

    print(f"FABLE pipeline: {normalized} normalized, {filtered} filtered, "
          f"{len(results)} accepted")
    return results


def format_for_sft(trace: ResponsesTrace) -> str:
    """Render a ResponsesTrace to a tokenizable string with chat template markers."""
    parts: list[str] = []

    if trace.system:
        parts.append(f"<|system|>{trace.system}")

    for msg in trace.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|user|>{content}")
        elif role == "assistant":
            tool_part = ""
            if trace.tool_calls:
                tc = trace.tool_calls[0]  # first tool call
                tool_part = (
                    f"<|tool_call|>"
                    f"<|function_call|>{json.dumps(tc, ensure_ascii=False)}<|end|>"
                )
            if trace.reasoning:
                parts.append(f"<|reasoning|>{trace.reasoning}<|end|>")
            parts.append(f"<|assistant|>{content}{tool_part}<|end|>")

    return "".join(parts)


def tokenize_sft_dataset(
    traces: list[ResponsesTrace],
    tokenizer,
    max_seq_len: int = 2048,
    w_tool: float = 2.0,
) -> dict[str, Tensor]:
    """Tokenize traces with loss masking.

    - Assistant tokens: loss weight 1.0
    - Tool-call JSON spans: loss weight w_tool
    - Tool-result tokens: loss weight 0.0 (masked out)
    - System/user tokens: loss weight 0.0

    Returns {"input_ids": (N, L), "labels": (N, L), "loss_mask": (N, L)}
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_masks: list[list[float]] = []

    # Find special token IDs
    assistant_id = tokenizer.token_to_id("<|assistant|>") or -1
    tool_call_id = tokenizer.token_to_id("<|tool_call|>") or -1
    tool_result_id = tokenizer.token_to_id("<|tool_result|>") or -1
    user_id = tokenizer.token_to_id("<|user|>") or -1
    system_id = tokenizer.token_to_id("<|system|>") or -1
    end_id = tokenizer.token_to_id("<|end|>") or -1
    pad_id = tokenizer.token_to_id("<|pad|>") or 0

    for trace in traces:
        text = format_for_sft(trace)
        encoded = tokenizer.encode(text)
        if encoded is None:
            continue
        ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)

        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # Create labels and mask
        labels = list(ids)
        mask = [0.0] * len(ids)

        in_assistant = False
        in_tool_call = False
        in_tool_result = False

        for i, tok_id in enumerate(ids):
            if tok_id == assistant_id:
                in_assistant = True
                in_tool_call = False
                in_tool_result = False
            elif tok_id == tool_call_id:
                in_tool_call = True
                in_assistant = False
            elif tok_id == tool_result_id:
                in_tool_result = True
                in_tool_call = False
            elif tok_id in (user_id, system_id):
                in_assistant = False
                in_tool_call = False
                in_tool_result = False
            elif tok_id == end_id:
                in_assistant = False
                in_tool_call = False
                in_tool_result = False

            if in_tool_result:
                mask[i] = 0.0
            elif in_tool_call:
                mask[i] = w_tool
            elif in_assistant:
                mask[i] = 1.0
            else:
                mask[i] = 0.0
                labels[i] = -100  # ignore in loss

        # Pad
        pad_len = max_seq_len - len(ids)
        ids.extend([pad_id] * pad_len)
        labels.extend([-100] * pad_len)
        mask.extend([0.0] * pad_len)

        all_input_ids.append(ids)
        all_labels.append(labels)
        all_masks.append(mask)

    if not all_input_ids:
        return {
            "input_ids": torch.empty(0, max_seq_len, dtype=torch.long),
            "labels": torch.empty(0, max_seq_len, dtype=torch.long),
            "loss_mask": torch.empty(0, max_seq_len, dtype=torch.float),
        }

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "loss_mask": torch.tensor(all_masks, dtype=torch.float),
    }
