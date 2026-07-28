"""P7: coding-agent evaluation on 30 manually-curated prompts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import ModelConfig
from kda_moe.eval import generate
from kda_moe.model import KDAMoEModel
from kda_moe.tokenizer import load_tokenizer
from kda_moe.train import load_checkpoint


def is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON with a 'name' field (tool call)."""
    try:
        # Extract JSON from between markers or find first { }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start:end + 1])
            return isinstance(obj, dict) and "name" in obj
    except (json.JSONDecodeError, ValueError):
        pass
    return False


def check_tool_call(completion: str, expected_tool: str, expected_args: dict) -> bool:
    """Check if completion contains a valid tool call matching expectations."""
    if not is_valid_json(completion):
        return False
    try:
        start = completion.find("{")
        end = completion.rfind("}")
        obj = json.loads(completion[start:end + 1])
        if obj.get("name") != expected_tool:
            return False
        # Loose arg check: key names must match
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return False
        return set(expected_args.keys()).issubset(set(args.keys()))
    except (json.JSONDecodeError, ValueError, KeyError):
        return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prompts
    prompts_file = Path(__file__).resolve().parent.parent / "configs" / "coding_eval_prompts.jsonl"
    if not prompts_file.exists():
        print(f"Prompts file not found: {prompts_file}")
        print("Creating with default prompts...")
        create_default_prompts(prompts_file)

    prompts = []
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} coding eval prompts")

    # Load model and tokenizer
    config = ModelConfig.preset_1b()
    model = KDAMoEModel(config).to(device)

    # Check for SFT checkpoint, fall back to base
    ckpt_path = Path("artifacts/checkpoints_sft/step_1000.pt")
    if ckpt_path.exists():
        load_checkpoint(model, None, str(ckpt_path), device=device)
        print(f"Loaded SFT model from {ckpt_path}")
    else:
        ckpt_path = Path("artifacts/checkpoints_1b/step_10000.pt")
        if ckpt_path.exists():
            load_checkpoint(model, None, str(ckpt_path), device=device)
            print(f"Loaded base model from {ckpt_path}")
        else:
            print("Warning: no checkpoint found, using random init")

    try:
        tokenizer = load_tokenizer("artifacts/tokenizer/tokenizer.json")
    except FileNotFoundError:
        print("No tokenizer found — skipping eval")
        return

    # Evaluate
    passed = 0
    total = len(prompts)

    for i, entry in enumerate(prompts):
        prompt = entry["prompt"]
        expected_tool = entry.get("expected_tool", "")
        expected_args = entry.get("expected_args", {})

        completion = generate(
            model, tokenizer, prompt, max_new_tokens=128, temperature=0.0,
            device=device,
        )

        ok = check_tool_call(completion, expected_tool, expected_args)
        if ok:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{i+1:2d}/{total}] {status} | {expected_tool}"
              f"{' -> ' + completion[:80].replace(chr(10), ' ') if not ok else ''}")

    pass_rate = passed / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Coding eval: {passed}/{total} passed ({pass_rate:.1f}%)")
    print(f"{'='*50}")


def create_default_prompts(path: Path):
    """Create default coding eval prompts file."""
    defaults = [
        {"prompt": "<|system|>You are a coding agent.<|user|>Read the file src/config.py<|assistant|>",
         "expected_tool": "read", "expected_args": {"path": "src/config.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Write a hello world to hello.py<|assistant|>",
         "expected_tool": "write", "expected_args": {"path": "hello.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Edit line 5 of main.py to add an import<|assistant|>",
         "expected_tool": "edit", "expected_args": {"path": "main.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Search for 'TODO' in the codebase<|assistant|>",
         "expected_tool": "grep", "expected_args": {"pattern": "TODO"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>List all Python files in src/<|assistant|>",
         "expected_tool": "glob", "expected_args": {"pattern": "src/**/*.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Run pytest on the test suite<|assistant|>",
         "expected_tool": "bash", "expected_args": {"command": "pytest"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>What is the definition of the Model class?<|assistant|>",
         "expected_tool": "read", "expected_args": {"path": "model.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Create a new file called utils.py with helper functions<|assistant|>",
         "expected_tool": "write", "expected_args": {"path": "utils.py"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Find all occurrences of 'deprecated' in comments<|assistant|>",
         "expected_tool": "grep", "expected_args": {"pattern": "deprecated"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Delete the temporary file /tmp/test.py<|assistant|>",
         "expected_tool": "bash", "expected_args": {"command": "rm"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Show me the first 20 lines of README.md<|assistant|>",
         "expected_tool": "read", "expected_args": {"path": "README.md"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Rename function 'old_name' to 'new_name' in all files<|assistant|>",
         "expected_tool": "edit", "expected_args": {"path": "src"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Count the number of lines in all .py files<|assistant|>",
         "expected_tool": "bash", "expected_args": {"command": "wc"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Check the current git diff<|assistant|>",
         "expected_tool": "bash", "expected_args": {"command": "git"}},
        {"prompt": "<|system|>You are a coding agent.<|user|>Find all JSON config files in the project<|assistant|>",
         "expected_tool": "glob", "expected_args": {"pattern": "*.json"}},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in defaults:
            f.write(json.dumps(entry) + "\n")
    print(f"Created {len(defaults)} default prompts at {path}")


if __name__ == "__main__":
    main()
