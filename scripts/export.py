"""Export model checkpoint to safetensors or GGUF format.

Usage:
    python scripts/export.py --ckpt artifacts/checkpoints_1b/step_200.pt --format safetensors
    python scripts/export.py --ckpt artifacts/checkpoints_1b/step_200.pt --format gguf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from kda_moe.compat import apply_triton_patch

apply_triton_patch()

from kda_moe.config import ModelConfig
from kda_moe.model import KDAMoEModel


def export_safetensors(ckpt_path: str, output_path: str, config: ModelConfig):
    """Export model weights as safetensors (handles tied embeddings)."""
    from safetensors.torch import save_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KDAMoEModel(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt, strict=False)

    # Clone tied weights — safetensors rejects shared memory
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.clone().contiguous().cpu()

    save_file(state, output_path)
    print(f"Exported safetensors: {output_path} ({Path(output_path).stat().st_size / 1e6:.1f} MB)")


def export_gguf(ckpt_path: str, output_path: str, config: ModelConfig):
    """Export model as GGUF (llama.cpp format).

    Note: KDA + MoE is non-standard. This produces a basic GGUF
    with a custom architecture tag. Full llama.cpp inference support
    requires a matching C++ implementation of the KDA-MoE forward pass.
    """
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        print("gguf package not installed. Install with: pip install gguf")
        print("Falling back to safetensors...")
        export_safetensors(ckpt_path, output_path.replace(".gguf", ".safetensors"), config)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KDAMoEModel(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt, strict=False)

    writer = GGUFWriter(output_path, "kda-moe")

    # Metadata
    writer.add_architecture("kda-moe")
    writer.add_context_length(config.seq_len)
    writer.add_embedding_length(config.d_model)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_kv_heads)
    writer.add_file_type(1)  # bf16

    # Write tensors
    state = model.state_dict()
    name_map = {
        "token_embedding.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    for name, tensor in state.items():
        gguf_name = name_map.get(name, name.replace(".", "_"))
        writer.add_tensor(gguf_name, tensor.cpu().to(torch.float32).numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Exported GGUF: {output_path} ({Path(output_path).stat().st_size / 1e6:.1f} MB)")
    print("Warning: GGUF inference requires a custom llama.cpp backend for KDA-MoE.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--format", default="safetensors", choices=["safetensors", "gguf"])
    parser.add_argument("--output", default=None, help="Output path (auto-generated if omitted)")
    parser.add_argument("--config", default="configs/kda_moe_1b.toml")
    args = parser.parse_args()

    config = ModelConfig.from_toml(args.config)

    ckpt_stem = Path(args.ckpt).stem
    ext = ".safetensors" if args.format == "safetensors" else ".gguf"
    output = args.output or f"artifacts/exports/{ckpt_stem}{ext}"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if args.format == "safetensors":
        export_safetensors(args.ckpt, output, config)
    else:
        export_gguf(args.ckpt, output, config)


if __name__ == "__main__":
    main()
