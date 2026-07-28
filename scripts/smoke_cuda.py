"""P0 gate: CUDA is available and can do a matmul."""
import torch


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    a = torch.randn(128, 128, device=dev, dtype=torch.bfloat16)
    b = torch.randn(128, 128, device=dev, dtype=torch.bfloat16)
    c = a @ b
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    print(f"bf16 matmul 128x128: {c.mean().item():.4f}")
    print("P0 PASS")


if __name__ == "__main__":
    main()
