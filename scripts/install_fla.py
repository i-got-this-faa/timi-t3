"""Utility: attempt to install flash-linear-attention (FLA)."""
import subprocess
import sys


def main():
    print("Attempting to install flash-linear-attention...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "git+https://github.com/fla-org/flash-linear-attention",
             "--no-deps"],
        )
        import fla  # noqa: F401
        print("FLA installed and importable.")
    except Exception:
        print("FLA skipped, chunked PyTorch fallback active.")


if __name__ == "__main__":
    main()
