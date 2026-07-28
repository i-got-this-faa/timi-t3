"""NixOS compatibility: monkey-patch Triton's ldconfig probe."""
import os
import subprocess

_ORIG_CHECK_OUTPUT = subprocess.check_output

_PATCHED = False


def apply_triton_patch():
    """On NixOS, /sbin/ldconfig can't read the ld.so.cache.
    Returns True if patch was applied."""
    global _PATCHED
    if _PATCHED:
        return False
    _PATCHED = True

    if os.path.exists("/etc/ld.so.cache"):
        return False  # not NixOS

    cuda_dir = None
    for search in ["/run/opengl-driver/lib", "/usr/lib", "/usr/local/cuda/lib64"]:
        if os.path.exists(os.path.join(search, "libcuda.so.1")):
            cuda_dir = search
            break
    if cuda_dir is None:
        return False

    fake_output = f"libcuda.so.1 (libc6,x86-64) => {cuda_dir}/libcuda.so.1\n"
    fake_bytes = fake_output.encode()

    def _patched(*args, **kwargs):
        if isinstance(args[0], list) and "/sbin/ldconfig" in str(args[0]):
            return fake_bytes
        return _ORIG_CHECK_OUTPUT(*args, **kwargs)

    subprocess.check_output = _patched
    return True
