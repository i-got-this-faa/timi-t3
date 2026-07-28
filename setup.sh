#!/usr/bin/env bash
# setup.sh — provision a fresh Google Colab (free T4) runtime for kda-poc.
#
# Idempotent: safe to re-run after a 12h disconnect. Re-cloning, re-installing
# deps, and re-linking Drive checkpoints are all no-ops if already done.
#
# Usage (in a Colab cell):
#   !bash setup.sh            # full setup
#   !bash setup.sh --check    # verify only, change nothing
#
# Env overrides:
#   REPO_URL   default: https://github.com/i-got-this-faa/timi-t3
#   REPO_REF   default: trunk (branch/tag/commit to checkout)
#   WITH_FLA=1            also try to install flash-linear-attention (optional)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/i-got-this-faa/timi-t3}"
REPO_REF="${REPO_REF:-trunk}"
REPO_NAME="$(basename "${REPO_URL}" .git)"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# Resolve paths. If this script lives inside a git repo, that repo is the
# project (local dev / pre-placed copy). Otherwise on Colab we clone into
# /content/<repo>. Drive root is only meaningful on Colab.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "${SCRIPT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REPO_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
  CONTENT="$(dirname "${REPO_DIR}")"
  INPLACE=1
else
  CONTENT=/content
  REPO_DIR="${CONTENT}/${REPO_NAME}"
  INPLACE=0
fi
DRIVE_ROOT="${CONTENT}/drive/MyDrive/kda-poc"
CKPT_LINK="${REPO_DIR}/artifacts"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. sanity: are we on a GPU runtime? -------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
  ok "GPU: ${GPU:-unknown}"
else
  warn "nvidia-smi not found — CPU-only runtime. Training will be very slow."
fi

# --- 1. torch: use the PREINSTALLED build, never reinstall -------------------
log "Checking preinstalled PyTorch (do not pip install torch on Colab)…"
python3 - <<'PY' || die "torch not importable — this runtime is broken."
import torch, sys
print(f"[ ok ] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
if not torch.cuda.is_available():
    print("[warn ] torch sees no GPU — request a GPU runtime (Runtime > Change runtime type > T4).")
PY

# --- 2. mount Google Drive (checkpoint persistence across 12h disconnects) ---
if [ "${CHECK_ONLY}" -eq 0 ]; then
  log "Mounting Google Drive…"
  python3 - <<'PY' || warn "Drive mount failed (not on Colab?) — checkpoints will NOT persist."
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive', force_remount=False)
    print("[ ok ] Drive mounted at /content/drive")
except Exception as e:
    raise SystemExit(f"no colab drive: {e}")
PY
else
  [ -d /content/drive/MyDrive ] && ok "Drive already mounted" || warn "Drive not mounted"
fi

# --- 3. get the repo ----------------------------------------------------------
if [ "${CHECK_ONLY}" -eq 0 ]; then
  if [ "${INPLACE}" -eq 1 ]; then
    ok "Using repo in place at ${REPO_DIR} (not cloning)"
  elif [ -d "${REPO_DIR}/.git" ]; then
    log "Repo present — fetching ${REPO_REF}…"
    git -C "${REPO_DIR}" fetch --all --quiet || warn "git fetch failed (offline?)"
    git -C "${REPO_DIR}" checkout --quiet "${REPO_REF}" || die "checkout '${REPO_REF}' failed — ref does not exist"
  else
    log "Cloning ${REPO_URL} (${REPO_REF}) → ${REPO_DIR}…"
    git clone --quiet "${REPO_URL}" "${REPO_DIR}" || die "git clone failed (private repo? set up auth)"
    git -C "${REPO_DIR}" checkout --quiet "${REPO_REF}" || die "checkout '${REPO_REF}' failed — ref does not exist"
  fi
  ok "Repo at ${REPO_DIR}"
else
  [ -d "${REPO_DIR}/.git" ] && ok "Repo present at ${REPO_DIR}" || die "Repo missing at ${REPO_DIR}"
fi

# --- 4. python deps (torch excluded — already present) ------------------------
if [ "${CHECK_ONLY}" -eq 0 ]; then
  log "Installing Python dependencies (this may take a minute)…"
  python3 -m pip install --quiet --no-input \
    datasets tokenizers transformers safetensors tqdm numpy einops tomli-w tensorboard \
    || die "pip install failed"
  ok "Dependencies installed"

  if [ "${WITH_FLA:-0}" = "1" ]; then
    log "WITH_FLA=1 — attempting flash-linear-attention (optional backend)…"
    if python3 -m pip install --quiet --no-input \
        "git+https://github.com/fla-org/flash-linear-attention" --no-deps 2>/dev/null \
       && python3 -c "import fla" 2>/dev/null; then
      ok "flash-linear-attention installed (Triton fallback on T4/sm_75)"
    else
      warn "FLA install/import failed — continuing with chunked PyTorch KDA (supported fallback)"
    fi
  fi
else
  python3 - <<'PY' || die "Missing dependencies — run without --check to install."
import importlib.util
mods = ["datasets","tokenizers","transformers","safetensors","tqdm","numpy","einops","tomli_w","tensorboard"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("[ ok ] all deps importable" if not missing else f"missing: {missing}")
raise SystemExit(1 if missing else 0)
PY
fi

# --- 5. Drive-backed checkpoint dir, symlinked into the repo ------------------
if [ -d /content/drive/MyDrive ]; then
  if [ "${CHECK_ONLY}" -eq 0 ]; then
    mkdir -p "${DRIVE_ROOT}/checkpoints" "${DRIVE_ROOT}/data"
    # Repo artifacts/ -> Drive, so a disconnect never loses checkpoints.
    if [ -L "${CKPT_LINK}" ]; then
      ok "artifacts symlink already linked"
    else
      rm -rf "${CKPT_LINK}" 2>/dev/null || true
      ln -s "${DRIVE_ROOT}" "${CKPT_LINK}"
      ok "artifacts/ → ${DRIVE_ROOT} (checkpoints persist on Drive)"
    fi
  else
    [ -L "${CKPT_LINK}" ] && ok "artifacts symlink present" || warn "artifacts not linked to Drive"
  fi
else
  warn "No Drive — checkpoints under ${REPO_DIR}/artifacts will NOT survive a disconnect"
fi

# --- 6. summary ---------------------------------------------------------------
echo
if [ "${CHECK_ONLY}" -eq 1 ]; then log "Setup check complete."; else log "Setup complete."; fi
if [ -d /content/drive/MyDrive ]; then PERSIST="(on Drive — survives disconnect)"; else PERSIST="(LOCAL — NOT persistent)"; fi
cat <<EOF
  repo         ${REPO_DIR}
  checkpoints  ${DRIVE_ROOT}/checkpoints  ${PERSIST}
  data         ${DRIVE_ROOT}/data         ${PERSIST}

  Next:
    cd ${REPO_DIR}
    python -c "import torch; print(torch.cuda.is_available())"
    # train:  python -m kda_moe.train --config configs/kda_moe_1b.toml
EOF
if [ "${CHECK_ONLY}" -eq 0 ]; then
  ok "Ready. Re-run any time after a disconnect — it is idempotent."
fi
exit 0
