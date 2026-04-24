#!/usr/bin/env bash
# SCommander one-button setup
#
# Usage:
#   bash scripts/setup.sh                      # full install (conda env + CUDA toolkit + deps)
#   bash scripts/setup.sh --skip-cuda          # skip conda CUDA toolkit (use system nvcc / module load)
#   bash scripts/setup.sh --no-ssm             # skip mamba-ssm/causal-conv1d/cupy (CPU-friendly dev)
#   bash scripts/setup.sh --env-name=foo       # custom env name (default: scommander)
#   bash scripts/setup.sh --python=3.10        # custom Python version (default: 3.11)
#
# Works identically on local dev machine AND remote GPU server, assuming:
#   - conda (anaconda or miniconda) available
#   - git clone of this repo
#   - NVIDIA driver present (for GPU paths)
#
# Idempotent: safe to re-run.

set -euo pipefail

ENV_NAME="scommander"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.8"  # Blackwell sm_120 (RTX 5090) requires 12.8+; 12.4 works via PTX JIT with perf penalty
SKIP_CUDA=false
NO_SSM=false

for arg in "$@"; do
  case "$arg" in
    --skip-cuda) SKIP_CUDA=true ;;
    --no-ssm) NO_SSM=true ;;
    --env-name=*) ENV_NAME="${arg#*=}" ;;
    --python=*) PYTHON_VERSION="${arg#*=}" ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[FAIL] Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf "\033[1;34m[%-5s]\033[0m %s\n" "STEP" "$*"; }
ok()   { printf "\033[1;32m[%-5s]\033[0m %s\n" "OK"   "$*"; }
warn() { printf "\033[1;33m[%-5s]\033[0m %s\n" "WARN" "$*"; }
die()  { printf "\033[1;31m[%-5s]\033[0m %s\n" "FAIL" "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# 1. Conda bootstrap check
# -----------------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
  die "conda not found. Install miniconda first:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3
    source \$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
ok "conda available: $(conda --version)"

# -----------------------------------------------------------------------------
# 2. Create or reuse conda env
# -----------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  ok "conda env '$ENV_NAME' exists — reusing"
else
  log "creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
ok "activated env: $(python -c 'import sys;print(sys.executable)')"

# -----------------------------------------------------------------------------
# 3. CUDA toolkit (nvcc) — needed for mamba-ssm / cupy compile
# -----------------------------------------------------------------------------
if [ "$SKIP_CUDA" = true ]; then
  warn "skipping CUDA toolkit install (--skip-cuda). Relying on system nvcc."
elif command -v nvcc &>/dev/null; then
  nvcc_ver="$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)"
  ok "nvcc already present: $nvcc_ver"
else
  log "installing CUDA toolkit $CUDA_VERSION into env via nvidia channel"
  conda install -c nvidia "cuda-toolkit=$CUDA_VERSION" -y \
    || warn "conda CUDA toolkit install failed; mamba-ssm/cupy compile may fail. Continuing."
fi

# -----------------------------------------------------------------------------
# 4. uv + pip upgrade
# -----------------------------------------------------------------------------
log "installing uv (fast Python package resolver)"
python -m pip install -q -U pip uv
ok "uv: $(uv --version)"

# -----------------------------------------------------------------------------
# 5. Install PyTorch with cu128 wheels (Blackwell sm_120)
# -----------------------------------------------------------------------------
# Must install torch explicitly from cu128 index BEFORE pyproject deps; PyPI default
# wheels do not carry sm_120 kernels. Skip with SKIP_TORCH_INDEX=true if already installed.
if [ "${SKIP_TORCH_INDEX:-false}" != true ]; then
  log "installing torch+torchaudio+torchvision from cu128 wheel index (sm_120 support)"
  uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7.0" "torchvision>=0.22.0" "torchaudio>=2.7.0" \
    || warn "cu128 torch install failed; pyproject resolution may fall back to default index (risk: no sm_120 kernels)"
fi

# -----------------------------------------------------------------------------
# 6. Project deps
# -----------------------------------------------------------------------------
if [ "$NO_SSM" = true ]; then
  log "installing SCommander (dev + tracking, no SSM)"
  uv pip install -e ".[dev,tracking]"
  warn "mamba-ssm + cupy NOT installed (--no-ssm). Track C forward pass will need pure-PyTorch SSM fallback."
else
  log "installing SCommander (ssm + dev + tracking)"
  # Blackwell sm_120: mamba-ssm needs source build; set TORCH_CUDA_ARCH_LIST explicitly
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  export MAMBA_FORCE_BUILD="${MAMBA_FORCE_BUILD:-TRUE}"
  if ! uv pip install -e ".[ssm,dev,tracking]" --no-build-isolation; then
    warn "full install failed (likely mamba-ssm sm_120 source build). Retrying without ssm extra."
    uv pip install -e ".[dev,tracking]"
    warn "SSM extras skipped. Track C falls back to pure-PyTorch SSM. Re-attempt on server: MAMBA_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0 uv pip install -e '.[ssm]' --no-build-isolation"
  fi
fi

# cupy separate (not in pyproject because version depends on CUDA major)
if [ "$NO_SSM" = false ] && [ "$SKIP_CUDA" = false ]; then
  if ! python -c "import cupy" 2>/dev/null; then
    log "installing cupy-cuda12x (<14 — tonic pins numpy<2; cupy 14.x requires numpy>=2)"
    uv pip install "cupy-cuda12x>=13.3,<14" || warn "cupy install failed (will fall back to PyTorch backend, ~3x slower)"
  else
    ok "cupy already importable"
  fi
fi

# -----------------------------------------------------------------------------
# 6. Lockfile (reproducibility) + env export
# -----------------------------------------------------------------------------
log "freezing requirements-lock.txt"
uv pip freeze > "$REPO_ROOT/requirements-lock.txt" || warn "lockfile freeze failed"
ok "lockfile: $REPO_ROOT/requirements-lock.txt"

log "exporting environment.yml (conda from-history)"
conda env export --from-history --name "$ENV_NAME" > "$REPO_ROOT/environment.yml" 2>/dev/null \
  || warn "environment.yml export failed"

# -----------------------------------------------------------------------------
# 7. Verify
# -----------------------------------------------------------------------------
log "running verify_env.py"
python scripts/verify_env.py || warn "verify_env reported issues — check output above"

echo
ok "Setup complete."
echo
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  pytest -q"
echo
echo "Server deploy: git clone <repo> && cd SCommander && bash scripts/setup.sh"
