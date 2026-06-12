#!/usr/bin/env bash
# setup_env.sh — Full environment setup for EDGE on Linux x86_64 + CUDA 11.8
#
# Usage:
#   bash setup_env.sh
#
# What this script does (and WHY each step is needed):
#   1. Creates the conda env from environment-linux.yml
#      — The yml's pip section is SKIPPED because conda's bundled pip cannot
#        install jukemirlib from PyPI (it isn't published there).
#   2. Downgrades setuptools inside the new env
#      — The default setuptools >=70 removes pkg_resources as a public API,
#        but jukebox (a dependency of jukemirlib) uses it in its setup.py.
#        Pinning to <70 restores pkg_resources.
#   3. Installs all pip packages manually, with jukemirlib from GitHub
#      and --no-build-isolation so the already-installed setuptools is used.
#   4. Installs pytorch3d 0.7.5 via --find-links
#      — The prebuilt wheel is on fbaipublicfiles but is NOT a PEP 503 index,
#        so --extra-index-url silently finds nothing; --find-links works.
#   5. Writes a non-interactive accelerate config (single GPU, fp16).
#
# Tested: 2026-04-24, Python 3.10, CUDA 11.8, PyTorch 2.1.2, H200 GPU.

set -euo pipefail

# ── Note for shared-cluster / HPC users ──────────────────────────────────────
# If your $HOME is on a non-persistent or quota-limited filesystem, remap HOME,
# TMPDIR and PATH to a persistent location BEFORE running this script, so the
# conda env, pip cache and accelerate config don't land on ephemeral disk and
# vanish after the session. (On the original dev cluster this was done with a
# `source init_env.sh` step.) On a normal workstation, no action is needed.

CONDA_BASE="$(conda info --base)"
ENV_NAME="edge"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

# ── Step 1: Create conda env (conda packages only; pip section will fail) ──────
echo "==> [1/5] Creating conda env '${ENV_NAME}' from environment-linux.yml ..."
# We catch the expected pip failure and continue — conda packages are already
# installed by the time pip is attempted.
conda env create -f environment-linux.yml || true

# Verify the env actually exists (conda part succeeded)
if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "ERROR: conda env '${ENV_NAME}' was not created. Check the yml file."
    exit 1
fi
echo "    conda packages installed."

# ── Step 2: Fix setuptools (restore pkg_resources) ────────────────────────────
echo "==> [2/5] Downgrading setuptools to <70 to restore pkg_resources ..."
"${PIP}" install "setuptools<70" -q
"${PYTHON}" -c "import pkg_resources" 2>/dev/null \
    && echo "    pkg_resources OK" \
    || { echo "ERROR: pkg_resources still missing after setuptools downgrade."; exit 1; }

# ── Step 3: Install pip packages (jukemirlib from GitHub) ─────────────────────
echo "==> [3/5] Installing pip packages (this may take a few minutes) ..."
"${PIP}" install \
    einops \
    librosa \
    soundfile \
    p_tqdm \
    wandb \
    "accelerate>=0.24,<1.0" \
    "git+https://github.com/rodrigo-castellon/jukemirlib.git" \
    --no-build-isolation \
    -q
echo "    pip packages installed."

# ── Step 4: Install pytorch3d ──────────────────────────────────────────────────
echo "==> [4/5] Installing pytorch3d 0.7.5 (py310 + cu118 + pyt210 wheel) ..."
# --find-links is required; --extra-index-url does NOT work for this URL.
"${PIP}" install pytorch3d \
    --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html \
    -q
echo "    pytorch3d installed."

# ── Step 5: Write accelerate config (single GPU, fp16) ────────────────────────
echo "==> [5/5] Writing accelerate config (single GPU, fp16) ..."
ACCEL_CFG="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname "${ACCEL_CFG}")"
cat > "${ACCEL_CFG}" << 'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
echo "    accelerate config written to ${ACCEL_CFG}"

# ── Verification ───────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation ..."
"${PYTHON}" - << 'PYEOF'
import pytorch3d, torch, jukemirlib, accelerate, librosa, wandb
print(f"  pytorch3d  {pytorch3d.__version__}")
print(f"  torch      {torch.__version__}  (cuda: {torch.cuda.is_available()})")
print(f"  jukemirlib OK")
print(f"  accelerate {accelerate.__version__}")
print(f"  librosa    {librosa.__version__}")
print(f"  wandb      {wandb.__version__}")
PYEOF

echo ""
echo "==> Done. Activate with:  conda activate ${ENV_NAME}"
