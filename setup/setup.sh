#!/usr/bin/env bash
# ============================================================================
# Reproducible setup for rl-gym-workout (AgentGym + AgentGym-RL).
#
# This script recreates the two conda envs we built on the GCP VM:
#   - agentgym-rl       (Python 3.10) : training stack (verl + vllm + torch + ...)
#   - agentenv-textcraft (Python 3.10): the TextCraft HTTP env server
#
# Assumptions:
#   - You are at the root of the rl-gym-workout repo.
#   - Submodules are already initialized (git submodule update --init --recursive).
#   - You have a GPU + recent NVIDIA driver (CUDA 12.x compatible).
#   - Conda is installed and reachable (`conda --version` works).
#
# Usage:
#   bash setup/setup.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 0. Submodules ---------------------------------------------------------------
git submodule update --init --recursive

# 1. Accept Anaconda ToS (required by recent conda versions) -----------------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# 2. agentgym-rl env (training side) -----------------------------------------
conda create -n agentgym-rl python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentgym-rl

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

WHEEL=flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/$WHEEL"
pip install "./$WHEEL"
rm -f "$WHEEL"

pip install -e ./AgentGym-RL
pip install -e ./AgentGym/agentenv
pip install transformers==4.51.3

python - <<'PY'
import torch, vllm, transformers, flash_attn, ray
from verl import DataProto
from agentenv.envs import TextCraftEnvClient
print("torch         :", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("vllm          :", vllm.__version__)
print("transformers  :", transformers.__version__)
print("flash_attn    :", flash_attn.__version__)
print("ray           :", ray.__version__)
print("verl + agentenv: OK")
PY

conda deactivate

# 3. agentenv-textcraft env (env server side) --------------------------------
# The README says python=3.9 but the agentenv package requires >=3.10.
conda create -n agentenv-textcraft python=3.10 -y
conda activate agentenv-textcraft

pip install -e ./AgentGym/agentenv
pip install -e ./AgentGym/agentenv-textcraft

which textcraft
textcraft --help
conda deactivate

echo
echo "=================================================="
echo "Setup done. To launch the TextCraft server:"
echo "  conda activate agentenv-textcraft"
echo "  textcraft --host 127.0.0.1 --port 36005"
echo "=================================================="
