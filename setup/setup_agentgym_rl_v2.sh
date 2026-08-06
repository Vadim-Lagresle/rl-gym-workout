#!/bin/bash
# Construit l'environnement d'entraînement v2 : vLLM récent + TRL récent + flash-attention.
# (Migration 2026-07-22 — possible depuis le passage de la VM à CentOS Stream 10 / glibc 2.39.)
#
# ⚠ L'env vit sur /tmp (overlay, VOLATIL — purgé lors des restarts de pod, cf. exp19) :
#   le home (35 Go) est trop petit pour ses ~12 Go. Ce script permet de le reconstruire
#   en ~10 min. Le wheel flash-attn compilé (50 min de compilation évitée) est conservé
#   sur le disque persistant : saves/wheels/flash_attn-*.whl.
#
# Usage :
#   bash setup/setup_agentgym_rl_v2.sh            # construit /tmp/envs/agentgym-rl-v2
#
# (L'ancien env rollback ~/envs/agentgym-rl a été supprimé le 2026-07-29 — ménage
#  disque après validation de la v2 sur les runs longs exp20/22/22.1.)
set -euo pipefail

ENV_DIR="${1:-/tmp/envs/agentgym-rl-v2}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Python 3.11 requis (wheel flash-attn cp311). pyenv d'abord, sinon l'env textcraft.
BASE_PYTHON="/opt/pyenv/versions/3.11.7/bin/python"
[ -x "$BASE_PYTHON" ] || BASE_PYTHON="$HOME/envs/agentenv-textcraft/bin/python"
CACHE="/tmp/pip-cache"

echo "[setup-v2] Création du venv : $ENV_DIR"
"$BASE_PYTHON" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip wheel ninja -q --cache-dir "$CACHE"

echo "[setup-v2] Stack principale (vLLM récent + TRL récent)..."
"$PIP" install "vllm>=0.25" "trl>=1.9.0" --cache-dir "$CACHE"

echo "[setup-v2] flash-attention (wheel précompilé si présent, sinon compilation ~50 min)..."
FA_WHEEL=$(ls "$REPO_ROOT"/saves/wheels/flash_attn-*.whl 2>/dev/null | head -1 || true)
if [ -n "$FA_WHEEL" ]; then
    "$PIP" install "$FA_WHEEL" --cache-dir "$CACHE"
else
    "$PIP" install flash-attn --no-build-isolation --cache-dir "$CACHE"
fi

echo "[setup-v2] Compléments training (peft, datasets, wandb, bitsandbytes)..."
"$PIP" install peft datasets wandb bitsandbytes --cache-dir "$CACHE"

echo "[setup-v2] Client TextCraft (agentenv, sans ses deps lourdes deepspeed/trl-pin)..."
"$PIP" install -e "$REPO_ROOT/external/AgentGym/agentenv" --no-deps --cache-dir "$CACHE"
"$PIP" install jsonlines --cache-dir "$CACHE"   # seule dep du client réellement absente

echo "[setup-v2] Vérification des imports critiques..."
"$ENV_DIR/bin/python" - << 'EOF'
import vllm, trl, peft, flash_attn, datasets, wandb, bitsandbytes, transformers, torch
from agentenv.envs import TextCraftEnvClient
print(f"vllm={vllm.__version__} trl={trl.__version__} peft={peft.__version__} "
      f"flash_attn={flash_attn.__version__} transformers={transformers.__version__} "
      f"torch={torch.__version__} bnb={bitsandbytes.__version__}")
print("TextCraftEnvClient importé OK")
EOF
echo "[setup-v2] Terminé : $ENV_DIR"
echo "[setup-v2] Utilisation : $ENV_DIR/bin/python src/train/train_grpo.py ... (PATH: export PATH=$ENV_DIR/bin:\$PATH)"
