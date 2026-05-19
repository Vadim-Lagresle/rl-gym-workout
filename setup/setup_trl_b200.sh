#!/usr/bin/env bash
# Setup complet de l'env TRL 1.3.0 sur B200 (ou A100 avec cu124).
#
# Usage :
#   bash setup/setup_trl_b200.sh          # B200 / cu126 (défaut)
#   CUDA=cu124 bash setup/setup_trl_b200.sh  # A100 / cu124
#
# Durée estimée : 20-30 min (téléchargement vLLM ~1 Go)

set -euo pipefail

CUDA="${CUDA:-cu126}"
ENV_NAME="trl-b200"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Création de l'env conda '$ENV_NAME' (Python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10

echo "==> Installation PyTorch (CUDA=$CUDA)"
conda run -n "$ENV_NAME" pip install "torch>=2.5.0" \
    --index-url "https://download.pytorch.org/whl/$CUDA"

echo "==> Installation des dépendances ML"
conda run -n "$ENV_NAME" pip install -r "$REPO_ROOT/setup/requirements-trl-b200.txt"

echo "==> Installation de agentenv (client HTTP TextCraft)"
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT/external/AgentGym/agentenv"

echo ""
echo "==> Setup terminé. Pour activer :"
echo "    conda activate $ENV_NAME"
echo ""
echo "==> Test rapide (serveur TextCraft doit tourner sur 127.0.0.1:36005) :"
echo "    python src/train_grpo.py --use-vllm --max-items 4 --max-steps 2 --num-generations 2"
