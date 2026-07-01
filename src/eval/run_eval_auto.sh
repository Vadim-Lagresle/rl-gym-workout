#!/bin/bash
# Lance une éval TextCraft en choisissant AUTOMATIQUEMENT la stack :
#   - vLLM (rapide, KV cache)        si l'archi du modèle est supportée par la vLLM installée
#   - sinon fallback HF generate()   (eval_fullft.py)
#
# Usage : bash src/eval/run_eval_auto.sh <model_path_or_hub_id> <run_name> [args supplémentaires...]
# Ex.   : bash src/eval/run_eval_auto.sh Qwen/Qwen3.5-4B exp15_qwen35_4b --no-thinking --max-items 100
#
# Les [args supplémentaires] sont transmis tels quels au script choisi. Attention :
# --no-thinking n'existe que côté eval_fullft.py (fallback). Pour un modèle vLLM-compatible,
# ne pas passer de flag spécifique au fallback.
#
# Contrainte disque de cette VM : /home/criteo (35 Go) est saturé → exporter
#   HF_HOME=/tmp/hf_cache   (overlay, 221 Go)   avant de lancer, pour les modèles du Hub.
set -uo pipefail
MODEL="${1:?model path ou hub id requis}"
RUN="${2:?run name requis}"
shift 2
EXTRA=("$@")
PY="$HOME/envs/agentgym-rl/bin/python"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[run_eval] Test de compatibilité vLLM pour : $MODEL"
if "$PY" src/utils/vllm_supports.py "$MODEL"; then
    echo "[run_eval] → vLLM COMPATIBLE : stack vLLM (rapide)"
    bash src/utils/start_vllm_server.sh "$MODEL"
    "$PY" src/eval/eval_vllm.py --model "$MODEL" --run-name "$RUN" "${EXTRA[@]}"
    kill "$(cat /tmp/vllm_server.pid)" 2>/dev/null || true
else
    echo "[run_eval] → vLLM INCOMPATIBLE : fallback HuggingFace generate (eval_fullft.py)"
    "$PY" src/eval/eval_fullft.py --checkpoint "$MODEL" --run-name "$RUN" "${EXTRA[@]}"
fi
