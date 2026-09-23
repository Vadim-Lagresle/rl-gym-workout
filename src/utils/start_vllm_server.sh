#!/bin/bash
# In plain words: starts an OpenAI-compatible vLLM server on port 8001 for a given
# checkpoint, used by the evaluation's vllm backend.
#
# Lance un serveur vLLM compatible OpenAI sur le port 8001.
# Usage : bash src/utils/start_vllm_server.sh <chemin_checkpoint> [max_model_len]
# Exemple : bash src/utils/start_vllm_server.sh models/Qwen2.5-3B-Instruct
#           bash src/utils/start_vllm_server.sh models/Qwen2.5-3B-Instruct 32768  # prompts longs (few-shot)
#
# Le serveur tourne en arrière-plan. Pour l'arrêter : kill $(cat /tmp/vllm_server.pid)
# Vérification : curl http://localhost:8001/health

MODEL_PATH="${1:-models/Qwen2.5-3B-Instruct}"
MAX_LEN="${2:-16384}"
PORT=8001
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Env par défaut : v1 historique. Pour servir avec l'env v2 (vLLM récent, requis
# pour Qwen3.5) : PYTHON=/tmp/envs/agentgym-rl-v2/bin/python bash src/utils/start_vllm_server.sh ...
PYTHON="${PYTHON:-$HOME/envs/agentgym-rl/bin/python}"
# Les sous-process vLLM V1 (EngineCore) doivent trouver ninja & co dans le PATH.
export PATH="$(dirname "$PYTHON"):$PATH"

# Chemin absolu si relatif
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$REPO_ROOT/$MODEL_PATH"
fi

echo "[vllm] Démarrage serveur sur port $PORT avec modèle : $MODEL_PATH"
echo "[vllm] Logs : /tmp/vllm_server.log"

nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.45 \
    --max-model-len "$MAX_LEN" \
    --enable-prefix-caching \
    --trust-remote-code \
    > /tmp/vllm_server.log 2>&1 &

echo $! > /tmp/vllm_server.pid
echo "[vllm] PID : $! (sauvegardé dans /tmp/vllm_server.pid)"
echo "[vllm] Attente démarrage..."

WAIT_ITERS="${WAIT_ITERS:-90}"  # ×2 s ; monter (ex. WAIT_ITERS=240) pour les gros modèles (7B)
for i in $(seq 1 "$WAIT_ITERS"); do    # défaut 3 min — le démarrage vLLM v2 (compilation JIT + CUDA graphs) dépasse souvent 60 s
    sleep 2
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "[vllm] Serveur prêt sur http://localhost:$PORT"
        exit 0
    fi
done

echo "[vllm] TIMEOUT — vérifier /tmp/vllm_server.log"
exit 1
