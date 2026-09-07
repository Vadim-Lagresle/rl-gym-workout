#!/bin/bash
# pass@20 du Qwen2.5-7B-Instruct NU sur le test set (borne AVANT fine-tuning, exp26).
# Résumable : relancer la même commande reprend aux passes manquantes (passes.jsonl).
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"

bash setup/ensure_qwen_tmp.sh /tmp/models/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct \
  || { echo "[job31] téléchargement 7B ÉCHOUÉ"; exit 1; }

WAIT_ITERS=240 PYTHON=/tmp/envs/agentgym-rl-v2/bin/python \
  bash src/utils/start_vllm_server.sh /tmp/models/Qwen2.5-7B-Instruct 16384 \
  || { echo "[job31] serveur vLLM 7B pas prêt (voir /tmp/vllm_server.log)"; exit 1; }

python src/eval/eval_oracle.py --model /tmp/models/Qwen2.5-7B-Instruct \
    --run-name 7_oracle/oracle_qwen25_7b_base --n-samples 20 \
    > logs/oracle_qwen25_7b_base.log 2>&1
rc=$?
kill "$(cat /tmp/vllm_server.pid)" 2>/dev/null
sleep 10   # laisser la VRAM se libérer avant le job suivant
exit $rc
