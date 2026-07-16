#!/bin/bash
# Benchmark baseline Pass@1 pour Llama-3.2-1B, SmolLM2-1.7B, Gemma-3-1B, DeepSeek-R1-Distill-1.5B
# Stratégie disque : télécharge → eval → supprime → modèle suivant.
# Lance avec : nohup bash src/train/run_benchmark_baselines.sh > logs/benchmark_baselines.log 2>&1 &
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/envs/agentgym-rl/bin/python"

echo "========================================"
echo "[benchmark] START $(date -u)"
echo "========================================"

run_model() {
    local repo_id="$1"
    local local_dir="$2"
    local run_name="$3"
    local system_prompt="$4"

    echo ""
    echo "[benchmark] ── $repo_id ──────────────────────────────────"
    echo "[benchmark] Disque avant téléchargement :"
    df -h /home/criteo | tail -1

    # Téléchargement
    echo "[benchmark] Téléchargement de $repo_id → $local_dir"
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', local_dir='$REPO_ROOT/$local_dir', ignore_patterns=['*.pt','*.gguf'])
print('Download OK')
"

    # Eval baseline (serveur vLLM)
    echo "[benchmark] Démarrage serveur vLLM ($(date -u))"
    bash "$REPO_ROOT/src/utils/start_vllm_server.sh" "$REPO_ROOT/$local_dir"
    echo "[benchmark] Eval baseline $run_name ($(date -u))"
    "$PYTHON" "$REPO_ROOT/src/eval/eval_vllm.py" \
        --model "$REPO_ROOT/$local_dir" \
        --run-name "$run_name" \
        --system-prompt "$system_prompt"
    kill "$(cat /tmp/vllm_server.pid)" 2>/dev/null || true
    sleep 3

    echo "[benchmark] Eval $run_name terminée ($(date -u))"

    # Suppression du modèle pour libérer l'espace
    echo "[benchmark] Suppression de $local_dir pour libérer l'espace"
    rm -rf "$REPO_ROOT/$local_dir"
    echo "[benchmark] Disque après suppression :"
    df -h /home/criteo | tail -1
}

run_model \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "models/Llama-3.2-1B-Instruct" \
    "0_baselines/exp11_llama1b_baseline" \
    "You are a helpful assistant."

run_model \
    "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    "models/SmolLM2-1.7B-Instruct" \
    "0_baselines/exp12_smollm2_baseline" \
    "You are a helpful AI assistant."

run_model \
    "google/gemma-3-1b-it" \
    "models/Gemma-3-1B-it" \
    "0_baselines/exp13_gemma3_baseline" \
    ""

run_model \
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    "models/DeepSeek-R1-Distill-Qwen-1.5B" \
    "0_baselines/exp14_deepseekr1_baseline" \
    "You are a helpful assistant."

echo ""
echo "========================================"
echo "[benchmark] ALL DONE $(date -u)"
echo "========================================"
echo ""
echo "Résultats :"
for run in 0_baselines/exp11_llama1b_baseline 0_baselines/exp12_smollm2_baseline 0_baselines/exp13_gemma3_baseline 0_baselines/exp14_deepseekr1_baseline; do
    n_success=$(grep -l '"reward": 1' "$REPO_ROOT/runs/$run/eval_logs/"*.json 2>/dev/null | wc -l)
    n_total=$(ls "$REPO_ROOT/runs/$run/eval_logs/"*.json 2>/dev/null | wc -l)
    echo "  $run : $n_success / $n_total"
done
