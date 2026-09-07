#!/bin/bash
# exp26 — full-FT Qwen2.5-7B, recette exp23 À L'IDENTIQUE, 150 epochs (~6600 steps).
# Objectif : vérifier les chiffres du papier AgentGym-RL sur le 7B avec notre stack TRL.
# Design : runs/14_fullft_7b/exp26_7b_fullft_150ep/config.yaml
# Disque : best = POIDS SEULS sur le home (--best-delete-before-save, pas d'optimizer :
# 15 Go + 15 Go ne tiennent pas sur 35 Go) ; ckpts périodiques sur /tmp (volatils).
# Durée attendue : 7-11 jours GPU (~25-40 steps/h) — arrêt manuel si plafond/collapse.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp26_7b_fullft_150ep
MODEL=/tmp/models/Qwen2.5-7B-Instruct

bash setup/ensure_qwen_tmp.sh "$MODEL" Qwen/Qwen2.5-7B-Instruct \
  || { echo "[job32] téléchargement 7B ÉCHOUÉ"; exit 1; }

# --- Étape 1 : smoke GPU (~15 min) — le 7B + Adam 8-bit + vLLM colocate tiennent-ils ?
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --full-ft --model-path "$MODEL" \
    --num-generations 2 --gradient-accumulation-steps 4 \
    --max-items 8 --max-steps 2 --max-completion-length 128 \
    --vllm-gpu-util 0.2 \
    --use-vllm-inprocess --run-name smoke_7b > logs/smoke_7b.log 2>&1 \
  || { echo "[job32] SMOKE 7B ÉCHOUÉ (OOM ?) — run réel ANNULÉ (logs/smoke_7b.log)"; exit 1; }
echo "[job32] smoke 7B OK — lancement du run réel (150 epochs)"

# --- Étape 2 : run réel — recette exp23 : vrai PPO buffer 256, LR 1e-6, β 0.001,
# entropie 0.001, zero-shot, max_rounds 30.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --full-ft --model-path "$MODEL" \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --steps-per-generation 256 --entropy-coef 0.001 \
    --learning-rate 1e-6 --beta 0.001 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 150 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.2 \
    --eval-every 47 --eval-items 100 \
    --best-init-score -1 --best-delete-before-save \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" > "logs/$RUN.log" 2>&1
