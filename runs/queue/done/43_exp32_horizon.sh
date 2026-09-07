#!/bin/bash
# exp32 — PHASE 2 curriculum HORIZON (ScalingInter) : recette exp25/best-73 à
# l'identique (r8/α32, LR 3e-6, β0.01, ancre mobile /4 ep, N=8, batch 64), départ
# Qwen BASE, mais l'horizon d'épisode est progressif : 10 tours dès l'epoch 0,
# 20 dès l'epoch 8, 30 dès l'epoch 20 (vs '30:0' constant pour la baseline exp25).
# Question : démarrer court accélère-t-il l'apprentissage (épisodes courts = crédit
# plus dense) sans coûter le score final ? 200 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp32_horizon
bash setup/ensure_qwen_tmp.sh || { echo "[job43] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job43.log 2>&1 \
  || { echo "[job43] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job43] $(date '+%F %T') — exp32 curriculum horizon, 200 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 200 --max-rounds-schedule-epochs '10:0,20:8,30:20' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp32 — curriculum horizon (ScalingInter)"
