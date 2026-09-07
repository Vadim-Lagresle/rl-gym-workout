#!/bin/bash
# exp31.1 — PHASE 1 ablation d'ancre : ancre FIXE + β0.0001 (redéfinie le 24/08).
# Question : l'ancre mobile est-elle NÉCESSAIRE, ou un β quasi nul avec ancre fixe
# (Qwen nu du début à la fin) suffit-il à ce type d'alignement ? La lignée dit :
# ancre fixe + β0.001 = collapse (exp24), ancre fixe + β0.01 = collapse ep ~11
# (exp30) ; ici on relâche presque toute la contrainte KL (100× sous exp30) SANS
# ré-ancrage. Recette exp25/best-73 sinon identique (r8/α32, lr 3e-6, N=8, GA 64).
# 200 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp31.1_r8_fixedanchor_b00001

bash setup/ensure_qwen_tmp.sh || { echo "[job42b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Pré-vol CPU : selftests schedules.
python src/train/schedules.py > logs/selftest_schedules_job42b.log 2>&1 \
  || { echo "[job42b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }

echo "=== [job42b] $(date '+%F %T') — exp31.1 ancre FIXE r8 β0.0001, 200 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.0001 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 200 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp31.1 — ancre FIXE + β0.0001 (ablation : l'ancre mobile est-elle nécessaire ?)"
