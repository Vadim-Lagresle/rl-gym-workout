#!/bin/bash
# exp31 — PHASE 1 ablation d'ancre, case "ancre MOBILE + β0.001" du carré ancre × β.
# Recette exp25 À L'IDENTIQUE sauf : --beta 0.001 (au lieu de 0.01), 20 epochs.
# Question : l'ancre mobile (ré-ancrage /4 ep) sauve-t-elle le régime β faible,
# qui en ancre FIXE donnait des pics 31-35 puis collapse (exp24 β0.001) ?
# Si oui → c'était l'ancre, pas le β (réponse à la remarque de Vadim du 18/08).
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp31_r8_anchor_b0001

bash setup/ensure_qwen_tmp.sh || { echo "[job42] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Pré-vol CPU : selftests schedules (MovingAnchor).
python src/train/schedules.py > logs/selftest_schedules_job42.log 2>&1 \
  || { echo "[job42] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }

echo "=== [job42] $(date '+%F %T') — exp31 ancre mobile r8 β0.001, 20 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.001 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 150 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp31 — ancre mobile + β0.001 (ablation phase 1)"
