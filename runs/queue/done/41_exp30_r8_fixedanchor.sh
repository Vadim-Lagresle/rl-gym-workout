#!/bin/bash
# exp30 — PHASE 1 ablation d'ancre, case "ancre FIXE + β0.01" du carré ancre × β.
# Recette exp25 À L'IDENTIQUE sauf : PAS d'ancre mobile (aucun --moving-anchor-*),
# 20 epochs. Question : la montée d'exp25 vient-elle de l'ancre mobile, ou
# r8 + β0.01 + LR 3e-6 suffisait ? (les 2 cases "ancre fixe β0.001 / β0.01"
# r16 existent dans exp24, celle-ci isole le rang r8 à β0.01.)
# Se lance automatiquement quand le job 40 (exp25) se termine ou est coupé.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp30_r8_fixedanchor

bash setup/ensure_qwen_tmp.sh || { echo "[job41] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

echo "=== [job41] $(date '+%F %T') — exp30 ancre fixe r8 β0.01, 20 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 20 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
