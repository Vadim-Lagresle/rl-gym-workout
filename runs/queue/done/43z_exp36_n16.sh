#!/bin/bash
# exp36 — CONTRÔLE N=16 sans curriculum (ajouté le 24/08 sur décision Vadim,
# c'est le bras "exp36_n16" anticipé par le bench exp25.1) : recette exp25/best-73
# STRICTEMENT à l'identique (r8/α32, LR 3e-6, β0.01, ancre mobile /4 ep, départ
# Qwen BASE, horizon '30:0' constant), SEULS deltas : N=16 rollouts/prompt à
# 64 traj/step constant (4 prompts/step) + gpu-util 0.5 (config D du bench :
# 46.9 s/it vs 57.1 pour N=8/0.17). Double rôle :
#   1. effet propre de N (vs exp25, N=8) : meilleure estimation d'avantage GRPO,
#      groupes mixtes ~2× plus fréquents sur les items difficiles ;
#   2. baseline sans-curriculum à N=16 → lève le double-delta d'exp32 v2
#      (horizon + N) et sert de référence homogène à toute la phase 2.
# On reste à N=16 (pas 32) : à 64 traj/step, N=32 = 2 items/update seulement
# (gradient dominé par 2 tâches) ; monter GA à 128 doublerait le temps de step.
# 80 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp36_n16
bash setup/ensure_qwen_tmp.sh || { echo "[job43z] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job43z.log 2>&1 \
  || { echo "[job43z] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job43z] $(date '+%F %T') — exp36 contrôle N=16 sans curriculum, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp36 — contrôle N=16 sans curriculum"
