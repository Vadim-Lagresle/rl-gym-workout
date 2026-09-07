#!/bin/bash
# exp32 — PHASE 2 curriculum HORIZON (ScalingInter) — v2 corrigée le 24/08 (Vadim) :
# la v1 (job 43, avortée avant la 1re éval) montait l'horizon trop vite (paliers
# aux epochs 0/8/20). Ici : 10 tours dès l'epoch 0, 20 dès l'epoch 15, 30 dès
# l'epoch 30. Et on applique ENFIN le verdict du bench GPU exp25.1 (20/08) :
# N=16 rollouts/prompt à 64 traj/step constant (4 prompts/step) + gpu-util 0.5
# = config D, 46.9 s/it vs 57.1 pour N=8/0.17 — meilleure estimation d'avantage
# GRPO et groupes mixtes plus fréquents, POUR UN TEMPS MOINDRE.
# ⚠ double delta vs baseline exp25 ('30:0', N=8) : horizon ET N — assumé (24/08).
# Recette exp25/best-73 sinon (r8/α32, LR 3e-6, β0.01, ancre mobile /4 ep,
# départ Qwen BASE). 200 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp32_horizon
bash setup/ensure_qwen_tmp.sh || { echo "[job43b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job43b.log 2>&1 \
  || { echo "[job43b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job43b] $(date '+%F %T') — exp32 curriculum horizon v2 (10/20/30 @ ep 0/15/30, N=16, gpu-util 0.5), 200 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 200 --max-rounds-schedule-epochs '10:0,20:15,30:30' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp32 — curriculum horizon (ScalingInter, N=16)"
