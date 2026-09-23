#!/bin/bash
# Job 57 — exp44 : curriculum HORIZON à G=8 — le run manquant de la phase 2 (décision Vadim 15/09).
# = exp25 (LoRA r8/α32, LR 3e-6, β0,01, entropy-coef 0,001, G=8 × 8 tâches = 64 traj/pas,
#   ancre mobile /4 ép., départ Qwen base) + UN SEUL delta : horizon progressif 10/20/30 tours
#   aux époques 0/15/30 (le calendrier d'exp32).
# Lecture : vs exp25 ('30:0', 65 @ ép. 111 / 73 après reprise) → effet propre du curriculum à G fixe ;
#           vs exp32 (même curriculum, G=16, 82 @ ép. 66) → effet propre de G à curriculum fixe.
# Le 24/08, la v1 d'exp32 à G=8 avait été avortée avant la 1re éval (paliers trop rapides + passage
# de toute la phase 2 à G=16) : l'effet du curriculum n'a jamais été isolé de celui de G.
# Pas de --kl-clamp : comparabilité stricte avec exp25 et exp32 (à G=8 aucun collapse observé).
# gpu-util 0,5 comme exp41 (sans effet sur la sémantique). 80 ép., ckpt sur le HOME (règle 04/09).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp44_horizon_g8
bash setup/ensure_qwen_tmp.sh || { echo "[job57] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job57.log 2>&1 \
  || { echo "[job57] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job57] $(date '+%F %T') — exp44 Horizon 10/20/30 (ép. 0/15/30) à G=8 × 8 tâches, ancre /4 ép., 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule-epochs '10:0,20:15,30:30' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp44 — curriculum Horizon 10/20/30 à G=8 (exp25 + un seul delta) : effet propre du curriculum"
