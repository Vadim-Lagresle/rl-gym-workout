#!/bin/bash
# Job 53 — exp41 : référence G=8 (8 tâches × 8 traj = 64 traj/pas), SANS curriculum,
# ancre mobile toutes les 8 époques = 8 × 46,75 = 374 pas : le rythme EN PAS d'exp36,
# d'exp40 et des trois curriculums.
#
# Pourquoi : exp40 (8 tâches × 16 = 128 traj/pas, ancre 374 pas) a collapsé comme exp36
# (4 × 16 = 64 traj/pas, ancre 374 pas). Mais exp40 consommait 2× plus de trajectoires
# par pas que les curriculums. exp41 remet 64 traj/pas avec 8 tâches, à ancre 374 pas :
# - s'il tient  → à 64 traj/pas et ancre 374, 8 tâches suffisent ; le collapse d'exp40
#                 tenait aux 128 traj/pas (ou à G=16) ;
# - s'il casse  → la référence G=8 (exp25) ne tenait que grâce à son ancre à 187 pas ;
#                 sans curriculum, seule l'ancre fréquente protège.
# Seul delta avec exp25 : ancre 4 → 8 époques (et 80 ép., ckpt home, vllm-gpu-util 0.5
# comme les curriculums). Décision Vadim 10/09/2026.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp41_g8_8tasks_anchor8
bash setup/ensure_qwen_tmp.sh || { echo "[job53] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job53.log 2>&1 \
  || { echo "[job53] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job53] $(date '+%F %T') — exp41 G=8 × 8 tâches/pas (64 traj), ancre /8 époques (= 374 pas), sans curriculum, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 8 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp41 — G=8 × 8 tâches/pas (64 traj), ancre /8 ép. (374 pas), sans curriculum"
