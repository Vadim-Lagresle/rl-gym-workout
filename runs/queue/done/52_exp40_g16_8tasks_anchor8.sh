#!/bin/bash
# Job 52 — exp40 : contrôle PROPRE G=16 × 8 tâches par pas, SANS curriculum,
# ancre mobile au même rythme EN PAS que le contrôle exp36 et les curriculums.
#
# Pourquoi : exp39 (job 50/51) est identique à exp36 sauf 8 tâches/pas au lieu de 4,
# MAIS l'ancre est définie en époques (4) : à 46,75 pas/époque elle tombe tous les
# 187 pas au lieu de 374 pour exp36 (93,5 pas/époque). Deux facteurs confondus.
# Ici : --moving-anchor-every-epochs 8 → 8 × 46,75 = 374 pas entre deux ancres,
# soit exactement le rythme (en pas, en trajectoires et en remises à zéro d'Adam)
# d'exp36 et des trois curriculums. SEUL delta avec exp39 : 4 → 8 époques d'ancre.
# SEUL delta avec exp36 : 64 → 128 trajectoires par pas (4 → 8 tâches par pas).
#
# Décision Vadim 08/09/2026. Checkpoints périodiques sur le HOME (règle 04/09).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp40_g16_8tasks_anchor8
bash setup/ensure_qwen_tmp.sh || { echo "[job52] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job52.log 2>&1 \
  || { echo "[job52] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job52] $(date '+%F %T') — exp40 G=16 × 8 tâches/pas, ancre /8 époques (= 374 pas), sans curriculum, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 128 \
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
bash scripts/append_results.sh "$RUN" "exp40 — G=16 × 8 tâches/pas, ancre /8 ép. (374 pas comme exp36), sans curriculum : contrôle propre du confondeur"
