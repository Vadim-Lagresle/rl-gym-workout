#!/bin/bash
# Job 54 — exp42 : autocurriculum MAGELLAN (sélection des tâches par progrès d'apprentissage)
# COMBINÉ au curriculum Horizon (ScalingInter, tours 10 → 20 → 30 aux époques 0, 15, 30).
#
# Pourquoi : MAGELLAN seul (exp34) a collapsé à l'époque 7 selon la signature G=16 sans
# curriculum ; l'annexe E conclut qu'il suppose un apprenant stable. Horizon (exp32, 82/78)
# fournit cette base. Question : l'autocurriculum ajoute-t-il quelque chose à une base
# stabilisée (vitesse, profondeur 3), ou au moins ne nuit-il pas ?
# Job = celui d'exp32 (43b) + `--goal-sampler magellan`, rien d'autre ne change.
# Comparaisons : exp32 (Horizon seul), exp34 (MAGELLAN seul). Décision Vadim 11/09/2026.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp42_magellan_horizon
bash setup/ensure_qwen_tmp.sh || { echo "[job54] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job54.log 2>&1 \
  || { echo "[job54] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job54] $(date '+%F %T') — exp42 MAGELLAN + Horizon 10/20/30 (ép. 0/15/30), G=16, 64 traj/pas, ancre /4 ép., 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule-epochs '10:0,20:15,30:30' \
    --goal-sampler magellan \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp42 — MAGELLAN + curriculum Horizon (ScalingInter), G=16, ancre /4 ép."
