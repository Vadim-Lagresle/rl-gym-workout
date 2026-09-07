#!/bin/bash
# exp37 — GRPO simple + FEW-SHOT (ajouté 27/08, décision Vadim) : pas de curriculum,
# recette N=16 avec ancre mobile /12 ep (alignée exp36.1), SEUL delta : --fewshot 10.
# Question : le few-shot amorce et accélère les premières epochs (acquis exp18/22.6 :
# +8 pts vs zero-shot), mais tient-il la durée à N=16 ? Risque identifié (Vadim) :
# l'algo pourrait sur-regarder les exemples (reprendre les recettes des exemples
# plutôt que celle de la tâche courante) — surveiller err_recipe_wrong à l'éval.
# Passe APRÈS exp36.1 (49 > 48) — SI le calendrier le permet (gel 2/09) ; sinon le
# rapport dira : « le few-shot a permis de démarrer/accélérer, sans garantie de
# stabilité ni de plus-value vs GRPO classique ».
# 80 epochs — arrêt manuel.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp37_fewshot_n16
bash setup/ensure_qwen_tmp.sh || { echo "[job49] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job49.log 2>&1 \
  || { echo "[job49] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job49] $(date '+%F %T') — exp37 GRPO simple + few-shot 10, N=16, ancre /12 ep, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 12 \
    --fewshot 10 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp37 — GRPO simple + few-shot 10 (N=16, ancre /12 ep)"
