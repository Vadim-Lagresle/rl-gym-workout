#!/bin/bash
# exp35 — PHASE 2 CURRICULUM DE BUDGET DE SORTIE (redéfinie 26/08, Vadim — option A) :
# recette exp25/best-73 à l'identique, départ Qwen BASE, mais le budget de tokens
# PAR TOUR monte par paliers : 256 dès l'epoch 0, 512 dès l'epoch 15, 1024 dès
# l'epoch 35 (--max-completion-schedule-epochs, schedules.current_max_completion —
# même mécanique que ScalingInter, appliquée à max_tokens au lieu du nb de tours).
# --max-completion-length reste 1024 = le MAX du schedule (budgets TRL dimensionnés
# dessus ; garde-fou dans train_grpo.py). Question : commencer concis (crédit dense,
# esprit exp32) puis débloquer du raisonnement par tour aide-t-il depth 3 ?
# Coût attendu : fin de run plus lente (épisodes 1024×30 tours proches de
# --vllm-max-len 32768 — surveiller préemptions/troncatures vLLM dans le log).
# Harmonisation 24/08 (Vadim) : N=16 à 64 traj/step constant (4 prompts/step) +
# gpu-util 0.5 — verdict bench exp25.1, aligné sur exp32 v2. NB : 1024 tokens/tour
# × 30 tours = épisodes plus longs que le pire cas benché (KV/épisode ↑) — si
# préemptions vLLM dans le log, c'est LE run où surveiller.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp35_budget1024
bash setup/ensure_qwen_tmp.sh || { echo "[job46] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job46.log 2>&1 \
  || { echo "[job46] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job46] $(date '+%F %T') — exp35 curriculum budget 256-1024/tour, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 1024 --max-items 0 \
    --max-completion-schedule-epochs '256:0,512:15,1024:35' \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp35 — curriculum budget de sortie 256→512→1024/tour"
