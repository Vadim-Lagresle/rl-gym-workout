#!/bin/bash
# exp33 — PHASE 2 curriculum DEPTH (design humain) : recette exp25/best-73 à
# l'identique, départ Qwen BASE, mais l'échantillonnage des items suit des paliers
# de difficulté : depth<=1 dès l'epoch 0, <=2 dès l'epoch 12, <=3 dès l'epoch 25,
# <=4 dès l'epoch 37 (paliers recalés par Vadim le 26/08 pour un run de 80 ep ;
# implémenté par sampler pondéré, dataset entier — flag
# --depth-schedule-epochs, src/train/magellan.py DepthScheduleProvider).
# Répartition du train : d1=109, d2=221, d3=43, d4=1 (le palier 4 n'ajoute qu'UN item).
# Question : ordonner par difficulté aide-t-il le transfert d3-d4 vs uniforme (exp25) ?
# Harmonisation 24/08 (Vadim) : N=16 à 64 traj/step constant (4 prompts/step) +
# gpu-util 0.5 — verdict bench exp25.1, aligné sur exp32 v2 (comparabilité phase 2).
# 80 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp33_depth
bash setup/ensure_qwen_tmp.sh || { echo "[job44] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job44.log 2>&1 \
  || { echo "[job44] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/magellan.py > logs/selftest_magellan_job44.log 2>&1 \
  || { echo "[job44] selftest magellan (sampler pondéré) ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job44] $(date '+%F %T') — exp33 curriculum depth, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --depth-schedule-epochs '1:0,2:12,3:25,4:37' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp33 — curriculum depth (paliers humains)"
