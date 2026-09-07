#!/bin/bash
# exp33.1 — curriculum DEPTH AUTO-DÉCLENCHÉ (règle Vadim 28/08) : remplace exp33
# (paliers calendaires 0/12/25/37), morte purge pod 28/08 ~01:48 à l'ep ~22 en
# pleine stagnation (reward médian figé à 0.25 sur le palier d<=2 pendant 10 ep,
# éval 26-29 ; et 8 ep PERDUES à reward=1.0 saturé sur le palier d<=1 : la limite
# des paliers statiques, documentée §2.4/§4.3 du rapport). Nouvelle règle :
# palier suivant si reward train MOYEN sur la dernière epoch complète >= 0.8,
# sinon passage forcé après 10 epochs au palier (magellan.py
# DepthAutoScheduleProvider, selftest CPU obligatoire en pré-vol).
# Recette N=16 identique à exp32/33 (r8/α32, LR 3e-6, β0.01, ancre mobile /4 ep,
# gpu-util 0.5). 80 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp33.1_depth_auto
bash setup/ensure_qwen_tmp.sh || { echo "[job44b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job44b.log 2>&1 \
  || { echo "[job44b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/magellan.py > logs/selftest_magellan_job44b.log 2>&1 \
  || { echo "[job44b] selftest magellan (depth-auto) ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job44b] $(date '+%F %T') — exp33.1 depth auto (seuil 0.8 / cap 10 ep), 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --depth-schedule-auto --depth-auto-threshold 0.8 --depth-auto-max-epochs 10 \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp33.1 — curriculum depth auto-déclenché (seuil 0.8, cap 10 ep)"
