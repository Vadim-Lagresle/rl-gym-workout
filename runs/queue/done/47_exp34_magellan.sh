#!/bin/bash
# exp34 — PHASE 2 AUTOCURRICULUM MAGELLAN (port fidèle Gaven et al. 2025, voir
# src/train/magellan.py et docs/MAGELLAN_ANALYSE.md) : recette exp25/best-73 à
# l'identique, départ Qwen BASE, mais les items sont échantillonnés ∝ progrès
# d'apprentissage ABSOLU prédit par un estimateur de compétence appris sur les
# embeddings du LLM de la politique (tête SR MLP + adapters LoRA séparés r16/α32,
# BCE à chaque step, compétence retardée par snapshots — leurs hyperparamètres :
# N=100, ε 1.0→0.2 decay 320, buffer 5000, batch 256, recompute /32 steps).
# Prédiction écrite AVANT le run (MAGELLAN_ANALYSE.md §4) : l'ALP doit désinvestir
# depth 4 (support nul, 1 seul item train) et concentrer sur la frontière d2-d3.
# Harmonisation 24/08 (Vadim) : N=16 à 64 traj/step constant (4 prompts/step) +
# gpu-util 0.5 — verdict bench exp25.1, aligné sur exp32 v2. NB pour l'ALP : le
# sampler ne tire plus que 4 items/step (vs 8) — moitié moins d'items visités par
# step, mais succès/item estimé sur 16 rollouts (SR par item plus précis).
# 80 epochs — arrêt manuel par Vadim. Suivi : wandb magellan/* + logs/magellan_<run>.jsonl.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp34_magellan
bash setup/ensure_qwen_tmp.sh || { echo "[job47] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Pré-vol 1 (CPU) : selftests schedules (ancre mobile multi-adapters) + magellan.
python src/train/schedules.py > logs/selftest_schedules_job47.log 2>&1 \
  || { echo "[job47] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/magellan.py > logs/selftest_magellan_job47.log 2>&1 \
  || { echo "[job47] selftest magellan ÉCHOUÉ — run ANNULÉ"; exit 1; }

# Pré-vol 2 (GPU, ~10 min) : smoke bout-en-bout — adapters SR sur le vrai 3B,
# sync vLLM, sampler pondéré, sr_update, ré-ancrage compatible multi-adapters.
rm -rf saves/trl_grpo/smoke_magellan_anchors
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 2 --gradient-accumulation-steps 4 \
    --max-items 8 --max-steps 3 --max-completion-length 128 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 0.4 \
    --goal-sampler magellan --magellan-recompute-freq 2 --magellan-batch-size 8 \
    --use-vllm-inprocess --run-name smoke_magellan > logs/smoke_magellan.log 2>&1 \
  || { echo "[job47] SMOKE magellan ÉCHOUÉ — run réel ANNULÉ (logs/smoke_magellan.log)"; exit 1; }

echo "=== [job47] $(date '+%F %T') — exp34 autocurriculum MAGELLAN, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --goal-sampler magellan \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp34 — autocurriculum MAGELLAN (ALP appris)"
