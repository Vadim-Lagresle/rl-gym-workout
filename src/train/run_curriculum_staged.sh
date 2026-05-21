#!/usr/bin/env bash
# Curriculum d'entraînement par depth : 3 phases successives.
# Chaque phase reprend depuis le checkpoint de la phase précédente.
#
# Usage :
#   bash src/train/run_curriculum_staged.sh <run_name>
#
# Pré-requis :
#   1. conda activate agentenv-textcraft && python src/utils/label_depths.py
#   2. Serveur TextCraft lancé sur 127.0.0.1:36005
#   3. conda activate agentgym-rl

set -euo pipefail

RUN_NAME="${1:-exp9_curriculum_depth}"
SAVES_ROOT="saves/trl_grpo"

PHASE1_NAME="${RUN_NAME}_phase1"
PHASE2_NAME="${RUN_NAME}_phase2"
PHASE3_NAME="${RUN_NAME}_phase3"

PHASE1_CKPT="${SAVES_ROOT}/${PHASE1_NAME}/checkpoint-100"
PHASE2_CKPT="${SAVES_ROOT}/${PHASE2_NAME}/checkpoint-100"

COMMON_ARGS="--use-vllm --full-ft --num-generations 8 --max-completion-length 512 --max-items 0 --max-steps 100"

echo "=== Phase 1 : depth ≤ 2 ==="
python src/train/train_grpo.py ${COMMON_ARGS} \
    --max-depth 2 \
    --run-name "${PHASE1_NAME}"

echo "=== Phase 2 : depth ≤ 3 (resume depuis phase 1) ==="
python src/train/train_grpo.py ${COMMON_ARGS} \
    --max-depth 3 \
    --resume-from-checkpoint "${PHASE1_CKPT}" \
    --run-name "${PHASE2_NAME}"

echo "=== Phase 3 : tous les depths (resume depuis phase 2) ==="
python src/train/train_grpo.py ${COMMON_ARGS} \
    --max-depth 0 \
    --resume-from-checkpoint "${PHASE2_CKPT}" \
    --run-name "${PHASE3_NAME}"

echo "=== Curriculum terminé. Checkpoints finaux : ==="
echo "  ${SAVES_ROOT}/${PHASE3_NAME}/"