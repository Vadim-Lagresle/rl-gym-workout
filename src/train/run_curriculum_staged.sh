#!/usr/bin/env bash
# Curriculum d'entraînement par depth (exp9) — 4 stages séquentiels.
#
# Chaque stage est entraîné UNIQUEMENT sur les items de son depth (--depth-exact),
# pour un nombre d'epochs propre (--num-epochs, = passes sur le sous-ensemble du depth),
# et démarre en warm-start depuis le checkpoint final du stage précédent (--model-path).
#
# Curriculum (10 epochs au total) :
#   stage 1 : depth 1   (109 items) × 3 epochs
#   stage 2 : depth 2   (221 items) × 4 epochs
#   stage 3 : depth 3   (43 items)  × 2 epochs
#   stage 4 : depth 3+4 (44 items)  × 1 epoch   (depth 4 seul = 1 item, fusionné avec depth 3)
#
# Note disque : les checkpoints (~5.8 Go chacun) sont écrits sous saves/trl_grpo/, qui est
# un symlink vers /tmp (overlay 233 Go). On supprime le checkpoint du stage k dès que le
# stage k+1 a démarré (poids déjà chargés), pour ne jamais garder plus de 2 stages.
#
# Pré-requis :
#   1. ~/envs/agentenv-textcraft/bin/python src/utils/label_depths.py  (depuis le dossier du pkg)
#      -> data/train/textcraft_train_with_depth.json
#   2. Serveur TextCraft up sur 127.0.0.1:36005
#   3. models/Qwen2.5-3B-Instruct présent
#
# Usage :
#   bash src/train/run_curriculum_staged.sh [run_name]

set -euo pipefail

RUN_NAME="${1:-exp9_curriculum_depth}"
PY="${PY:-$HOME/envs/agentgym-rl/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-3B-Instruct}"
SAVES_ROOT="saves/trl_grpo"

S1="${RUN_NAME}_s1_d1"
S2="${RUN_NAME}_s2_d2"
S3="${RUN_NAME}_s3_d3"
S4="${RUN_NAME}_s4_d34"

# Batch papier : grad_accum=256, N=8 -> 32 prompts/step. Mêmes hyperparamètres validés exp8.2.
COMMON="--full-ft --num-generations 8 --gradient-accumulation-steps 256 \
--max-completion-length 512 --max-items 0 --use-vllm-inprocess --save-steps 100000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
    local name="$1" depth="$2" epochs="$3" init="$4"
    echo "=== Stage ${name} : depth=${depth}, epochs=${epochs}, init=${init} ==="
    "$PY" src/train/train_grpo.py ${COMMON} \
        --depth-exact "${depth}" \
        --num-epochs "${epochs}" \
        --model-path "${init}" \
        --run-name "${name}"
}

run_stage "$S1" "1"   3 "$BASE_MODEL"
run_stage "$S2" "2"   4 "${SAVES_ROOT}/${S1}"
rm -rf "${SAVES_ROOT}/${S1}"
run_stage "$S3" "3"   2 "${SAVES_ROOT}/${S2}"
rm -rf "${SAVES_ROOT}/${S2}"
run_stage "$S4" "3,4" 1 "${SAVES_ROOT}/${S3}"
rm -rf "${SAVES_ROOT}/${S3}"

echo "=== Curriculum terminé. Checkpoint final : ${SAVES_ROOT}/${S4} ==="
echo "Eval : bash src/utils/start_vllm_server.sh ${SAVES_ROOT}/${S4} && \\"
echo "       python src/eval/eval_textcraft.py --model ${SAVES_ROOT}/${S4} --run-name 4_curriculum/${RUN_NAME}_final"
