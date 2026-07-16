#!/usr/bin/env bash
# ============================================================================
# auto_eval_after_train.sh
#
# Surveille le PID du training verl jusqu'à sa fin, puis :
#   1. attend que le dernier checkpoint soit bien écrit,
#   2. déclenche `textcraft_eval.4gpu_ckpt.sh` (merge + eval, métrique = succès/100
#      avec 1 rollout par item — affichée « Pass@1 » dans les scripts),
#   3. dump le résumé final dans /tmp/eval_summary.txt.
#
# Lancement (en arrière-plan) :
#   TRAIN_PID=<pid> nohup setsid bash scratch/auto_eval_after_train.sh \
#       > /tmp/auto_eval.log 2>&1 < /dev/null & disown
# ============================================================================
set -uo pipefail

TRAIN_PID="${TRAIN_PID:?need TRAIN_PID}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[auto-eval] watching PID=$TRAIN_PID..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "[auto-eval] training process exited at $(date -Iseconds), waiting 30s for I/O flush..."
sleep 30

cd "$REPO_ROOT"
bash external/agentgym_rl_paper/eval/textcraft_eval.4gpu_ckpt.sh 2>&1 | tee /tmp/eval_summary.txt
echo "[auto-eval] done at $(date -Iseconds)"
