#!/usr/bin/env bash
# In plain words: a 3-update training run on the GPU that goes through every moving part
# of the recipe (LoRA, KL bound, moving KL anchor with one re-anchoring, depth-weighted task
# draw on the extended training file, periodic evaluation on 5 test tasks), then checks the
# log for each of them. Takes a few minutes. Usage:
#   bash src/tests/smoke_train.sh [merge|ref] [gpu_util]
# Exit code 0 = everything fired; non-zero = the first missing signal is printed.
set -u
cd "$(dirname "$0")/../.."
MODE=${1:-merge}
UTIL=${2:-0.3}
RUN=smoke_refacto_$MODE
LOG=logs/$RUN.log
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
rm -rf "saves/trl_grpo/${RUN}_anchors"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 2 --gradient-accumulation-steps 4 \
    --max-items 8 --max-steps 3 --max-completion-length 128 \
    --learning-rate 3e-6 --beta 0.01 --entropy-coef 0.001 --kl-clamp 10 \
    --moving-anchor-every-epochs 0.4 --moving-anchor-mode "$MODE" \
    --train-file data/train/textcraft_train_plus_reservoir.json --depth-balance uniform \
    --eval-every 2 --eval-items 5 \
    --vllm-gpu-util "$UTIL" --use-vllm-inprocess --run-name "$RUN" > "$LOG" 2>&1
code=$?
rm -rf "saves/trl_grpo/${RUN}_anchors" "saves/trl_grpo/${RUN}_best"
[ $code -eq 0 ] || { echo "[smoke] train_grpo.py a échoué (code $code) — voir $LOG"; tail -20 "$LOG"; exit 1; }
for sig in "\[kl-clamp\] TRL_KL_CLAMP=10" "\[data\] fichier de train" "\[depth-balance\]" \
           "RÉ-ANCRAGE #1" "\[test_eval\] step 2 — Pass@1" "TRL+GRPO training run finished"; do
    grep -q "$sig" "$LOG" || { echo "[smoke] signal absent : $sig — voir $LOG"; exit 1; }
done
echo "[smoke] OK ($MODE) — kl-clamp, train file, depth balance, re-anchoring, eval, end of run"
