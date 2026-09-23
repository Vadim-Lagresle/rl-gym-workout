#!/bin/bash
# Job 53b — REPRISE EXACTE d'exp41 après la recréation du pod du 11/09 ~06:45 (/tmp purgé,
# processus tués). Checkpoint-2162 (epoch 47, adapter + optimizer + états) sur le HOME.
# Base = Qwen3B ⊕ cycles 1-5 (steps ≤ 2162, cf. chain.jsonl) reconstruite sur /tmp ;
# le cycle 6 (step 2208, postérieur au ckpt) a été mis de côté dans post_ckpt2162_run1/.
# --moving-anchor-initial-cycle 5 → prochain ré-ancrage à l'époque 48, comme dans le run 1.
# Recette strictement identique au job 53. Décision : reprise (règle Vadim 04/09).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp41_g8_8tasks_anchor8
CKPT=saves/trl_grpo_ckpt/$RUN/checkpoint-2162
ANCHORS=saves/trl_grpo/${RUN}_anchors
BASE=/tmp/models/exp41_anchor5_base
[ -f "$CKPT/optimizer.pt" ] && [ -f "$CKPT/adapter_model.safetensors" ] && [ -f "$ANCHORS/chain.jsonl" ] \
  || { echo "[job53b] checkpoint-2162 ou chaîne d'ancres incomplets — run ANNULÉ"; exit 1; }
bash setup/ensure_qwen_tmp.sh || { echo "[job53b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job53b.log 2>&1 \
  || { echo "[job53b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[job53b] reconstruction de la base ancrée (Qwen3B ⊕ cycles 1-5, steps ≤ 2162) → $BASE"
  python src/utils/merge_anchor_chain.py \
      --base /tmp/models/Qwen2.5-3B-Instruct --anchors "$ANCHORS" --adapter-step 2162 --out "$BASE" \
    || { echo "[job53b] merge de la chaîne d'ancres ÉCHOUÉ"; exit 1; }
fi
echo "=== [job53b] $(date '+%F %T') — REPRISE exp41 depuis $CKPT (epoch 47) sur base ancrée cycle 5 ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --model-path "$BASE" \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 8 --moving-anchor-initial-cycle 5 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer --best-init-score 0.62 \
    --save-steps 94 --save-total-limit 1 \
    --resume-from-checkpoint "$CKPT" \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp41 — G=8 × 8 tâches/pas, ancre /8 ép. (374 pas), sans curriculum (reprise ckpt-2162 le 11/09)"
