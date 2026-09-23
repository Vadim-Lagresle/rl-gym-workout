#!/bin/bash
# Job 56b — REPRISE EXACTE d'exp43 (Contrôle-G16 + borne k3 = 10) après la recréation du pod du 17/09 ~09:00.
# Mort à l'ép. 58,6 (pas 5446) ; checkpoint-5358 (ép. 57,6, adapter + optimizer + états) sur le HOME.
# Base = Qwen3B ⊕ cycles 1-14 (steps ≤ 5358, dernier cycle à 5208) ; --moving-anchor-initial-cycle 14 →
# prochain ré-ancrage à l'ép. 60 comme dans le run 1. Recette identique au job 56 (kl-clamp 10 inclus).
# Objectif : les 22 époques manquantes, pour comparer le plateau à celui d'Horizon (78 sur les ép. 60-80).
# Décision Vadim 22/09 (expérience A).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp43_g16_klclamp10
CKPT=saves/trl_grpo_ckpt/$RUN/checkpoint-5358
ANCHORS=saves/trl_grpo/${RUN}_anchors
BASE=/tmp/models/exp43_anchor14_base
[ -f "$CKPT/optimizer.pt" ] && [ -f "$CKPT/adapter_model.safetensors" ] && [ -f "$ANCHORS/chain.jsonl" ] \
  || { echo "[job56b] checkpoint-5358 ou chaîne d'ancres incomplets — run ANNULÉ"; exit 1; }
bash setup/ensure_qwen_tmp.sh || { echo "[job56b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python setup/patch_trl_kl_clamp.py || { echo "[job56b] patch k3 TRL ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job56b.log 2>&1 \
  || { echo "[job56b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[job56b] reconstruction de la base ancrée (Qwen3B ⊕ cycles 1-14, steps ≤ 5358) → $BASE"
  python src/utils/merge_anchor_chain.py \
      --base /tmp/models/Qwen2.5-3B-Instruct --anchors "$ANCHORS" --adapter-step 5358 --out "$BASE" \
    || { echo "[job56b] merge de la chaîne d'ancres ÉCHOUÉ"; exit 1; }
fi
echo "=== [job56b] $(date '+%F %T') — REPRISE exp43 depuis $CKPT (ép. 57,6) sur base ancrée cycle 14, kl-clamp 10 ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --model-path "$BASE" \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --kl-clamp 10 \
    --moving-anchor-every-epochs 4 --moving-anchor-initial-cycle 14 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer --best-init-score 0.77 \
    --save-steps 94 --save-total-limit 1 \
    --resume-from-checkpoint "$CKPT" \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp43 — Contrôle-G16 + borne k3 (reprise ckpt-5358 le 22/09, ép. 57,6 → 80)"
