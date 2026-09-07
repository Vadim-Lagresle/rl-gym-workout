#!/bin/bash
# exp25 (suite) — ancre KL mobile r8, prolongation à 200 EPOCHS (décision 18/08 :
# soutenance < 1 mois → stabiliser LoRA 3B vite, le best d'exp25 était SA DERNIÈRE
# éval (35 @ step 690/fin), le run était encore ascendant).
#
# Deux branches :
#   REPRISE (si checkpoint-690 + optimizer + chaîne d'ancres présents) :
#     - reconstruit la base ancrée = Qwen3B ⊕ cycles 1-3 (merge_anchor_chain.py)
#     - --resume-from-checkpoint : adapter + optimizer + compteurs repris au step 690
#     - --moving-anchor-initial-cycle 3 : prochain ré-ancrage à l'epoch 16 (sinon le
#       callback enchaînerait des ré-ancrages parasites dès la reprise)
#     - --best-init-score 0.35 : ne remplace le best (35) que si on fait mieux
#   SCRATCH (si /tmp purgé) : même expé DE ZÉRO sur 200 epochs, anciens
#     _anchors/_best archivés en *_run1 pour éviter les collisions de chaîne.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp25_r8_anchor4ep
CKPT=/tmp/trl_grpo_runs/$RUN/checkpoint-690
ANCHORS=saves/trl_grpo/${RUN}_anchors

bash setup/ensure_qwen_tmp.sh || { echo "[job40] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Pré-vol CPU : selftests schedules (dont MovingAnchor + reprise sans ré-ancrage parasite).
python src/train/schedules.py > logs/selftest_schedules_job40.log 2>&1 \
  || { echo "[job40] selftest schedules ÉCHOUÉ — run ANNULÉ (logs/selftest_schedules_job40.log)"; exit 1; }

EXTRA=()
if [ -f "$CKPT/adapter_model.safetensors" ] && [ -f "$CKPT/optimizer.pt" ] \
   && [ -f "$ANCHORS/chain.jsonl" ]; then
  BASE=/tmp/models/exp25_anchor3_base
  if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
    echo "[job40] reconstruction de la base ancrée (Qwen3B ⊕ cycles 1-3) → $BASE"
    python src/utils/merge_anchor_chain.py \
        --base /tmp/models/Qwen2.5-3B-Instruct --anchors "$ANCHORS" --out "$BASE" \
      || { echo "[job40] merge de la chaîne d'ancres ÉCHOUÉ"; exit 1; }
  fi
  EXTRA=(--model-path "$BASE" --resume-from-checkpoint "$CKPT"
         --moving-anchor-initial-cycle 3 --best-init-score 0.35)
  echo "[job40] REPRISE depuis $CKPT (epoch 15, optimizer inclus) sur base ancrée cycle 3"
else
  ts=$(date +%Y%m%d)
  [ -d "$ANCHORS" ] && mv "$ANCHORS" "${ANCHORS}_run1_$ts"
  [ -d "saves/trl_grpo/${RUN}_best" ] && mv "saves/trl_grpo/${RUN}_best" "saves/trl_grpo/${RUN}_best_run1_$ts"
  echo "[job40] checkpoint-690 absent (/tmp purgé ?) — même expé DE ZÉRO, 200 epochs"
fi

echo "=== [job40] $(date '+%F %T') — exp25 200 epochs (${EXTRA[0]:-scratch}) ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 200 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    "${EXTRA[@]}" \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
