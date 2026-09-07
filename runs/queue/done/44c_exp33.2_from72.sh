#!/bin/bash
# exp33.2 — REPRISE d'exp33.1 depuis son BEST (72/100 @ step 4042, epoch ~44.5).
# Purge pod 29/08 ~23:06 a tué le run ET le runner de file en silence (job 44b
# resté dans runs/queue/, jamais déplacé dans done/ — parqué manuellement dans
# skipped/ pour ne pas le rejouer bêtement depuis zéro).
#
# Même principe que exp25.2/exp25 : la politique du best 72 est ENTIÈREMENT
# reconstructible depuis le home :
#   politique_72 = Qwen3B ⊕ cycles 1-11 (chaîne d'ancres) ⊕ adapter best (step 4042)
# On merge le tout en un modèle complet qui sert d'INIT et d'ANCRE KL, puis on
# repart avec un adapter r8 FRAIS — exactement ce que MovingAnchorCallback
# aurait fait au ré-ancrage suivant, aucune rupture de sémantique.
# Optimizer : PRÉSENT sur le home (job 44b avait --save-best-optimizer) mais
# --warm-start-dir ne le charge qu'en full-FT (train_grpo.py:639) ; en LoRA la
# politique repart avec un adapter neuf comme exp25.2 — moments Adam à zéro,
# coût faible (purgés tous les 4 epochs de toute façon par l'ancre mobile).
#
# Curriculum depth : REPRISE AU PALIER 4 (décision finale Vadim 30/08 — comme
# si le run ne s'était pas arrêté). exp33.1 était au palier depth<=4 (dataset
# complet, 374/374 items) depuis l'epoch 22.14 et y a construit son best 72 ;
# le curriculum avait donc terminé son travail de gradation, on ne le rejoue
# pas. --depth-auto-start-stage 4 : le provider démarre au palier final et la
# règle d'avancement est inerte (butée max_depth) — équivalent d'un GRPO
# uniforme sur le dataset entier, dans la continuité exacte du run tué.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp33.2_from72
SRC=exp33.1_depth_auto
BASE=/tmp/models/exp33.1_best72_full

bash setup/ensure_qwen_tmp.sh || { echo "[job44c] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Garde-fous : sans la chaîne + le best sur le home, rien à reprendre.
[ -f "saves/trl_grpo/${SRC}_anchors/chain.jsonl" ] \
  || { echo "[job44c] chaîne d'ancres absente — ABANDON"; exit 1; }
[ -f "saves/trl_grpo/${SRC}_best/adapter_model.safetensors" ] \
  || { echo "[job44c] adapter best 72 absent — ABANDON"; exit 1; }

if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[job44c] reconstruction politique 72 (Qwen3B ⊕ cycles 1-11 ⊕ adapter best) → $BASE"
  python src/utils/merge_anchor_chain.py \
      --base /tmp/models/Qwen2.5-3B-Instruct \
      --anchors "saves/trl_grpo/${SRC}_anchors" \
      --adapter "saves/trl_grpo/${SRC}_best" \
      --out "$BASE" \
    || { echo "[job44c] merge chaîne+best ÉCHOUÉ"; exit 1; }
fi

# Pré-vol CPU : selftests schedules + magellan (depth-auto).
python src/train/schedules.py > logs/selftest_schedules_job44c.log 2>&1 \
  || { echo "[job44c] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/magellan.py > logs/selftest_magellan_job44c.log 2>&1 \
  || { echo "[job44c] selftest magellan (depth-auto) ÉCHOUÉ — run ANNULÉ"; exit 1; }

echo "=== [job44c] $(date '+%F %T') — exp33.2 reprise depuis best 72 au palier 4, 40 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --model-path "$BASE" \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 40 --max-rounds-schedule '30:0' \
    --depth-schedule-auto --depth-auto-threshold 0.8 --depth-auto-max-epochs 10 \
    --depth-auto-start-stage 4 \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --best-init-score 0.72 --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp33.2 — reprise depth-auto depuis best 72 (palier final)"
