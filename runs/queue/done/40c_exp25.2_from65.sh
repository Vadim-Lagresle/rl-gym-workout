#!/bin/bash
# exp25.2 — reprise d'exp25 depuis son BEST (65/100 @ step 4794, epoch ~104.2).
# Décision 20/08 : objectif 70. La purge /tmp du 20/08 a emporté checkpoint-périodique
# ET optimizer (le job 40 ne passait pas --save-best-optimizer — corrigé partout depuis).
#
# Principe : la politique du best 65 est ENTIÈREMENT reconstructible depuis le home :
#   politique_65 = Qwen3B ⊕ cycles 1-26 (chaîne d'ancres) ⊕ adapter best (step 4794)
# On merge le tout en un modèle complet qui sert à la fois d'INIT et d'ANCRE KL, puis
# on repart avec un adapter r8 FRAIS — c'est exactement ce que le MovingAnchorCallback
# aurait fait au ré-ancrage suivant, donc aucune rupture de sémantique.
# Optimizer : perdu — moments Adam à zéro. Coût réel faible ici : le callback les
# purgeait déjà tous les 4 epochs (perte max = les moments intra-cycle).
# Nouveau run name (exp25.2_from65) → nouvelle chaîne d'ancres, zéro collision.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp25.2_from65
SRC=exp25_r8_anchor4ep
BASE=/tmp/models/exp25_best65_full

bash setup/ensure_qwen_tmp.sh || { echo "[job40c] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Garde-fous : sans la chaîne + le best sur le home, rien à reprendre — on n'improvise pas.
[ -f "saves/trl_grpo/${SRC}_anchors/chain.jsonl" ] \
  || { echo "[job40c] chaîne d'ancres absente — ABANDON"; exit 1; }
[ -f "saves/trl_grpo/${SRC}_best/adapter_model.safetensors" ] \
  || { echo "[job40c] adapter best 65 absent — ABANDON"; exit 1; }

if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[job40c] reconstruction politique 65 (Qwen3B ⊕ cycles 1-26 ⊕ adapter best) → $BASE"
  python src/utils/merge_anchor_chain.py \
      --base /tmp/models/Qwen2.5-3B-Instruct \
      --anchors "saves/trl_grpo/${SRC}_anchors" \
      --adapter "saves/trl_grpo/${SRC}_best" \
      --out "$BASE" \
    || { echo "[job40c] merge chaîne+best ÉCHOUÉ"; exit 1; }
fi

# Pré-vol CPU : selftests schedules (dont MovingAnchor).
python src/train/schedules.py > logs/selftest_schedules_job40c.log 2>&1 \
  || { echo "[job40c] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }

echo "=== [job40c] $(date '+%F %T') — exp25.2 depuis best 65, 60 epochs, objectif 70 ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --model-path "$BASE" \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 60 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --best-init-score 0.65 --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
