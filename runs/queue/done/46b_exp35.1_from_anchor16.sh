#!/bin/bash
# exp35.1 — REPRISE d'exp35 (curriculum de budget de sortie) après purge pod
# du 01/09. Le run est mort au step 5956 / epoch 64.04 sur 7440 steps (80 ep) :
# il reste 16 epochs pour finir le budget prévu (décision Vadim 01/09 :
# « il reste moins de 20 epochs, on finit ça aujourd'hui »).
#
# REPRISE « À LA DERNIÈRE ANCRE » (le mécanisme que Vadim jugeait le plus juste,
# noté en backlog le 30/08 — ici il est GRATUIT et EXACT) :
#   le MovingAnchorCallback a ré-ancré au cycle 16 au step 5952 / epoch 64.0,
#   soit 4 STEPS avant la mort. Or après un ré-ancrage la politique vaut
#   exactement  base ⊕ cycle1 ⊕ … ⊕ cycle16  (adapter neuf, B=0, delta ≡ 0)
#   et les moments Adam viennent d'être purgés par le callback.
#   On reconstruit donc la CHAÎNE SEULE (merge_anchor_chain.py sans --adapter) :
#     * politique identique à celle du step 5952 (4 steps perdus, pas 833) ;
#     * ancre KL au dernier point de ré-ancrage RÉEL du run (pas déplacée sur
#       le best, contrairement à la reprise exp33.2) ;
#     * moments Adam à zéro = état exact du run à cet instant ;
#     * le prochain ré-ancrage du nouveau run tombe à SON epoch 4, soit
#       64 + 4 = 68 en cumulé — exactement le calendrier d'origine.
#   C'est la reprise la plus fidèle possible depuis les artefacts du home.
#
# Budget de sortie : le curriculum ('256:0,512:15,1024:35') avait atteint son
# palier FINAL (1024 tokens/tour) à l'epoch 35. Le nouveau run repart d'un
# compteur d'epochs à zéro : on fixe donc directement 1024 (sans schedule),
# sinon le curriculum redémarrerait à 256 tokens.
# --best-init-score 0.78 : ne remplacer le best du home (78/100 @ step 5123)
# que si le nouveau run fait mieux.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp35.1_from_anchor16
SRC=exp35_budget1024
BASE=/tmp/models/exp35_anchor16_full

bash setup/ensure_qwen_tmp.sh || { echo "[job46b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Garde-fous : sans la chaîne d'ancres complète, rien à reprendre.
[ -f "saves/trl_grpo/${SRC}_anchors/chain.jsonl" ] \
  || { echo "[job46b] chaîne d'ancres absente — ABANDON"; exit 1; }
[ -d "saves/trl_grpo/${SRC}_anchors/cycle16" ] \
  || { echo "[job46b] cycle16 absent — ABANDON"; exit 1; }

if [ ! -f "$BASE/model.safetensors" ] && [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[job46b] reconstruction politique step 5952 (Qwen3B ⊕ cycles 1-16, CHAÎNE SEULE) → $BASE"
  python src/utils/merge_anchor_chain.py \
      --base /tmp/models/Qwen2.5-3B-Instruct \
      --anchors "saves/trl_grpo/${SRC}_anchors" \
      --out "$BASE" \
    || { echo "[job46b] merge chaîne ÉCHOUÉ"; exit 1; }
fi

# Pré-vol CPU : selftests schedules (MovingAnchor + completion-schedule).
python src/train/schedules.py > logs/selftest_schedules_job46b.log 2>&1 \
  || { echo "[job46b] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }

echo "=== [job46b] $(date '+%F %T') — exp35.1 reprise à l'ancre 16 (step 5952), 16 epochs pour finir les 80 ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --model-path "$BASE" \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 1024 --max-items 0 \
    --num-epochs 16 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --best-init-score 0.78 --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp35.1 — reprise exp35 à l'ancre 16, 16 ep finales (budget 1024)"
