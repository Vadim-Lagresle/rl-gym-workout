#!/bin/bash
# Job 55 — rejeu k3 token par token (étape 2 de l'analyse du collapse, 15/09/2026).
# Quatre checkpoints, même tirage de 64 tâches × 4 épisodes (seed 0), même protocole :
#   exp40 pas 705  : G=16 sans curriculum, 5 pas avant l'explosion (entropie 0,15)   — malade
#   exp34 pas 611  : MAGELLAN, ép. 6,5, une demi-époque avant le collapse (entropie 0,66) — sur le point
#   exp32 pas 7238 : Horizon, best 82, entropie 0,25, stable                         — sain à entropie basse
#   exp41 pas 2632 : G=8 ancre /8, best 64, entropie 0,8, stable                     — sain à entropie normale
# Modèles fusionnés par le script scratchpad merge_all.sh → /tmp/models/replay/<tag>_{ref,policy}.
# Décision Vadim 15/09 (« lance l'étape deux »).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
LOG=logs/replay_k3_20260915.log
for tag in exp40_s705 exp34_s611 exp32_s7238 exp41_s2632; do
  P=/tmp/models/replay/${tag}_policy R=/tmp/models/replay/${tag}_ref
  for i in $(seq 1 120); do  # attend la fin des fusions (≤ 60 min)
    [ -f $P/model.safetensors ] && [ -f $R/model.safetensors ] && [ -f $P/tokenizer.json ] && break; sleep 30
  done
  [ -f $P/model.safetensors ] || { echo "[job55] fusion absente pour $tag — SAUTÉ" >> $LOG; continue; }
  echo "=== [job55] $(date '+%F %T') — rejeu $tag ===" >> $LOG
  python src/analysis/replay_k3_tokens.py --policy $P --ref $R --tag $tag \
      --n-items 64 --episodes-per-item 4 --seed 0 --out-dir runs/15_replay_k3 >> $LOG 2>&1 \
    || echo "[job55] ÉCHEC $tag" >> $LOG
done
echo "=== [job55] $(date '+%F %T') — terminé ===" >> $LOG
