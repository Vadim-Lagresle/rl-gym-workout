#!/bin/bash
# append_results.sh <run_name> <titre court> — appelé en FIN de chaque job de la
# file : ajoute à docs/RESULTS.md le best et la trajectoire d'éval du run
# (les interprétations sont complétées à la relecture, voir aussi runs/INDEX.md).
cd "$(dirname "$0")/.."
RUN=$1
TITLE=$2
LOG=logs/$RUN.log
OUT=docs/RESULTS.md
{
  echo
  echo "## $TITLE — \`$RUN\` (fin $(date '+%F %H:%M'))"
  echo
  BEST=saves/trl_grpo/${RUN}_best/.best_info
  if [ -f "$BEST" ]; then
    echo "**Best test : $(cat "$BEST")**"
    echo
  fi
  echo '```'
  grep "Pass@1 =" "$LOG" | tail -40
  echo '```'
  echo
  echo "_Interprétation : à compléter à la relecture._"
} >> "$OUT"
echo "[results] $RUN ajouté à $OUT"
