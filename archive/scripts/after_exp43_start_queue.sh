#!/bin/bash
# Attend la fin du processus d'entraînement exp43 (job 56, lancé hors runner le 15/09) puis démarre
# le runner de file (runs/queue/57_… puis 58, 59). Un seul runner : vérifie qu'aucun ne tourne déjà.
cd "$(dirname "$0")/.."
while pgrep -f "run-name exp43_g16_klclamp10" > /dev/null; do sleep 120; done
echo "[after-exp43] $(date '+%F %T') — exp43 terminé, démarrage du runner de file" >> logs/queue.log
pgrep -f "scripts/run_queue.sh" > /dev/null || setsid nohup bash scripts/run_queue.sh >> logs/queue.log 2>&1 &
