#!/bin/bash
# Attend la fin de setup_agentgym_rl_v2.sh + ensure_qwen_tmp.sh (log logs/setup_env_20260922.log),
# puis démarre le runner de file (56b exp43 reprise → 56c exp45 → 57 exp44 → 58 exp41 reprise).
cd "$(dirname "$0")/.."
LOG=logs/setup_env_20260922.log
for i in $(seq 1 240); do   # ≤ 2 h
  grep -q "\[ensure_qwen_tmp\] \(OK\|déjà présent\)" $LOG 2>/dev/null && [ -x /tmp/envs/agentgym-rl-v2/bin/python ] && break
  grep -q "ÉCHOU\|Traceback" $LOG 2>/dev/null && { echo "[start-queue] $(date '+%F %T') — setup en échec, runner NON démarré" >> logs/queue.log; exit 1; }
  sleep 30
done
echo "[start-queue] $(date '+%F %T') — env prêt, démarrage du runner de file" >> logs/queue.log
pgrep -f "scripts/run_queue.sh" > /dev/null || setsid nohup bash scripts/run_queue.sh >> logs/queue.log 2>&1 &
