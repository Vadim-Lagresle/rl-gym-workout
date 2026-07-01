#!/bin/bash
# Copie périodique du dernier checkpoint de /tmp (volatil — purgé par l'infra
# le 2026-06-12) vers le home. Garde UNE seule copie (place limitée : 7.8G libres).
# Usage : nohup bash src/utils/backup_latest_ckpt.sh <run_name> &
RUN=${1:?usage: backup_latest_ckpt.sh <run_name>}
SRC=/tmp/trl_grpo_saves/$RUN
DST=/home/criteo/rl-gym-workout/saves/trl_grpo/${RUN}_backup
LOG=/home/criteo/rl-gym-workout/logs/ckpt_backup.log

while true; do
  # Dernier checkpoint complet : écrit depuis plus de 5 min (une save dure ~2 min,
  # on évite de copier un checkpoint en cours d'écriture).
  latest=$(ls -dt "$SRC"/checkpoint-* 2>/dev/null | head -1)
  if [ -n "$latest" ] && [ -n "$(find "$latest" -maxdepth 0 -mmin +5 2>/dev/null)" ] \
     && [ "$(basename "$latest")" != "$(cat "$DST/.ckpt_name" 2>/dev/null)" ]; then
    # rm-AVANT-cp : le home n'a de place que pour UN checkpoint (~5.8G sur ~7G libres).
    # L'ancienne version copiait vers .tmp d'abord -> exigeait 2x la place (11.6G) ->
    # echec systematique des le 2e backup (weekend du 13/06). Fenetre de ~2min sans
    # backup pendant la copie, acceptable vs cycle 2h ; ckpt400 reste le plancher.
    rm -rf "$DST"
    if cp -r "$latest" "$DST" && basename "$latest" > "$DST/.ckpt_name"; then
      echo "$(date '+%F %T') sauvegardé $(basename "$latest") -> $DST" >> "$LOG"
    else
      echo "$(date '+%F %T') ECHEC copie $(basename "$latest")" >> "$LOG"
    fi
  fi
  sleep 7200
done
