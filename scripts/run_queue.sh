#!/bin/bash
# run_queue.sh — exécute les jobs runs/queue/*.sh en séquence (ordre lexical).
#
# Usage (une seule fois, survit à la fermeture de la session Cursor) :
#   setsid nohup bash scripts/run_queue.sh > logs/queue.log 2>&1 &
#
# Fonctionnement :
# - chaque job est un script bash autonome (une expérience) exécuté en FOREGROUND :
#   le runner attend sa fin avant de passer au suivant ;
# - entre deux jobs : sanity checks de la stack (/tmp env v2, Qwen, serveur TextCraft),
#   reconstruction automatique si /tmp a été vidé pendant le job précédent ;
# - un job terminé (quel que soit son exit code) est déplacé dans runs/queue/done/ ;
# - on peut ajouter/retirer des jobs dans runs/queue/ pendant que le runner tourne
#   (la file est relue à chaque itération).
#
# LIMITE assumée : une purge pod tue ce runner comme tout le reste. Au redémarrage,
# le relancer (même commande) — il reprend à la première ligne non consommée. Le job
# interrompu N'est PAS relancé automatiquement (il est resté dans runs/queue/ : il
# repartira de zéro, ou le déplacer soi-même dans done/ pour le sauter).

set -u
cd "$(dirname "$0")/.."   # racine du repo
QUEUE_DIR=runs/queue
DONE_DIR=$QUEUE_DIR/done
mkdir -p "$DONE_DIR" logs

ensure_stack() {
    # env v2 sur /tmp (reconstruit après purge : ~10 min grâce au wheel flash-attn)
    [ -d /tmp/envs/agentgym-rl-v2 ] || bash setup/setup_agentgym_rl_v2.sh
    # Qwen2.5-3B sur /tmp (no-op si présent)
    bash setup/ensure_qwen_tmp.sh
    export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
    # serveur TextCraft (lancé depuis le dossier du package : chemin relatif recipes/)
    if ! curl -s -o /dev/null --max-time 3 http://127.0.0.1:36005/; then
        echo "[queue] serveur TextCraft absent — relance"
        (cd external/AgentGym/agentenv-textcraft \
         && setsid nohup "$HOME/envs/agentenv-textcraft/bin/textcraft" \
              --host 127.0.0.1 --port 36005 > /tmp/textcraft_server.log 2>&1 &)
        sleep 8
    fi
}

while true; do
    job=$(ls "$QUEUE_DIR"/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$job" ]; then
        echo "[queue] $(date '+%F %T') — file vide, terminé"
        break
    fi
    name=$(basename "$job")
    echo "[queue] $(date '+%F %T') — démarrage $name"
    ensure_stack
    bash "$job"
    rc=$?
    echo "[queue] $(date '+%F %T') — fin $name (exit $rc)"
    mv "$job" "$DONE_DIR/$name"
    # wandb local : synchronisé en live pendant le run, on libère le disque
    rm -rf wandb/run-* wandb/latest-run 2>/dev/null || true
done
