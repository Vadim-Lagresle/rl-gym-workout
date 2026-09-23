#!/usr/bin/env bash
# In plain words: downloads Qwen2.5-3B-Instruct to /tmp/models if it is missing (/tmp
# is wiped when the machine is reset). Safe to call before every job.
#
# S'assure que Qwen2.5-3B-Instruct est présent sur /tmp (volatil, purgé parfois).
# Politique disque 2026-07-29 : le home (35 Go) ne stocke QUE les best adapters
# (+ optimizers) ; les poids de base sont retéléchargés ici à la demande.
set -euo pipefail

# Usage : ensure_qwen_tmp.sh [dest] [repo_hf]
#   défaut          : Qwen2.5-3B-Instruct
#   7B (exp26)      : ensure_qwen_tmp.sh /tmp/models/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
DEST="${1:-/tmp/models/Qwen2.5-3B-Instruct}"
REPO="${2:-Qwen/Qwen2.5-3B-Instruct}"

if [ -f "$DEST/config.json" ] && ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "[ensure_qwen_tmp] déjà présent : $DEST"
    exit 0
fi

echo "[ensure_qwen_tmp] téléchargement de $REPO vers $DEST ..."
mkdir -p "$DEST"
# hf est fourni par huggingface_hub (présent dans l'env v2 et l'env textcraft).
# Le PATH n'est pas fiable selon le contexte (cron, setsid, sandbox) : on tente le
# PATH puis les emplacements connus des deux envs.
HF_CLI=""
for cand in hf huggingface-cli \
            "$HOME/envs/agentenv-textcraft/bin/hf" \
            /tmp/envs/agentgym-rl-v2/bin/hf; do
    if command -v "$cand" >/dev/null 2>&1; then
        HF_CLI="$(command -v "$cand")"
        break
    fi
done
if [ -z "$HF_CLI" ]; then
    echo "[ensure_qwen_tmp] ERREUR : CLI hf/huggingface-cli introuvable (PATH et envs connus)" >&2
    exit 1
fi
"$HF_CLI" download "$REPO" --local-dir "$DEST"
echo "[ensure_qwen_tmp] OK : $(du -sh "$DEST" | cut -f1)"
