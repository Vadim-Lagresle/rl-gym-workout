#!/usr/bin/env bash
# S'assure que Qwen2.5-3B-Instruct est présent sur /tmp (volatil, purgé parfois).
# Politique disque 2026-07-29 : le home (35 Go) ne stocke QUE les best adapters
# (+ optimizers) ; les poids de base sont retéléchargés ici à la demande.
set -euo pipefail

DEST="${1:-/tmp/models/Qwen2.5-3B-Instruct}"
REPO="Qwen/Qwen2.5-3B-Instruct"

if [ -f "$DEST/config.json" ] && ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "[ensure_qwen_tmp] déjà présent : $DEST"
    exit 0
fi

echo "[ensure_qwen_tmp] téléchargement de $REPO vers $DEST ..."
mkdir -p "$DEST"
# hf est fourni par huggingface_hub (présent dans l'env v2 et l'env textcraft)
HF_CLI="$(command -v hf || command -v huggingface-cli)"
"$HF_CLI" download "$REPO" --local-dir "$DEST"
echo "[ensure_qwen_tmp] OK : $(du -sh "$DEST" | cut -f1)"
