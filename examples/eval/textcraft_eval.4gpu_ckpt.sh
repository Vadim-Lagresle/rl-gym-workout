#!/usr/bin/env bash
# ============================================================================
# textcraft_eval.4gpu_ckpt.sh
#
# Post-training : prend un checkpoint verl FSDP (sharded sur 4 ranks) sauvegardé
# par `textcraft_train.4gpu.sh`, le merge en HuggingFace, puis lance l'eval
# sur 100 items (taux de succès avec **un rollout par item** — les scripts
# affichent aussi le libellé « Pass@1 », voir WORKLOG 2026-05-13) via
# `scratch/03_eval_qwen.py` (= vLLM standard, pas verl,
# pour éviter de re-exercer le path multi-GPU NCCL pour seulement une eval).
#
# Variables d'env :
#   CKPT_DIR=<path>   → dossier saves/.../global_step_N qui contient les .pt
#                        FSDP. Si non fourni : on prend le dernier checkpoint
#                        dans `saves/agentgym_rl_4gpu/<latest>/global_step_N`.
#   EVAL_TAG=<str>    → suffixe pour le LOG_DIR d'eval.
# ============================================================================
set -euo pipefail
set -x

source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl

REPO_ROOT="/home/v.lagresle/rl-gym-workout"
cd "$REPO_ROOT"

# --- 1. Résoudre CKPT_DIR ----------------------------------------------------
if [[ -z "${CKPT_DIR:-}" ]]; then
    LATEST_RUN="$(ls -1dt "$REPO_ROOT/saves/agentgym_rl_4gpu"/*/ 2>/dev/null | head -1)"
    if [[ -z "$LATEST_RUN" ]]; then
        echo "No training run found in saves/agentgym_rl_4gpu/. Set CKPT_DIR explicitly." >&2
        exit 1
    fi
    LATEST_STEP_DIR="$(ls -1dt "$LATEST_RUN"global_step_* 2>/dev/null | head -1)"
    if [[ -z "$LATEST_STEP_DIR" ]]; then
        echo "No global_step_* checkpoint in $LATEST_RUN" >&2
        exit 1
    fi
    CKPT_DIR="$LATEST_STEP_DIR"
fi

# verl writes actor shards under <CKPT_DIR>/actor/
ACTOR_DIR="$CKPT_DIR/actor"
if [[ ! -d "$ACTOR_DIR" ]]; then
    echo "Actor shards not found: $ACTOR_DIR" >&2
    exit 1
fi

HF_OUT_DIR="$ACTOR_DIR/huggingface"
EVAL_TAG="${EVAL_TAG:-$(basename "$CKPT_DIR")}"
LOG_DIR="$REPO_ROOT/scratch/eval_logs_4gpu_${EVAL_TAG}"

# --- 2. Merger FSDP → HuggingFace (idempotent) -------------------------------
# Ne pas se fier seul à config.json : un merge interrompu peut laisser tokenizer
# + config sans aucun *.safetensors → vLLM plante avec « Cannot find any model weights ».
# FORCE_MERGE=1 force un re-merge même si des poids sont déjà présents.
FORCE_MERGE="${FORCE_MERGE:-0}"
_has_weights=""
if [[ -d "$HF_OUT_DIR" ]]; then
    _has_weights="$(find "$HF_OUT_DIR" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model.bin' \) -print -quit 2>/dev/null || true)"
fi
if [[ "$FORCE_MERGE" == "1" ]] || [[ -z "$_has_weights" ]]; then
    echo "[eval] merging FSDP shards -> $HF_OUT_DIR (FORCE_MERGE=$FORCE_MERGE, had_weights=$([[ -n "$_has_weights" ]] && echo yes || echo no))"
    python3 "$REPO_ROOT/AgentGym-RL/scripts/model_merger.py" --local_dir "$ACTOR_DIR"
else
    echo "[eval] found merged weights under $HF_OUT_DIR, skipping merge (set FORCE_MERGE=1 to redo)"
fi

# --- 3. Vérif serveur TextCraft ----------------------------------------------
if ! curl -sSf -m 5 -X POST http://127.0.0.1:36005/create \
        -H 'content-type: application/json' -d '{}' >/dev/null; then
    echo "TextCraft server unreachable on :36005." >&2
    exit 1
fi

# --- 4. Eval Pass@1 (100 items) ---------------------------------------------
mkdir -p "$LOG_DIR"
MODEL_PATH="$HF_OUT_DIR" EVAL_LOG_DIR="$LOG_DIR" \
    python3 "$REPO_ROOT/scratch/03_eval_qwen.py" 2>&1 | tee "$LOG_DIR/eval.log"

# --- 5. Résumé Pass@1 sur stdout --------------------------------------------
python3 - <<PY
import json, pathlib
log_dir = pathlib.Path("$LOG_DIR")
results = sorted(log_dir.glob("textcraft_*.json"))
rewards = []
for f in results:
    try:
        d = json.loads(f.read_text())
        rewards.append(float(d.get("reward", 0)))
    except Exception:
        pass
if not rewards:
    print("No episode logs found in $LOG_DIR")
else:
    pass1 = 100.0 * sum(1 for r in rewards if r > 0) / len(rewards)
    print(f"\n========================================")
    print(f" Eval done: {len(rewards)} episodes")
    print(f" Pass@1 = {pass1:.1f} / 100")
    print(f"========================================\n")
PY
