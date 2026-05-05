#!/usr/bin/env bash
# ============================================================================
# textcraft_eval.local.sh
# Version adaptée à notre setup local : 1× A100 40 Go + Qwen2.5-3B-Instruct (HF brut).
# Différences vs `textcraft_eval.sh` upstream :
#   1. Active conda proprement (`source activate` upstream ne suffit pas).
#   2. Pas de `model_merger.py` (pas de checkpoint à fusionner, on part d'un HF brut).
#   3. `model.path` pointe directement sur `models/Qwen2.5-3B-Instruct/`.
#   4. Override `trainer.n_gpus_per_node=1` (le yaml par défaut suppose 8).
#   5. `data.path` pointe sur un dossier dédié qu'on construit pour contourner
#      la logique `category_map` de `main_generation.py` (sinon KeyError).
#   6. Paramètres mémoire un peu moins agressifs (gpu_mem 0.85, max_model_len 16384).
#   7. `rollout.load_format=safetensors` au lieu du `dummy_dtensor` par défaut.
#      `dummy_dtensor` est un format CUSTOM de verl qui (a) initialise vLLM avec des
#      poids random puis (b) attend qu'on patche les vrais poids depuis un DTensor
#      FSDP issu du training PPO. Pour de l'éval avec un modèle HF brut, ce flow
#      n'a pas lieu : vLLM tente d'init NCCL sur des tenseurs invalides et crash
#      en SIGSEGV silencieux. `safetensors` charge directement les vrais poids.
# ============================================================================
set -euo pipefail
set -x

# ----------------------------------------------------------------------------
# 1. Activation conda
# ----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl

# ----------------------------------------------------------------------------
# 2. Variables d'environnement vLLM
# ----------------------------------------------------------------------------
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

# ----------------------------------------------------------------------------
# 3. Paramètres locaux
# ----------------------------------------------------------------------------
REPO_ROOT="/home/v.lagresle/rl-gym-workout"
cd "$REPO_ROOT"

TASK_NAME="textcraft"
ENV_SERVER_URL="http://127.0.0.1:36005"
MODEL_PATH="$REPO_ROOT/models/Qwen2.5-3B-Instruct"

# ----------------------------------------------------------------------------
# 4. Sanity check : serveur TextCraft up sur ENV_SERVER_URL ?
# ----------------------------------------------------------------------------
if ! curl -sSf -X POST "$ENV_SERVER_URL/create" \
        -H 'content-type: application/json' -d '{}' >/dev/null; then
    echo "ERROR: TextCraft server unreachable at $ENV_SERVER_URL." >&2
    echo "Start it with:" >&2
    echo "  cd $REPO_ROOT/AgentGym/agentenv-textcraft && \\" >&2
    echo "  conda activate agentenv-textcraft && \\" >&2
    echo "  textcraft --host 127.0.0.1 --port 36005" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 5. Préparation du dataset attendu par main_generation.py
# main_generation.py attend dans <data.path> :
#   - <task>_test.json     : les items à évaluer (champ "item_id")
#   - <task>_<other>.json  : "catégories" pour le breakdown stats
# Sans au moins une catégorie, le code plante en KeyError sur category_map[item_id].
# Workaround : on copie textcraft_test.json sous deux noms, dont un en "catégorie".
#
# MAX_ITEMS (env var, défaut: 0=tout) : tronque le dataset pour smoke tests.
# ----------------------------------------------------------------------------
MAX_ITEMS="${MAX_ITEMS:-0}"
EVAL_DIR="$REPO_ROOT/AgentEval/textcraft"
mkdir -p "$EVAL_DIR"
cp -f "$REPO_ROOT/AgentEval/eval/textcraft_test.json" "$EVAL_DIR/textcraft_test.json"
cp -f "$REPO_ROOT/AgentEval/eval/textcraft_test.json" "$EVAL_DIR/textcraft_default.json"

if [[ "$MAX_ITEMS" -gt 0 ]]; then
    python3 - <<PY
import json
for fn in ["$EVAL_DIR/textcraft_test.json", "$EVAL_DIR/textcraft_default.json"]:
    with open(fn) as f:
        items = json.load(f)
    items = items[:$MAX_ITEMS]
    with open(fn, "w") as f:
        json.dump(items, f, indent=2)
    print(f"[truncate] {fn} -> {len(items)} items")
PY
fi

# ----------------------------------------------------------------------------
# 6. Lancement de la génération multi-tour
# ----------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    data.path="$EVAL_DIR" \
    data.max_prompt_length=750 \
    data.max_response_length=14098 \
    data.n_samples=1 \
    data.batch_size=8 \
    agentgym.task_name="$TASK_NAME" \
    agentgym.env_addr="$ENV_SERVER_URL" \
    agentgym.max_rounds=30 \
    agentgym.timeout=500 \
    model.path="$MODEL_PATH" \
    rollout.gpu_memory_utilization=0.85 \
    rollout.temperature=1 \
    rollout.max_model_len=16384 \
    rollout.max_tokens=512 \
    rollout.tensor_model_parallel_size=1 \
    rollout.load_format=safetensors \
    rollout.rollout_log_dir="$REPO_ROOT/executer_logs"
