#!/bin/bash
# Recette d'installation ÉPINGLÉE pour faire tourner le code verl du papier AgentGym-RL
# (external/AgentGym-RL) sur un nœud 8× H100 — cible Nebius, 2026-08-06.
#
# POURQUOI CETTE RECETTE EST CONTRAINTE
# ------------------------------------
# Leur boucle de rollout n'importe pas vLLM directement mais une version PATCHÉE et
# vendorisée dans leur repo :
#     verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:38
#         from verl.third_party.vllm import LLM, vllm_version
# et verl/third_party/vllm/__init__.py:30-60 aiguille sur la version EXACTE de vLLM
# installée (0.3.1 / 0.4.2 / 0.5.4 / 0.6.3, ou un chemin SPMD séparé pour >=0.6.6).
# Pour une réplication FIDÈLE du run à 75/100, il faut viser exactement vLLM 0.6.3,
# qui impose (requirements-cuda.txt de vLLM 0.6.3) :
#     torch==2.4.0  torchvision==0.19  xformers==0.0.27.post2  transformers>=4.45
# soit une stack CUDA 12.1 — parfaite sur H100 (sm90), IMPOSSIBLE sur notre B200
# (sm100 exige CUDA 12.8+/torch 2.7+). C'est la raison de fond de louer les H100.
#
# transformers est épinglé à 4.47.1 : c'est la version contre laquelle verl 0.2.0.post2
# est écrit (cf. le commentaire "adapt from transformers 4.47.1" dans
# verl/models/transformers/llama.py:44). À partir de 4.48, transformers a supprimé les
# classes Qwen2FlashAttention2/LlamaFlashAttention2 que leur monkeypatch importe
# (verl/models/transformers/monkey_patch.py:23,29). Ce monkeypatch n'est appelé que si
# use_remove_padding=True ET ulysses_sequence_parallel_size>1 (agent_fsdp_workers.py:175),
# donc inactif avec les params du papier — mais on ne prend pas le risque.
#
# DEUX ENVIRONNEMENTS SÉPARÉS (comme sur notre VM)
#   1. entraînement : verl + vLLM 0.6.3 + torch 2.4 (+ client agentenv en --no-deps,
#      car ses dépendances déclarées tirent deepspeed et un pin trl qui casseraient tout)
#   2. serveur d'environnement TextCraft : fastapi/uvicorn/gymnasium seulement
#
# Usage :
#   bash setup/setup_verl_h100.sh                 # installe dans $HOME/envs/
#   ENV_ROOT=/opt/envs bash setup/setup_verl_h100.sh
#
# Après installation, voir la section "LANCEMENT" affichée en fin de script.
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-$HOME/envs}"
TRAIN_ENV="$ENV_ROOT/verl-h100"
SERVER_ENV="$ENV_ROOT/textcraft-server"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_DIR="$REPO_ROOT/external/AgentGym-RL"
CACHE="${PIP_CACHE:-/tmp/pip-cache}"

# --- Versions épinglées (ne pas modifier sans relire l'en-tête) ---------------
VLLM_VER="0.6.3"           # impose torch 2.4.0 / xformers 0.0.27.post2 / cu121
TRANSFORMERS_VER="4.47.1"  # dernière version avant la suppression de *FlashAttention2
TENSORDICT_VER="0.5.0"     # requirements verl : tensordict<0.6
FLASH_ATTN_VER="2.6.3"     # wheels précompilés pour torch 2.4 + cu12

echo "=============================================================="
echo "[verl-h100] Installation verl (papier AgentGym-RL) — 8x H100"
echo "[verl-h100] repo verl : $VERL_DIR"
echo "[verl-h100] envs      : $TRAIN_ENV  et  $SERVER_ENV"
echo "=============================================================="

# --- 0. Vérifications préalables ---------------------------------------------
[ -d "$VERL_DIR/verl" ] || { echo "ERREUR: $VERL_DIR/verl introuvable (repo mal cloné ?)"; exit 1; }

if command -v nvidia-smi > /dev/null; then
    echo "[verl-h100] GPU visibles :"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    N_GPU=$(nvidia-smi --list-gpus | wc -l)
    echo "[verl-h100] -> $N_GPU GPU détectés (le script du papier suppose trainer.n_gpus_per_node=8)"
    # cu121 exige un driver >= 530 ; les images H100 récentes sont en 535/550+.
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    [ "$DRV" -ge 530 ] || echo "!! ATTENTION driver $DRV < 530 : torch cu121 risque de ne pas démarrer"
else
    echo "!! nvidia-smi absent — installation possible, mais le run échouera sans GPU"
fi

# Python 3.10 ou 3.11 (vLLM 0.6.3 : cp38-cp312 ; wheels flash-attn : cp310/cp311).
BASE_PYTHON=""
for cand in python3.10 python3.11; do
    command -v "$cand" > /dev/null && BASE_PYTHON="$(command -v $cand)" && break
done
[ -n "$BASE_PYTHON" ] || { echo "ERREUR: ni python3.10 ni python3.11 trouvé (requis)"; exit 1; }
PY_TAG="cp$("$BASE_PYTHON" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
echo "[verl-h100] Python de base : $BASE_PYTHON ($PY_TAG)"

# --- 1. Environnement d'ENTRAÎNEMENT ------------------------------------------
echo
echo "[verl-h100] (1/4) venv d'entraînement : $TRAIN_ENV"
"$BASE_PYTHON" -m venv "$TRAIN_ENV"
PIP="$TRAIN_ENV/bin/pip"
"$PIP" install --upgrade pip wheel setuptools ninja -q --cache-dir "$CACHE"

# vLLM en PREMIER : il tire lui-même torch 2.4.0 (wheel PyPI = cu121), torchvision 0.19
# et xformers 0.0.27.post2 dans les versions binairement compatibles avec ses kernels.
echo "[verl-h100] (2/4) vLLM $VLLM_VER (+ torch 2.4.0 cu121, xformers, ray)..."
"$PIP" install "vllm==$VLLM_VER" --cache-dir "$CACHE"

echo "[verl-h100] Dépendances verl épinglées..."
"$PIP" install --cache-dir "$CACHE" \
    "transformers==$TRANSFORMERS_VER" \
    "tensordict==$TENSORDICT_VER" \
    "accelerate" "peft" "datasets" "wandb" \
    "codetiming" "hydra-core" "dill" "pandas" "pyarrow>=15.0.0" "pybind11" "pylatexenc"

# flash-attn : indispensable (attn_implementation=flash_attention_2 côté FSDP actor).
# Wheel précompilé si disponible — la compilation depuis les sources prend ~50 min
# (mesuré sur notre VM en juillet).
echo "[verl-h100] (3/4) flash-attn $FLASH_ATTN_VER..."
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/flash_attn-${FLASH_ATTN_VER}+cu123torch2.4cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
if "$PIP" install "$FA_URL" --cache-dir "$CACHE" 2>/dev/null; then
    echo "[verl-h100] flash-attn installé depuis le wheel précompilé"
else
    echo "[verl-h100] wheel indisponible -> compilation depuis les sources (~50 min)"
    MAX_JOBS="${MAX_JOBS:-8}" "$PIP" install "flash-attn==$FLASH_ATTN_VER" \
        --no-build-isolation --cache-dir "$CACHE"
    # Conserver le wheel pour les réinstallations (même logique que saves/wheels/ chez nous)
    mkdir -p "$REPO_ROOT/saves/wheels"
    "$PIP" wheel "flash-attn==$FLASH_ATTN_VER" --no-build-isolation --no-deps \
        -w "$REPO_ROOT/saves/wheels" --cache-dir "$CACHE" || true
fi

# verl lui-même : --no-deps, ses deps sont déjà posées ci-dessus dans les bonnes versions
# (son setup.py demanderait "transformers" et "vllm<=0.6.3" non épinglés).
echo "[verl-h100] verl en editable (--no-deps)..."
"$PIP" install -e "$VERL_DIR" --no-deps --cache-dir "$CACHE"

# Client TextCraft : verl importe agentenv.envs.TextCraftEnvClient
# (verl/utils/agentgym/client.py:4-20). --no-deps OBLIGATOIRE : les deps déclarées
# d'agentenv incluent deepspeed>0.15 et trl>=0.8.6, qui écraseraient torch/transformers.
echo "[verl-h100] client agentenv (--no-deps) + jsonlines..."
"$PIP" install -e "$REPO_ROOT/external/AgentGym/agentenv" --no-deps --cache-dir "$CACHE"
"$PIP" install jsonlines --cache-dir "$CACHE"

# --- 2. Environnement du SERVEUR TextCraft -----------------------------------
echo
echo "[verl-h100] (4/4) venv du serveur d'environnement : $SERVER_ENV"
"$BASE_PYTHON" -m venv "$SERVER_ENV"
"$SERVER_ENV/bin/pip" install --upgrade pip -q --cache-dir "$CACHE"
"$SERVER_ENV/bin/pip" install -e "$REPO_ROOT/external/AgentGym/agentenv-textcraft" --cache-dir "$CACHE"

# --- 3. Vérification ----------------------------------------------------------
echo
echo "[verl-h100] Vérification des imports critiques..."
"$TRAIN_ENV/bin/python" - << 'EOF'
import torch, transformers, tensordict, ray, flash_attn
import vllm
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
print(f"vllm={vllm.__version__} transformers={transformers.__version__} "
      f"tensordict={tensordict.__version__} ray={ray.__version__} flash_attn={flash_attn.__version__}")

# LE test qui compte : prouve qu'on emprunte bien le chemin vLLM VENDORISÉ de verl
# (et non le chemin SPMD des vLLM récents, non utilisé par le papier).
from verl.third_party.vllm import LLM, vllm_version
assert vllm_version == "0.6.3", f"chemin vendorisé INATTENDU : vllm_version={vllm_version}"
print(f"verl third_party vLLM path = {vllm_version}  <-- chemin du papier OK")

from agentenv.envs import TextCraftEnvClient
print("TextCraftEnvClient importé OK")

# Le rollout du papier importe aussi ceci au chargement du module :
from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout
print("vLLMRollout (boucle multi-tour du papier) importé OK")
EOF

"$SERVER_ENV/bin/python" -c "import agentenv_textcraft; print('serveur agentenv_textcraft importé OK')"

cat <<EOF

=============================================================
[verl-h100] Installation terminée.

LANCEMENT (deux terminaux)

  # Terminal 1 — serveur d'environnement (à lancer depuis le dossier du package :
  # le code utilise le chemin RELATIF agentenv_textcraft/recipes/)
  cd $REPO_ROOT/external/AgentGym/agentenv-textcraft
  $SERVER_ENV/bin/textcraft --host 127.0.0.1 --port 36005

  # Terminal 2 — entraînement (cwd = repo verl, cf. 'cd AgentGym-RL' de leur script)
  export PATH="$TRAIN_ENV/bin:\$PATH"
  cd $VERL_DIR
  # NE PAS exporter WANDB_BASE_URL (le script du papier pointe un proxy tiers)
  export WANDB_ENTITY=v-lagresle-criteo
  export WANDB_API_KEY=<clé>
  python3 -m verl.agent_trainer.main_ppo ...   # runbook à écrire séparément

RESTE À FAIRE avant le premier run (hors périmètre de ce script) :
  1. Données : leur script attend AgentItemId/textcraft_train.json, absent du repo.
     Notre data/train/textcraft_train.json a EXACTEMENT le bon schéma
     ([{"item_id": "textcraft_31"}, ...]) -> simple copie, aucune conversion.
  2. Modèle : leur script charge Qwen2.5-7B-Instruct ; notre cible de comparaison
     (les 75/100 du papier) est la ligne 3B -> télécharger Qwen2.5-3B-Instruct
     dans \$VERL_DIR/models/.
  3. Script de lancement : partir de leur textcraft_train.sh en retirant les
     placeholders (WANDB_BASE_URL, 'wandb login xxx', project_name=xxx) et en
     ajoutant trainer.n_gpus_per_node=8 trainer.nnodes=1 explicitement.
  4. Robustesse (2 lignes) : vllm_rollout.py:221 attrape TimeoutError alors que
     requests lève requests.exceptions.Timeout -> un timeout au reset tue le worker.
  5. Disque : checkpoints FSDP shardés poids+Adam, ~30 Go chacun pour un 3B, et
     remove_previous_ckpt_in_save=False par défaut -> prévoir ~400 Go ou activer
     la rétention.
=============================================================
EOF
