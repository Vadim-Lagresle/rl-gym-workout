#!/usr/bin/env bash
# ============================================================================
# textcraft_train.4gpu.sh
#
# Réplication la plus fidèle possible de la recette papier AgentGym-RL sur
# TextCraft, avec :
#   - Qwen2.5-3B-Instruct (au lieu de 7B dans le script upstream)
#   - 4× A100 40 GB (au lieu d'un cluster multi-noeud)
#   - rounds_ctrl.type=fixed (= pas de ScalingInter, baseline GRPO pure)
#
# Le but est de comparer l'output de verl tel quel à notre baseline TRL+LoRA
# en isolant l'effet "framework / config" du reste.
#
# Variables d'env utiles :
#   SMOKE=1                 → force total_training_steps=1, save_freq=-1
#                             pour valider le pipeline sans dépenser des heures.
#   TOTAL_STEPS=<int>       → override le nombre de steps (défaut : 50).
#                             Mettre 120 pour la recette papier complète (30 epochs ≈ 120 steps).
#   EXP_NAME=<str>          → nom de l'expérience (par défaut date + git sha court).
# ============================================================================
set -euo pipefail
set -x

# ----------------------------------------------------------------------------
# 1. Activation conda + variables vLLM
# ----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl

# ----------------------------------------------------------------------------
# 1.bis  NCCL workaround for GCP A100 VMs.
# The default GCP image sources /usr/local/gib/scripts/set_nccl_env.sh which
# sets NCCL_NET=gIB and a host of IB-specific tuner flags meant for multi-node
# A3-Ultra clusters with NVIDIA Collective Fabric. On a single-node 4× A100 box
# this plugin tries to load libibverbs (not installed), then NCCL crashes
# silently with SYSTEM_ERROR exit code 2 — symptom : worker dies right after
# "NCCL version 2.20.5+cuda12.4". Fix: force NCCL to use plain TCP socket
# transport (NVLink/P2P intra-node still works on top) and remove the gIB
# library path so the plugin can't be loaded at all.
# ----------------------------------------------------------------------------
unset NCCL_NET NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC \
      NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_IB_QPS_PER_CONNECTION NCCL_IB_ADAPTIVE_ROUTING \
      NCCL_NVLS_CHUNKSIZE NCCL_P2P_NET_CHUNKSIZE
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/usr/local/gib' | paste -sd: -)"
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----------------------------------------------------------------------------
# 2. Paramètres locaux
# ----------------------------------------------------------------------------
REPO_ROOT="/home/v.lagresle/rl-gym-workout"
cd "$REPO_ROOT/AgentGym-RL"

TASK_NAME="textcraft"
ENV_SERVER_URL="http://127.0.0.1:36005"
MODEL_PATH="$REPO_ROOT/models/Qwen2.5-3B-Instruct"
TRAIN_FILE="$REPO_ROOT/AgentEval/train/textcraft_train.json"
VAL_FILE="$REPO_ROOT/AgentEval/eval/textcraft_test.json"

GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo nogit)"
DEFAULT_EXP_NAME="agentgym_rl_qwen3b_4gpu_fixed_$(date +%Y%m%d_%H%M)_${GIT_SHA}"
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"
SAVE_ROOT="$REPO_ROOT/saves/agentgym_rl_4gpu"
RUN_DIR="$SAVE_ROOT/$EXP_NAME"
LOG_FILE="$RUN_DIR/run.log"
mkdir -p "$RUN_DIR"

# ----------------------------------------------------------------------------
# 3. Mode smoke vs run réel
# ----------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
if [[ "$SMOKE" == "1" ]]; then
    TOTAL_STEPS=1
    SAVE_FREQ=-1
    TEST_FREQ=-1
else
    SAVE_FREQ=25
    TEST_FREQ=-1   # On évalue à la fin séparément avec main_generation, plus rapide.
fi

# ----------------------------------------------------------------------------
# 4. Health-check serveur TextCraft
# ----------------------------------------------------------------------------
if ! curl -sSf -m 5 -X POST "$ENV_SERVER_URL/create" \
        -H 'content-type: application/json' -d '{}' >/dev/null; then
    echo "ERROR: TextCraft server unreachable at $ENV_SERVER_URL" >&2
    echo "Start it with: cd $REPO_ROOT/AgentGym/agentenv-textcraft &&" >&2
    echo "  conda activate agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 5. Lancement training
#
# Choix de config relatifs à la recette papier (textcraft_train.sh upstream) :
#   - rollout.n=8                   : identique paper (pas N=2 comme nos runs LoRA)
#   - train_batch_size=8            : down de 32 ; 4 GPUs au lieu du cluster
#   - max_response_length=8192      : down de 10240 (~20 % d'éco mémoire)
#   - max_tokens=512                : identique paper
#   - gpu_memory_utilization=0.55   : down de 0.7 → marge pour FSDP+activations
#                                     sur A100 40 GB
#   - tensor_model_parallel_size=1  : identique paper (4 data-parallel rollouts)
#   - kl_loss_coef=0.001            : identique paper (low_var_kl)
#   - lr=1e-6                       : identique paper
#   - rounds=30, rounds_ctrl=fixed  : identique paper (PAS de ScalingInter ici,
#                                     c'est explicitement le baseline GRPO pur)
# ----------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds=30 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_file="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=4096 \
  actor_rollout_ref.agentgym.task_name="$TASK_NAME" \
  actor_rollout_ref.agentgym.env_addr="$ENV_SERVER_URL" \
  actor_rollout_ref.agentgym.timeout=600 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_tokens=512 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.rollout_log_dir="$RUN_DIR/executer_logs" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.project_name=rl-gym-workout \
  trainer.experiment_name="$EXP_NAME" \
  trainer.logger='[console]' \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  2>&1 | tee "$LOG_FILE"
