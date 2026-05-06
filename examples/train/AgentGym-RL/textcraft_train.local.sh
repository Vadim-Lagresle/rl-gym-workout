#!/usr/bin/env bash
set -euo pipefail
set -x

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=disabled

REPO_ROOT="/home/v.lagresle/rl-gym-workout"
cd "$REPO_ROOT/AgentGym-RL"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl

TASK_NAME="textcraft"
ENV_SERVER_URL="http://127.0.0.1:36005"
MODEL_PATH="$REPO_ROOT/models/Qwen2.5-3B-Instruct"
TRAIN_FILE="$REPO_ROOT/AgentEval/train/textcraft_train.json"
VAL_FILE="$REPO_ROOT/AgentEval/eval/textcraft_test.json"

SAVE_ROOT="$REPO_ROOT/saves/textcraft_smoke"
EXP_NAME="smoke_grpo_step1"
RUN_DIR="$SAVE_ROOT/$EXP_NAME"
LOG_FILE="$RUN_DIR/run.log"
mkdir -p "$RUN_DIR"

# Health-check env server before training
curl -sS -m 8 -X POST "$ENV_SERVER_URL/create" -H 'content-type: application/json' -d '{}' >/dev/null

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m verl.agent_trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds=20 \
  data.train_file="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.agentgym.task_name="$TASK_NAME" \
  actor_rollout_ref.agentgym.env_addr="$ENV_SERVER_URL" \
  actor_rollout_ref.agentgym.timeout=600 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_tokens=512 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.rollout_log_dir="$RUN_DIR/executer_logs" \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.project_name=rl-gym-workout \
  trainer.experiment_name="$EXP_NAME" \
  trainer.logger='[console]' \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  2>&1 | tee "$LOG_FILE"
