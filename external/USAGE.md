# Dépendances externes utilisées

## external/AgentGym — client HTTP TextCraft

- `agentenv/agentenv/envs/textcraft_env.py`  
  Import : `from agentenv.envs import TextCraftEnvClient`  
  Rôle : wrapper HTTP vers le serveur TextCraft (port 36005)

- `agentenv_textcraft/` — package du serveur TextCraft  
  Conda env : `agentenv-textcraft`  
  Lancement : `textcraft --host 127.0.0.1 --port 36005`

## external/AgentGym-RL — framework verl (training multi-GPU)

- `verl/agent_trainer/ppo/ray_trainer.py` — boucle PPO + ScalingInter
- `verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` — rollout multi-tour
- `verl/trainer/ppo/reward_score/` — fonctions de reward

## external/agentgym_rl_paper — scripts de référence du papier

- `train/AgentGym-RL/textcraft_train.sh` — config officielle (Qwen2.5-7B, 30 epochs)
- `train/AgentGym-RL/textcraft_train.4gpu.sh` — notre adaptation (Qwen2.5-3B, 4 GPU)
- `eval/textcraft_eval.*.sh` — scripts d'évaluation

## data/ — datasets

- `data/eval/textcraft_test.json` — 100 items de test
- `data/train/textcraft_train.json` — items d'entraînement
