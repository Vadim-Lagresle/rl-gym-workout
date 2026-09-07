#!/bin/bash
# exp25 — LoRA r8, ancre KL MOBILE (merge-and-restart toutes les 4 epochs).
# Design : runs/13_moving_anchor/exp25_r8_anchor4ep/config.yaml
# Hypothèse testée : le plafond 27-29 de l'arm β0.01 d'exp24 vient du rappel KL
# vers Qwen nu qui croît avec le progrès ; une ancre rapprochée doit le lever.
# vs exp24 : r8 (blog : petit rang suffit en RL), batch 64 ON-POLICY (pas de
# buffer 256 — caveat gros batch LoRA du blog), β=0.01 (validé), LR 3e-6 constant.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp25_r8_anchor4ep

# --- Étape 1 : smoke GPU (~5 min) — machinerie de ré-ancrage en conditions réelles.
# 8 items, 2 prompts/step → 4 steps/epoch ; frontière à 0.4 ep → ré-ancrage au step 2.
rm -rf saves/trl_grpo/smoke_anchor_anchors
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 2 --gradient-accumulation-steps 4 \
    --max-items 8 --max-steps 3 --max-completion-length 128 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 0.4 \
    --use-vllm-inprocess --run-name smoke_anchor > logs/smoke_anchor.log 2>&1 \
  || { echo "[job30] SMOKE ancre mobile ÉCHOUÉ — run réel ANNULÉ (logs/smoke_anchor.log)"; exit 1; }
grep -q "RÉ-ANCRAGE #1" logs/smoke_anchor.log \
  || { echo "[job30] smoke fini mais AUCUN ré-ancrage déclenché — run réel ANNULÉ"; exit 1; }
rm -rf saves/trl_grpo/smoke_anchor_anchors
echo "[job30] smoke ancre mobile OK — lancement du run réel"

# --- Étape 2 : run réel — 15 epochs (~705 steps), ré-ancrages aux epochs 4, 8, 12.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 15 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" > "logs/$RUN.log" 2>&1
