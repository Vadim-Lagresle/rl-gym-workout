#!/bin/bash
# Chaîne : eval exp7 → smoke test rollout parallèle → lancement exp7.1
# Lancé avec : nohup bash src/train/run_exp7.1_chain.sh > logs/run_exp7.1_chain.log 2>&1 &
# Si une étape plante (exit code != 0), le script s'arrête (set -e).
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/envs/agentgym-rl/bin/python"

echo "========================================"
echo "[chain] START $(date -u)"
echo "========================================"

# ── Étape 1 : Eval checkpoint-564 ────────────────────────────────────────────
echo ""
echo "[chain] STEP 1/3 — Eval exp7 checkpoint-564 ($(date -u))"
"$PYTHON" "$REPO_ROOT/src/eval/eval_fullft.py" \
    --checkpoint "$REPO_ROOT/saves/trl_grpo/exp7_b200_fullft/checkpoint-564" \
    --run-name exp7_b200_fullft
echo "[chain] STEP 1/3 — Eval terminée ($(date -u))"

# ── Étape 2 : Smoke test rollout HTTP parallèle ───────────────────────────────
echo ""
echo "[chain] STEP 2/3 — Smoke test exp7.1 (2 steps, 8 items) ($(date -u))"
"$PYTHON" "$REPO_ROOT/src/train/train_grpo.py" \
    --full-ft \
    --num-generations 8 \
    --max-completion-length 512 \
    --max-items 8 \
    --max-steps 2 \
    --run-name exp7.1_smoke
echo "[chain] STEP 2/3 — Smoke test OK ($(date -u))"

# ── Étape 3 : Lancement exp7.1 ────────────────────────────────────────────────
echo ""
echo "[chain] STEP 3/3 — Lancement exp7.1 (max_steps=4488, resume checkpoint-564) ($(date -u))"
"$PYTHON" "$REPO_ROOT/src/train/train_grpo.py" \
    --full-ft \
    --num-generations 8 \
    --max-completion-length 512 \
    --max-items 0 \
    --max-steps 4488 \
    --resume-from-checkpoint "$REPO_ROOT/saves/trl_grpo/exp7.1_b200_fullft_12ep/checkpoint-1269" \
    --run-name exp7.1_b200_fullft_12ep

echo ""
echo "========================================"
echo "[chain] ALL DONE $(date -u)"
echo "========================================"
