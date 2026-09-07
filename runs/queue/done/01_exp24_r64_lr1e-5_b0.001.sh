#!/bin/bash
# exp24 — grille LoRA (design : runs/12_lora_grid/exp24_grid/README.md)
# r=64, LR=1e-5, beta=0.001 — recette exp23 par ailleurs identique
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp24_r64_lr1e-5_b0.001
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 64 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --steps-per-generation 256 --entropy-coef 0.001 \
    --learning-rate 1e-5 --beta 0.001 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 15 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --eval-every 47 --eval-items 100 \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" > "logs/$RUN.log" 2>&1
