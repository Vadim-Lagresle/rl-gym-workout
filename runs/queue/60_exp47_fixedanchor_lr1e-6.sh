#!/usr/bin/env bash
# job60 — exp47 : expérience E (décision Vadim 22-23/09, en dernier, courte).
# Recette exp30 à l'identique (LoRA r8/α32, référence FIXE, β0.01, entropy-coef 0.001, G=8, 8 tâches/pas,
# 30 tours, 512 tokens), UN SEUL delta : --learning-rate 3e-6 → 1e-6 (le LR du full-FT).
# Question : la dérive KL géométrique de la LoRA à référence fixe vient-elle du LR effectif ?
# Lecture : KL médiane par époque. exp30 franchissait 0.05-0.1 vers l'époque 7-8 puis divergeait ;
# si la dérive est linéaire ici (comme le full-FT, ~1e-3/époque), le LR suffit à l'expliquer ;
# si elle reste géométrique mais retardée ~×3, c'est la dynamique du produit B·A.
# 30 époques : de quoi voir le franchissement ou son absence à trois fois l'horizon d'exp30.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp47_fixedanchor_lr1e-6
bash setup/ensure_qwen_tmp.sh || { echo "[job60] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job60.log 2>&1 \
  || { echo "[job60] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job60] $(date '+%F %T') — exp47 = exp30 (référence FIXE, G=8) avec LR 1e-6, 30 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 1e-6 --beta 0.01 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 30 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp47 — expérience E : LoRA r8 à référence fixe, LR 1e-6 (exp30 + un seul delta) : la dérive KL géométrique vient-elle du LR ?"
