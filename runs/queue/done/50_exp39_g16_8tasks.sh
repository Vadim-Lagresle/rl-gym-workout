#!/bin/bash
# exp39 — DÉCONFONDRE G × tâches-par-pas (décision Vadim 03/09 soir). exp25 (8 tâches × 8 = 64
# traj/pas, stable 110 ép.) et exp36 (4 tâches × 16 = 64 traj/pas, collapse ép. 10) changent DEUX
# choses à la fois. Ici : recette exp36 à l'identique (LoRA r8/α32, LR 3e-6, β0.01, ancre mobile
# /4 ép., entropy-coef 0.001, G=16, 30 tours fixes, sans curriculum) avec SEUL delta
# --gradient-accumulation-steps 128 → 8 tâches × 16 = 128 traj par pas, 47 pas/époque (comme G=8).
# Question : le collapse d'exp36 vient-il de G=16 ou du passage de 8 à 4 tâches par pas ?
# Verdict attendu en 10-15 époques. 80 époques — arrêt manuel par Vadim.
# Éval tous les 47 pas = une fois par époque ici (47 pas/ép.) ; save-steps 94 = toutes les 2 ép.
# Points de contrôle périodiques (adapter + optimizer, ~180 Mo, 1 conservé) sur le HOME (saves/trl_grpo_ckpt),
# plus sur /tmp : disque du nœud plein le 04/09 14:44 (OSError 28) = 2h30 perdues. Règle : ckpt sur le disque.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp39_g16_8tasks
bash setup/ensure_qwen_tmp.sh || { echo "[job50] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job50.log 2>&1 \
  || { echo "[job50] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job50] $(date '+%F %T') — exp39 G=16, 8 tâches × 16 = 128 traj/pas, sans curriculum, ancre /4, 80 époques ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 128 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp39 — G=16 à 8 tâches × 16 traj par pas (déconfondre G et tâches/pas), sans curriculum"
