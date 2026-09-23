#!/bin/bash
# Job 56 — exp43 : le contrôle G=16 sans curriculum (exp36, job 43z) À L'IDENTIQUE, SEUL delta :
# --kl-clamp 10 = borne haute de l'estimateur k3 par token dans la loss, comme verl (low_var_kl
# clampé à [-10, 10]) — TRL ne borne pas. Test du mécanisme du 15/09 (rejeu k3) : sans borne,
# un token éliminé par le sharpening (« Thought » redevenu rare, <|im_end|> forcé après
# troncature) pèse 1e7-1e19 et dicte le pas ; exp36 est mort à l'époque 10.
#   tient (> ép. 12, KL bornée, pas de boucles)  → la KL non bornée était la CAUSE de la divergence ;
#   boucle et meurt quand même                    → la KL n'était qu'un symptôme du collapse d'entropie.
# Recette : LoRA r8/α32, LR 3e-6, β 0.01, entropy-coef 0.001, G=16 × 4 tâches = 64 traj/pas,
# ancre mobile /4 ép. (372 pas), 30 tours, 512 tokens, 80 ép. Ckpt sur le HOME (règle 04/09).
# Décision Vadim 15/09/2026 (« oui je valide go »).
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp43_g16_klclamp10
bash setup/ensure_qwen_tmp.sh || { echo "[job56] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python setup/patch_trl_kl_clamp.py || { echo "[job56] patch k3 TRL ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job56.log 2>&1 \
  || { echo "[job56] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job56] $(date '+%F %T') — exp43 = exp36 (G=16, 4 tâches, ancre /4 ép.) + --kl-clamp 10, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --kl-clamp 10 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp43 — exp36 + borne k3 à 10 par token (verl) : la KL non bornée cause-t-elle la divergence ?"
