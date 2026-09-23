#!/bin/bash
# Job 56c — exp45 : expérience D (décision Vadim 22/09). exp30 À L'IDENTIQUE (LoRA r8/α32, LR 3e-6, β0,01,
# G=8 × 8 tâches, RÉFÉRENCE KL FIXE = Qwen nu, aucun --moving-anchor-*), SEUL delta : --kl-clamp 10.
# Question : la borne k3 remplace-t-elle l'ancre mobile ? exp30 est mort à l'ép. 11 (KL exponentielle,
# 0,05 franchi à l'ép. ~6,5). Prédiction théorique : non — au-delà de ρ ≈ 2,4 la borne met le gradient KL à
# zéro, la régularisation vaut β=0 pour la majorité des tokens, la politique dérive sans frein.
#   tient et progresse → la borne suffit, l'ancre mobile n'est pas nécessaire à G=8 ;
#   dérive / plafonne / collapse → l'ancre est nécessaire, la borne ne traite que la queue.
# 20 époques suffisent (exp30 : verdict à l'ép. 11). Ckpt sur le HOME.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp45_fixedanchor_klclamp10
bash setup/ensure_qwen_tmp.sh || { echo "[job56c] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python setup/patch_trl_kl_clamp.py || { echo "[job56c] patch k3 TRL ÉCHOUÉ — run ANNULÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job56c.log 2>&1 \
  || { echo "[job56c] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job56c] $(date '+%F %T') — exp45 = exp30 (référence FIXE, G=8) + --kl-clamp 10, 20 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --kl-clamp 10 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 20 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp45 — référence fixe + borne k3 (exp30 + un seul delta) : la borne remplace-t-elle l'ancre ?"
