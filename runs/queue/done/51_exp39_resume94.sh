#!/bin/bash
# exp39 — REPRISE EXACTE depuis le checkpoint-94 (home) après l'arrêt du workspace Coder du 04/09
# 19:45 (run tué au pas 120, aucune erreur). Décision Vadim 06/09 : reprendre du ckpt plutôt que de zéro.
# Reprise TRL native (--resume-from-checkpoint) : adapter + optimizer + scheduler + RNG + compteurs
# repris au pas 94 (époque 2,04). Aucun ré-ancrage n'avait eu lieu (1er à l'époque 4 = pas 188) →
# base = Qwen nu, --moving-anchor-initial-cycle 0 (défaut). --best-init-score 0.17 = best déjà sauvé.
# Recette inchangée : LoRA r8/α32, LR 3e-6, β0.01, ancre /4, G=16, 8 tâches × 16 = 128 traj/pas,
# 30 tours, sans curriculum, 80 ép. (arrêt manuel). Ckpt périodiques sur le HOME.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp39_g16_8tasks
CKPT=saves/trl_grpo_ckpt/$RUN/checkpoint-94
[ -f "$CKPT/optimizer.pt" ] && [ -f "$CKPT/adapter_model.safetensors" ] \
  || { echo "[job51] checkpoint-94 incomplet ($CKPT) — run ANNULÉ"; exit 1; }
bash setup/ensure_qwen_tmp.sh || { echo "[job51] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job51.log 2>&1 \
  || { echo "[job51] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job51] $(date '+%F %T') — exp39 REPRISE depuis $CKPT (pas 94, époque 2,04, optimizer inclus) ===" >> "logs/$RUN.log"
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
    --save-best-optimizer --best-init-score 0.17 \
    --save-steps 94 --save-total-limit 1 \
    --resume-from-checkpoint "$CKPT" \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp39 — G=16 à 8 tâches × 16 traj/pas, sans curriculum (reprise ckpt-94 le 06/09)"
