#!/bin/bash
# exp38 — OPTION A (décision Vadim 03/09) : la mécanique de mise à jour du PAPIER
# (collecte 256 trajectoires → 4 pas d'optimisation clippés eps 0.2, TRL
# `steps_per_generation`, celle d'exp23/exp24) appliquée à la recette LoRA + ancre
# mobile de la lignée curriculum. SEUL delta vs exp36 (contrôle N=16 sans
# curriculum, collapse ep 10-12) : --steps-per-generation 256. Tout le reste est la
# recette exp25/32/36 à l'identique : r8/α32, LR 3e-6 constant, β 0.01, ancre mobile
# /4 ep, entropy-coef 0.001, N=16 (16 prompts × 16 = 256 traj par collecte), 64 traj
# par pas d'optimisation (grad_accum 64 → 4 pas par collecte, pas 1 on-policy ratio≡1,
# pas 2-4 hors-politique clippés), 512 tok/tour, 30 tours fixes, 80 epochs.
# Même nombre de pas par epoch qu'exp36 (~93) et même nombre de trajectoires par
# epoch → comparaison à compute égal. Question : le clipping (région de confiance
# locale) et la collecte 4× plus large ralentissent-ils le sharpening qui a tué
# exp36 et exp34 ? Signaux nouveaux à lire : clip_ratio/* (≠ 0 dès le 2e pas de
# chaque collecte), KL, entropie.
# Le ré-ancrage peut tomber au milieu d'une collecte : la fusion est l'identité sur
# la politique (adapter fusionné dans la base, adapter neuf à zéro), donc les
# old_logprobs restent valides ; seul l'ancre KL bouge. Vérifié par le smoke ci-dessous.
# 80 epochs — arrêt manuel par Vadim.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp38_buffer256_n16
bash setup/ensure_qwen_tmp.sh || { echo "[job49] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

# Pré-vol 1 (CPU) : selftest schedules (ancre mobile).
python src/train/schedules.py > logs/selftest_schedules_job49.log 2>&1 \
  || { echo "[job49] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }

# Pré-vol 2 (GPU, ~5 min) : smoke LoRA + buffer (2 pas par collecte) + ré-ancrage
# toutes les 0.4 epoch (= tous les ~1.6 pas → un ré-ancrage tombe AU MILIEU d'une
# collecte) + sync vLLM après fusion. 4 pas = 2 collectes, 2 ré-ancrages.
rm -rf saves/trl_grpo/smoke_buffer_anchor_anchors
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 2 --gradient-accumulation-steps 4 --steps-per-generation 8 \
    --max-items 8 --max-steps 4 --max-completion-length 128 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 0.4 \
    --use-vllm-inprocess --run-name smoke_buffer_anchor > logs/smoke_buffer_anchor.log 2>&1 \
  || { echo "[job49] SMOKE buffer+ancre ÉCHOUÉ — run réel ANNULÉ (logs/smoke_buffer_anchor.log)"; exit 1; }
grep -q "RÉ-ANCRAGE #1" logs/smoke_buffer_anchor.log \
  || { echo "[job49] SMOKE : aucun ré-ancrage observé — run réel ANNULÉ"; exit 1; }

echo "=== [job49] $(date '+%F %T') — exp38 mécanique papier (buffer 256, 4 pas clippés) sur recette LoRA+ancre mobile, N=16 sans curriculum, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --steps-per-generation 256 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp38 — mécanique papier (buffer 256, 4 pas clippés) × LoRA + ancre mobile, N=16 sans curriculum"
