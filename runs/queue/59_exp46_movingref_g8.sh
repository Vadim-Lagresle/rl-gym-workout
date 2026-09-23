#!/usr/bin/env bash
# job59 — exp46 : « vraie LoRA à référence mobile » (expérience C, décision Vadim 23/09).
# Recette exp25 à l'identique (LoRA r8/α32, LR 3e-6, β0.01, entropy-coef 0.001, G=8, 8 tâches/pas,
# 30 tours, 512 tokens, ré-ancrage toutes les 4 ép.), UN SEUL delta : --moving-anchor-mode ref.
#   merge (exp25→43) : fusion + adaptateur neuf + purge Adam à chaque cycle  (= ReLoRA)
#   ref   (exp46)    : adaptateur vivant jamais touché ; adaptateur figé 'ref' recopié depuis le vivant
# Question : la référence mobile suffit-elle, ou c'est le reset de B·A / d'Adam qui stabilisait ?
# Lecture : KL intra-fenêtre (dents de scie à 1e-3 = référence active ; dérive géométrique = ReLoRA).
# Garde-fous : selftest CPU (7 invariants) puis smoke GPU 3 pas avec 1 ré-ancrage, sinon run annulé.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp46_movingref_g8
bash setup/ensure_qwen_tmp.sh || { echo "[job59] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python -m src.tests.test_schedules > logs/selftest_schedules_job59.log 2>&1 \
  || { echo "[job59] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
CUDA_VISIBLE_DEVICES= python -m src.tests.test_moving_ref > logs/selftest_moving_ref_job59.log 2>&1 \
  || { echo "[job59] selftest moving-ref (CPU) ÉCHOUÉ — run ANNULÉ (logs/selftest_moving_ref_job59.log)"; exit 1; }
rm -rf saves/trl_grpo/smoke_movingref_anchors
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 2 --gradient-accumulation-steps 4 \
    --max-items 8 --max-steps 3 --max-completion-length 128 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 0.4 --moving-anchor-mode ref \
    --vllm-gpu-util 0.5 \
    --use-vllm-inprocess --run-name smoke_movingref > logs/smoke_movingref.log 2>&1 \
  || { echo "[job59] SMOKE moving-ref ÉCHOUÉ — run réel ANNULÉ (logs/smoke_movingref.log)"; exit 1; }
grep -q "\[moving-ref\] >>> RÉ-ANCRAGE #1" logs/smoke_movingref.log \
  || { echo "[job59] smoke fini mais AUCUNE recopie default→ref — run réel ANNULÉ"; exit 1; }
grep -q "adaptateur de référence 'ref' ajouté" logs/smoke_movingref.log \
  || { echo "[job59] smoke : adaptateur 'ref' non attaché — run réel ANNULÉ"; exit 1; }
rm -rf saves/trl_grpo/smoke_movingref_anchors
echo "[job59] smoke moving-ref OK — lancement du run réel"
echo "=== [job59] $(date '+%F %T') — exp46 = exp25 + --moving-anchor-mode ref (référence = adaptateur figé recopié /4 ép., pas de merge-and-restart), 80 ep ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 4 --moving-anchor-mode ref \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp46 — vraie LoRA à référence mobile (adaptateur 'ref' figé recopié /4 ép., un seul adaptateur vivant r8), recette exp25 sinon : la référence mobile suffit-elle sans merge-and-restart ?"
