#!/bin/bash
# exp36.1 — contrôle N=16 sans curriculum, 2e essai (ajouté 27/08, décision Vadim) :
# exp36 a collapsé (entropie 1.0 → 0.10 aux ep 8-12, puis KL runaway 10⁸-10¹² et
# générations détruites — diagnostic : N=16 accélère le sharpening ~2× et l'ancre
# ré-basée toutes les 4 ep SUIT la politique en cours de déterminisation au lieu
# de la freiner ; cf. session hebdo 27/08). SEUL delta vs exp36 :
# --moving-anchor-every-epochs 4 → 12. Rationale : une ancre plus ancienne garde
# une référence ENTROPIQUE sur toute la fenêtre critique (ep 8-12 chez exp36) —
# la KL redevient un frein à la déterminisation ; 12 laisse encore ~6 ré-ancrages
# sur 80 ep (mobilité préservée, pas un retour à l'ancre fixe d'exp24/30).
# β et entropy-coef INCHANGÉS (un seul delta, décision Vadim vs l'option
# entropy-coef 0.01) — plus honnête en comparaison : β fixe, on règle le curseur
# d'un mécanisme déjà en place plutôt que d'ajouter une force nouvelle. Rôle : contrôle « N=16 sans curriculum » qui survive, pour
# départager proprement N vs curriculum (exp32=82 vs exp25=73 vs exp36=54†collapse).
# 80 epochs — arrêt manuel.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp36.1_anchor12
bash setup/ensure_qwen_tmp.sh || { echo "[job48] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python src/train/schedules.py > logs/selftest_schedules_job48.log 2>&1 \
  || { echo "[job48] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
echo "=== [job48] $(date '+%F %T') — exp36.1 N=16 sans curriculum, ancre /12 ep, 80 epochs ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --moving-anchor-every-epochs 12 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp36.1 — contrôle N=16 sans curriculum, ancre mobile /12 ep"
