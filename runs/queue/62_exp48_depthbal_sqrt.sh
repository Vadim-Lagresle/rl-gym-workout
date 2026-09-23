#!/usr/bin/env bash
# job62 — exp48_depthbal_sqrt : expérience F (décision Vadim 23/09), variante 'sqrt'.
# Recette exp43 à l'identique (LoRA r8/α32, LR 3e-6, β0.01, --kl-clamp 10, ancre mobile /4 ép., G=16,
# 4 tâches/pas, 30 tours, 512 tokens, sans curriculum), DEUX changements côté données seulement :
#   --train-file data/train/textcraft_train_plus_reservoir.json : train + réservoir few-shot = 444 tâches
#       (d1 109, d2 239, d3 88, d4 8 au lieu de 1) ; test inchangé (100 tâches, dont 3 d4).
#   --depth-balance sqrt : uniform = 1/4 des tirages par profondeur ; sqrt = masse ∝ √n_d (d4 ≈ 7 %).
# Question : le mur d4 (0 % dans tous les runs) cède-t-il avec 8 recettes d4 bien échantillonnées ?
# Lecture : pass@1 test par profondeur (d4 sur 3 tâches, d3 sur 25), à comparer à exp43.
set -u
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
RUN=exp48_depthbal_sqrt
bash setup/ensure_qwen_tmp.sh || { echo "[job62] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }
python setup/patch_trl_kl_clamp.py || { echo "[job62] patch k3 TRL ÉCHOUÉ — run ANNULÉ"; exit 1; }
python -m src.tests.test_schedules > logs/selftest_schedules_job62.log 2>&1 \
  || { echo "[job62] selftest schedules ÉCHOUÉ — run ANNULÉ"; exit 1; }
[ -f data/train/textcraft_train_plus_reservoir.json ] && [ -f data/train/textcraft_train_plus_reservoir_with_depth.json ] \
  || { echo "[job62] fichiers de train étendus absents — run ANNULÉ"; exit 1; }
echo "=== [job62] $(date '+%F %T') — exp48_depthbal_sqrt = exp43 + train+réservoir (444, d4=8) + --depth-balance sqrt, 80 ep ===" >> "logs/$RUN.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 \
    --num-generations 16 --gradient-accumulation-steps 64 \
    --entropy-coef 0.001 \
    --learning-rate 3e-6 --beta 0.01 \
    --kl-clamp 10 \
    --moving-anchor-every-epochs 4 \
    --train-file data/train/textcraft_train_plus_reservoir.json --depth-balance sqrt \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 80 --max-rounds-schedule '30:0' \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 \
    --eval-every 47 --eval-items 100 \
    --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 \
    --output-root saves/trl_grpo_ckpt --use-vllm-inprocess \
    --run-name "$RUN" >> "logs/$RUN.log" 2>&1
bash scripts/append_results.sh "$RUN" "exp48_depthbal_sqrt — F : train + réservoir few-shot (d4 1→8), échantillonnage sqrt par profondeur, recette exp43 (G=16 + borne k3)"
