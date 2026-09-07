#!/bin/bash
# exp25.1 — bench d'optimisation GPU (~1 h) : 5 configs × 8 steps réels, 64 traj/step
# constant. Mesure s/it, KV cache, préemptions. Design : runs/13_moving_anchor/
# exp25.1_gpu_throughput/README.md. SA CONCLUSION FIXE LES FLAGS DES JOBS SUIVANTS.
cd "$(dirname "$0")/../.."
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"
OUT=runs/13_moving_anchor/exp25.1_gpu_throughput/results.md

bash setup/ensure_qwen_tmp.sh || { echo "[job40b] téléchargement Qwen3B ÉCHOUÉ"; exit 1; }

run_cfg() {  # <nom> <gpu-util> <num-generations>
  local name=$1 util=$2 ngen=$3
  local log=logs/bench_$name.log
  echo "[job40b] $(date '+%T') config $name : gpu-util=$util, N=$ngen (8 steps)"
  WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 45m \
  python src/train/train_grpo.py \
      --lora-r 8 --lora-alpha 32 \
      --num-generations "$ngen" --gradient-accumulation-steps 64 \
      --entropy-coef 0.001 --learning-rate 3e-6 --beta 0.01 \
      --max-completion-length 512 --max-items 0 --max-steps 8 \
      --max-rounds-schedule '30:0' --vllm-max-len 32768 \
      --vllm-gpu-util "$util" --eval-every 999 \
      --output-root /tmp/trl_grpo_runs --use-vllm-inprocess \
      --run-name "bench_$name" > "$log" 2>&1
  local rc=$? sit kv pre
  # s/it moyen sur les 4 derniers steps (les premiers incluent le warmup moteur)
  sit=$(grep -oE '[0-9]+\.[0-9]+s/it' "$log" | tail -4 | tr -d 'sit/' \
        | awk '{s+=$1; n++} END {if (n) printf "%.1f", s/n; else print "NA"}')
  kv=$(grep -ioE "GPU KV cache size:? [^,]*" "$log" | tail -1 | grep -oE "[0-9,.]+ ?(GiB|tokens)")
  pre=$(grep -ci "preempt" "$log")
  echo "| $name | $util | $ngen | $sit | ${kv:-?} | $pre | rc=$rc |" >> "$OUT"
}

{ echo "# exp25.1 — résultats bench GPU ($(date '+%F %T'))"
  echo
  echo "8 steps/config, 64 traj/step constant, s/it moyenné sur les 4 derniers steps."
  echo
  echo "| config | gpu-util | N | s/it | KV cache | préemptions | statut |"
  echo "|---|---|---|---|---|---|---|"; } > "$OUT"

run_cfg A_ref    0.17 8
run_cfg B_util04 0.40 8
run_cfg C_util06 0.60 8
run_cfg D_n16u05 0.50 16
run_cfg E_n16u06 0.60 16

rm -rf /tmp/trl_grpo_runs/bench_*
echo "[job40b] terminé — résultats : $OUT"
cat "$OUT"
