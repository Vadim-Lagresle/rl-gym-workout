# rl-gym-workout — training LLM agents by multi-turn RL on TextCraft

Research internship, Criteo AI Lab, April–September 2026 (Vadim Lagresle).

**The question.** Once an interactive environment is given, how does a language model improve
best on it by reinforcement learning: what makes training stable, and what makes it efficient?
We study it on **TextCraft**, a text version of Minecraft crafting from the
[AgentGym-RL](https://arxiv.org/abs/2509.08755) benchmark: the agent must obtain a target item by
issuing `get` and `craft` commands over several turns, and gets a reward of 1 if it succeeds,
0 otherwise. The model is **Qwen2.5-3B-Instruct**, the algorithm is **GRPO** (TRL + vLLM), on a
single B200 GPU.

**Main results** (pass@1 on the 100 test tasks, best / plateau):

| Configuration | pass@1 |
|---|---|
| Qwen2.5-3B, no training | 10.3 |
| Paper recipe, full fine-tuning | 69 / 61 |
| LoRA + moving KL anchor (ReLoRA), no curriculum | 73 / 70 |
| **Horizon curriculum** (10 → 20 → 30 turns) | **82 / 78** |
| No curriculum, G=16, KL estimator bounded | 82 / 76 |
| AgentGym-RL-3B (published, multi-GPU) | 75 |

The full registry of runs is in [`runs/INDEX.md`](runs/INDEX.md).

---

## 1. How it works, in one page

One training update goes like this:

1. **Draw tasks.** 4 recipes (or 8) are drawn from the 374 training recipes, uniformly or with
   depth weights (`src/train/sampling.py`).
2. **Play episodes.** For each recipe, the model plays G = 16 (or 8) episodes. At each turn it
   writes a thought and one action; the TextCraft server answers with an observation; this stops
   when the item is crafted or the turn cap is reached (`src/train/rollout.py`). Generation runs
   on a vLLM engine that TRL hosts in the training process (`src/train/vllm_engine.py`).
3. **Score.** Each episode gets 1 or 0. GRPO compares each episode to the others of its group:
   better than the group mean means a positive advantage.
4. **Update.** TRL computes the policy-gradient loss on the tokens the model wrote (the
   environment's text is kept in the sequence but masked), plus a KL penalty to a reference
   model, and updates a LoRA adapter of rank 8.
5. **Around the update.** Every few epochs the KL reference is moved to the current policy
   (`src/train/kl_anchor.py`); the turn cap or token budget can grow with the epochs
   (`src/train/horizon_schedules.py`); every 47 updates the model plays the 100 test tasks and the
   best one is saved (`src/train/periodic_eval.py`).

Three lessons that the code encodes, each explained in the report:

- the environment's observations must stay in the training sequence but be **masked**, or the
  model silently learns nothing (`rollout.py`);
- LoRA with a **fixed** KL reference drifts away geometrically and collapses; a **moving**
  reference fixes it (`kl_anchor.py`);
- TRL's per-token KL estimator k3 is **unbounded**, unlike the paper's code; a single token can
  then destroy the policy at G=16 (`--kl-clamp 10`, `setup/patch_trl_kl_clamp.py`).

---

## 2. Setup

The machine is a Coder pod with one B200. `/tmp` is wiped whenever the pod is recreated; the home
disk (35 GB) is persistent. The training environment and the base model therefore live on `/tmp`
and are rebuilt by script; checkpoints, best models and logs live on the home.

```bash
# Training environment (TRL >= 1.9, vLLM >= 0.25, flash-attention), about 10 minutes
bash setup/setup_agentgym_rl_v2.sh              # -> /tmp/envs/agentgym-rl-v2, applies the KL patch
bash setup/ensure_qwen_tmp.sh                   # -> /tmp/models/Qwen2.5-3B-Instruct
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"

# TextCraft server (its own virtualenv, requirements in setup/requirements-agentenv-textcraft.txt).
# It must be started from the package folder (it reads recipes/ with a relative path).
source ~/envs/agentenv-textcraft/bin/activate
cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005
```

`external/` holds third-party code (AgentGym, the AgentGym-RL paper code) and is not in the public
repository; clone [AgentGym](https://github.com/WooooDyy/AgentGym) to get the TextCraft server.

---

## 3. Running things

**A training run** (the Horizon curriculum of the report). Always detach with `setsid nohup`:
closing the IDE session kills the process group otherwise.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python src/train/train_grpo.py \
    --lora-r 8 --lora-alpha 32 --num-generations 16 --gradient-accumulation-steps 64 \
    --learning-rate 3e-6 --beta 0.01 --entropy-coef 0.001 --kl-clamp 10 \
    --moving-anchor-every-epochs 4 \
    --max-rounds-schedule-epochs '10:0,20:15,30:30' \
    --max-completion-length 512 --max-items 0 --num-epochs 80 \
    --vllm-max-len 32768 --vllm-gpu-util 0.5 --use-vllm-inprocess \
    --eval-every 47 --eval-items 100 --save-best-optimizer \
    --save-steps 94 --save-total-limit 1 --output-root saves/trl_grpo_ckpt \
    --run-name my_run > logs/my_run.log 2>&1 &
```

`python src/train/train_grpo.py --help` lists every option; `src/train/cli.py` explains each one.
A 3-update smoke test on the GPU: `bash src/tests/smoke_train.sh`.

**The run queue.** Long experiments are shell scripts in `runs/queue/`, run one after the other
by a single runner; finished jobs move to `runs/queue/done/`. Each job checks its prerequisites
and runs the CPU tests before touching the GPU.

```bash
setsid nohup bash scripts/run_queue.sh >> logs/queue.log 2>&1 &
```

**An evaluation** of a saved model on the test set:

```bash
bash src/utils/start_vllm_server.sh <full_model_dir>                      # port 8001
python src/eval/eval_textcraft.py --model <full_model_dir> --run-name <family>/<name>
kill $(cat /tmp/vllm_server.pid)
```

A LoRA best is saved as an adapter only: merge it first with `src/utils/merge_lora.py`, or, for a
moving-anchor run, rebuild the full model with `src/utils/merge_anchor_chain.py`.
`src/eval/eval_oracle.py` computes pass@k; `src/analysis/analyze_eval.py` breaks an evaluation
down by depth and error class.

---

## 4. Map of the code

```
src/
├── train/                       training (entry point: train_grpo.py)
│   ├── train_grpo.py            reads the options, assembles dataset, config, callbacks, trainer; trains
│   ├── cli.py                   every command-line option, with its meaning
│   ├── data.py                  training tasks and their prompts; shared paths
│   ├── rollout.py               the episode loop model <-> TextCraft, masking, reward
│   ├── vllm_engine.py           the vLLM engine hosted by TRL: generation, weight sync, model export
│   ├── grpo_config.py           the TRL GRPO configuration and the LoRA configuration
│   ├── callbacks.py             which callbacks a run gets; checks incompatible options
│   ├── kl_anchor.py             moving KL reference: merge-and-restart (ReLoRA) or frozen copy
│   ├── horizon_schedules.py     horizon (turns) and token-budget curricula
│   ├── sampling.py              depth curriculum and depth rebalancing of the task draw
│   ├── periodic_eval.py         test evaluation every N updates; saves the best model
│   ├── diagnostics.py           GPU memory telemetry
│   └── lr_schedules.py, lr_adaptive.py   learning-rate schedules of exp20-exp23 (legacy)
├── eval/                        evaluation (entry point: eval_textcraft.py)
│   ├── eval_textcraft.py        pass@1 of a model on the test set (vLLM server or HuggingFace)
│   ├── eval_oracle.py           pass@k with k independent tries per task
│   ├── textcraft_common.py      episode loop, logs, test set, pass@k estimator (shared)
│   ├── llm_chat.py              one chat interface over the two generation backends
│   └── error_taxonomy.py        the six error classes of TextCraft messages
├── analysis/
│   ├── analyze_eval.py          per-episode, per-depth and per-error analysis of an evaluation
│   ├── plot_oracle_passk.py     pass@k figures
│   └── replay_k3_tokens.py      token-by-token replay of the KL estimator on a checkpoint
├── utils/
│   ├── merge_lora.py            adapter + base -> full model
│   ├── merge_anchor_chain.py    rebuild a full model from a moving-anchor run
│   ├── label_depths.py          depth (1-4) of every TextCraft task
│   ├── build_fewshot_examples.py  solved examples from recipes outside train and test
│   ├── vllm_supports.py         can the installed vLLM serve this model?
│   └── start_vllm_server.sh     OpenAI-compatible vLLM server for evaluation
└── tests/                       run these before any GPU job
    ├── test_schedules.py        LR/beta schedules, horizon/budget schedules, merge-and-restart anchor
    ├── test_moving_ref.py       frozen-copy anchor (mode 'ref')
    ├── test_sampling.py         depth curriculum, depth rebalancing, sampler shape
    ├── check_queue_jobs.py      every queued job passes valid options
    └── smoke_train.sh           3 GPU updates with one re-anchoring and one evaluation
```

**How the training files call each other.** `train_grpo.py` → `cli.py` (options) → `data.py`
(tasks) → `grpo_config.py` (TRL config) → `callbacks.py` (which builds `periodic_eval`,
`kl_anchor`, `lr_schedules`, `diagnostics`) → TRL's `GRPOTrainer`, which calls
`rollout.grpo_rollout_func` at every update; `rollout.py` reads the current caps from
`horizon_schedules.py` and generates through `vllm_engine.py`; `sampling.py` replaces the trainer's
task sampler when a depth option is set. `periodic_eval.py` reuses the evaluation loop of
`eval/textcraft_common.py` so that training-time and offline scores are comparable.

**Other folders.**

| Folder | Content |
|---|---|
| `scripts/` | `run_queue.sh` (the job runner), `append_results.sh` (called at the end of each job) |
| `setup/` | environment build, base-model download, TRL KL patch, TextCraft requirements |
| `data/` | AgentGym task lists; TextCraft train/test, their depths, few-shot examples, train + reservoir |
| `runs/` | one folder per experiment family (configs, evaluation logs); `INDEX.md` is the registry; `queue/` the job queue |
| `archive/` | code no longer on the active path, with a README per folder saying why and what replaced it |
| `saves/`, `logs/`, `wandb/` | checkpoints, run logs, W&B files (not versioned) |

---

## 5. Conventions

- **One change at a time.** Every run in the registry differs from its reference by one named
  option; the job script says which in its header.
- **Test before GPU.** `python -m src.tests.test_schedules`, `test_moving_ref`, `test_sampling`
  and `check_queue_jobs` run on CPU in seconds; queue jobs call them first.
- **Persistence.** Anything that cannot be rebuilt (checkpoints, best models with optimizer,
  anchor chains, logs) goes to the home disk; `/tmp` only holds what a script can rebuild.
- **Reporting.** A best score is a maximum over noisy evaluations; report it with the plateau
  (mean of the last ten evaluations). One evaluation on 100 tasks has a standard deviation of
  up to 5 points.
- **Comments.** Each file starts with a plain-English paragraph; the detailed notes below it are
  in French and cite the runs (exp numbers) that motivated each choice.

## 6. References

- Xi et al., *AgentGym-RL* (ICLR 2026) — the reference system and the TextCraft setup.
- Shao et al., *DeepSeekMath* (2024) — GRPO.
- Lialin et al., *ReLoRA* (ICLR 2024) — what our moving anchor turned out to be.
- Cui et al., *The Entropy Mechanism of RL for Reasoning LMs* (2025) — reading grid for collapse.
