# rl-gym-workout — teaching a language model to act, by trial and error

Research internship at Criteo AI Lab, April–September 2026 (Vadim Lagresle). Supervisors:
Alberto Lumbreras, Patrick Gallinari, Alain Rakotomamonjy, Sylvain Lamprier.

This repository trains a small language model (Qwen2.5-3B-Instruct) to solve multi-step tasks
in a text game, using reinforcement learning (RL): the model plays, gets a score, and is
updated to make its successful behaviours more likely. The code is written so that the same
recipe can be moved to other interactive environments, such as a shopping agent that has to
search, compare and act over several steps.

### At a glance

- **82 % pass@1 on TextCraft with a 3B model on one GPU**, above the 75 % published by
  AgentGym-RL on several GPUs, with a LoRA adapter of 0.5 % of the weights.
- **Why multi-turn RL with LoRA collapses, and what prevents it.** Without intervention the
  adapter drifts away from its KL reference geometrically, the policy becomes deterministic, then
  the KL estimator explodes and destroys the generations. Periodically **resetting the adapter**
  (ReLoRA) is what stops it; moving the reference, lowering the learning rate or resetting Adam
  are not enough (§5).
- **The drift follows a clock counted in updates**: about 370 updates at our learning rate,
  whatever the group size. Re-anchoring must happen before that; we had defined the period in
  epochs, which silently doubled it when the group size doubled.
- **Curriculum and KL bound are interchangeable protections**, not sources of performance: either
  one makes groups of 16 viable, and both reach 82 %.
- About 80 registered runs, most differing from a reference run by one named option, all listed
  in [`runs/INDEX.md`](runs/INDEX.md); the final analyses are in
  [`docs/post_rapport/`](docs/post_rapport/).

![Pass@1 against epochs, groups of 8 and 16](docs/post_rapport/figures/fig1_g_x_curriculum.png)

**Contents.** 1. The problem · 2. The environment · 3. The training method · 4. Curricula ·
5. What we found · 6. Installation · 7. Running things · 8. Map of the code · 9. Conventions ·
10. Project status

---

## 1. The problem

A language model used as an agent does not answer once: it acts, observes the result, and acts
again, sometimes for dozens of steps. Training it by RL on such tasks is known to be unstable:
runs can improve for hours and then collapse. The project asks two practical questions, on one
environment studied in depth:

- **Stability.** Under which conditions does multi-turn RL keep improving without collapsing?
- **Efficiency.** How can the same level be reached with less computation and fewer
  interactions with the environment?

The starting point is the published system [AgentGym-RL](https://arxiv.org/abs/2509.08755)
(Xi et al., ICLR 2026), which reports 75 % success on TextCraft with the same 3B model on
several GPUs. We reproduce it on a single GPU, study why training fails, and test fixes.

---

## 2. The environment: TextCraft

TextCraft is a text version of Minecraft crafting. Each episode gives the agent a goal item and
a list of crafting recipes (some useful, some distractors). The agent has three commands:

| Command | Effect | Example |
|---|---|---|
| `get <n> <item>` | take a raw ingredient from the environment | `get 1 iron ingot` |
| `craft <n> <item> using <ingredients>` | apply one of the listed recipes | `craft 1 iron nugget using 1 iron ingot` |
| `inventory` | list what the agent currently holds | `inventory` |

At every turn the model must answer in the format `Thought: … Action: …`; the environment runs
the action and returns an observation (`Got 1 iron ingot`, `Could not find a valid recipe…`).
The episode ends when the goal is crafted (reward **1**) or when the turn cap is reached
(reward **0**). A real, short episode from our logs:

```
Goal: craft iron nugget.        (the recipe list, with distractors, is given above the goal)
Model:  Thought: To craft iron nugget, I need 1 iron ingot.   Action: get 1 iron ingot
Env:    Got 1 iron ingot
Model:  Thought: Now I can craft it.   Action: craft 1 iron nugget using 1 iron ingot
Env:    Crafted 9 minecraft:iron_nugget                                    -> reward 1
```

**Depth** measures difficulty: the number of crafting levels between raw ingredients and the
goal. A depth-1 goal needs one craft; a depth-4 goal needs intermediate items that themselves
need intermediate items, hence long episodes where one wrong step wastes the rest. The data:
374 training recipes, 100 fixed test recipes, and a reservoir of 70 more recipes kept aside for
few-shot examples (`data/`). The environment runs as a small HTTP server (from the
[AgentGym](https://github.com/WooooDyy/AgentGym) project) that the training code talks to.

---

## 3. The training method

We use **GRPO** (Group Relative Policy Optimization, from DeepSeekMath), through the
[TRL](https://github.com/huggingface/trl) library, with [vLLM](https://github.com/vllm-project/vllm)
for fast generation. One training update goes like this:

1. **Draw tasks.** A few recipes are drawn from the training set (`src/train/sampling.py`).
2. **Play a group of episodes per task.** The model plays G episodes of the same task
   (G = 8 or 16), turn by turn against the server (`src/train/rollout.py`).
3. **Compare within the group.** Each episode scores 1 or 0. An episode that succeeded while
   others of its group failed gets a positive *advantage*; the reverse gets a negative one.
   A group where all episodes succeed, or all fail, teaches nothing.
4. **Update the model.** The loss pushes up the probability of the tokens written in the
   good episodes and down in the bad ones. Only the tokens the model wrote are trained; the
   environment's observations stay in the sequence, so the model sees the context it acted in,
   but they are masked out of the loss.
5. **Keep the model close to a reference.** A KL penalty prevents the model from drifting too
   far from a reference model in a single run. Which reference to use turned out to be central
   (see §5).

Instead of updating all 3 billion weights (*full fine-tuning*), most runs train a **LoRA
adapter**: small extra matrices (0.5 % of the parameters) added to the frozen model. It divides
the optimizer memory and checkpoint size by about 50, which made many long runs possible.

Every 47 updates the model plays the 100 test tasks, and the best model is saved
(`src/train/periodic_eval.py`). The main metric is **pass@1**: the share of test tasks solved
in one try.

---

## 4. Curricula

A *curriculum* changes what the agent faces as training progresses, from easy to hard. We
compared three, each with one knob, all otherwise identical:

| Curriculum | What grows | Schedule used | Where |
|---|---|---|---|
| **Horizon** | the maximum number of turns per episode | 10 turns, then 20 at epoch 15, then 30 at epoch 30 | `--max-rounds-schedule-epochs`, `horizon_schedules.py` |
| **Budget** | the maximum number of tokens per turn | 256, then 512, then 1024 tokens | `--max-completion-schedule-epochs`, `horizon_schedules.py` |
| **Depth** | the depth of the recipes that can be drawn | depth ≤ 1, then ≤ 2, … raised when the training reward reaches 0.8 | `--depth-schedule-auto`, `sampling.py` |

Why it helps: with a short turn cap, the hard tasks cannot be solved at all, so their groups
all fail and produce no gradient; the model first becomes reliable on short tasks, and the
hard tasks come back when the cap is raised. The same file also implements a fixed
rebalancing of the draw by depth (`--depth-balance`), used to give more weight to the rare
deep recipes.

---

## 5. What we found

pass@1 on the 100 test tasks (best score / plateau = mean of the last ten evaluations):

| Configuration | pass@1 |
|---|---|
| Qwen2.5-3B, no training | 10 |
| Published recipe, full fine-tuning, fixed KL reference | 69 / 61 |
| LoRA, fixed KL reference | collapses (best ≤ 39) |
| LoRA, moving KL reference, no curriculum | 73 / 70 |
| **Horizon curriculum** | **82 / 78** |
| No curriculum, larger groups, bounded KL estimator | 82 / 76 |
| AgentGym-RL-3B, published, several GPUs | 75 |

The lessons that the code encodes:

1. **Mask the observations, do not drop them.** Our first version removed the environment's
   text from the training sequence; the loss decreased and the model learned nothing.
2. **LoRA needs a moving KL reference.** Against a fixed reference (the base model), LoRA drifts
   away faster and faster and collapses. Moving the reference to the current model every four
   epochs fixes it (`kl_anchor.py`). The way we move it (merge the adapter, start a fresh one)
   is in fact the ReLoRA method; lesson 5 shows which part of it matters.
3. **Bound the KL estimator.** TRL's per-token KL estimate is unbounded, unlike the code of the
   reference paper; with groups of 16, a single token can then dominate an update and destroy
   the model. Bounding it at 10 per token (`--kl-clamp 10`) removes the collapse.
4. **What a curriculum still buys once training is stable.** The same final level with about
   30 % fewer GPU hours and 40 % fewer environment interactions, and a model that keeps more
   diversity in its behaviour, hence more room to keep exploring. A curriculum and the KL bound
   are interchangeable protections at G = 16 (82 % both); at G = 8, where nothing collapses,
   the curriculum adds nothing.
5. **It is the reset of the adapter that stabilises, not the moving reference.** ReLoRA does three
   things at each re-anchoring: it moves the KL reference, it resets the adapter (the product
   B·A goes back to zero after being merged into the base weights), and it resets Adam. Taken
   apart: the moving reference alone collapses (`--moving-anchor-mode ref`); so does the same
   with a learning rate divided by three, only later; adding an Adam reset
   (`--moving-ref-reset-adam`) keeps the score but the policy becomes almost deterministic. Only
   the adapter reset keeps a healthy run.
6. **Re-anchor before about 370 updates.** Without a reset, the KL to the reference reaches about
   0.1 after ~370 updates at learning rate 3e-6, with groups of 8 as with groups of 16, and
   explodes ~100 updates later; a three times lower learning rate only moves the threshold to
   ~780. Every run whose re-anchoring period exceeded the threshold collapsed (444, 930 and
   1 100 updates); every run below it held. Our period was set in epochs: "4 epochs" meant 184
   updates at G = 8 but 372 at G = 16, right on the threshold. A period in updates is safer.

| Which part of ReLoRA matters | The drift clock |
|---|---|
| ![ReLoRA ablation](docs/post_rapport/figures/fig5_relora_ablation.png) | ![Drift clock](docs/post_rapport/figures/fig6_drift_clock.png) |

The figures are labelled in French. The internship report ([`docs/rapport/`](docs/rapport/), LaTeX sources in French and English) covers lessons 1 to 4; lessons 5 and 6
come from the runs that followed it, analysed in
[`docs/post_rapport/BILAN_FINAL.md`](docs/post_rapport/BILAN_FINAL.md) (French). Every run is
registered in [`runs/INDEX.md`](runs/INDEX.md).

---

## 6. Installation

**What you need.** A Linux machine with one recent NVIDIA GPU, Python 3.11, and about 15 GB of
disk for the environment and the base model, plus room for checkpoints. Our runs used one
GPU with 192 GB of memory, giving half of it to the vLLM generation engine
(`--vllm-gpu-util 0.5`); the 3-update smoke test runs with 15 % (about 27 GB for vLLM). Smaller
GPUs have not been tested: lower `--vllm-gpu-util`, `--vllm-max-len` and the group size first.

**Our setup, to adapt to yours.** Our machine wiped `/tmp` whenever it restarted, so the Python
environment and the base model lived in `/tmp` and were rebuilt by script, while checkpoints
and logs lived on a persistent disk. The default paths reflect that; change them if your
machine differs:

| What | Default | How to change it |
|---|---|---|
| Base model | `/tmp/models/Qwen2.5-3B-Instruct` | `--model-path`, or `DEFAULT_MODEL_PATH` in `src/train/data.py` |
| Training environment | `/tmp/envs/agentgym-rl-v2` | first argument of `setup/setup_agentgym_rl_v2.sh`; the `PATH` line of each job script |
| Checkpoints, best models | `saves/` in the repository | `--output-root` (periodic checkpoints); the best model always goes to `saves/trl_grpo/<run>_best` |
| TextCraft server | `http://127.0.0.1:36005` | `ENV_SERVER_URL` in `src/train/data.py` |
| Experiment tracking | Weights & Biases, project `rl-gym-workout` | `WANDB_PROJECT`, `WANDB_ENTITY` environment variables |

**Steps.**

```bash
# 0. The code, with the AgentGym and MAGELLAN submodules
git clone --recursive https://github.com/Vadim-Lagresle/rl-gym-workout.git && cd rl-gym-workout

# 1. Training environment: recent TRL and vLLM, flash-attention, and the KL-bound patch for TRL
bash setup/setup_agentgym_rl_v2.sh
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"

# 2. Base model
bash setup/ensure_qwen_tmp.sh

# 3. TextCraft server, in its own virtualenv (requirements: setup/requirements-agentenv-textcraft.txt).
#    AgentGym is a git submodule (clone with --recursive, or fetch it now), then start the server
#    FROM the package folder: it reads its recipe files with a relative path.
git submodule update --init external/AgentGym
cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005

# 4. Check everything on CPU, then on GPU (3 updates, a few minutes)
python -m src.tests.test_schedules && python -m src.tests.test_sampling
bash src/tests/smoke_train.sh
```

---

## 7. Running things

**A training run.** The command below is the Horizon curriculum of §5. On our machine, closing
the IDE session killed attached processes, hence `setsid nohup` to detach the run.

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

What the main options mean: `--num-generations` is the group size G; `--gradient-accumulation-steps`
the number of episodes per update (64, so 4 tasks of 16 episodes); `--beta` the weight of the KL
penalty; `--moving-anchor-every-epochs` how often the KL reference moves; `--kl-clamp` the bound
on the KL estimator. `python src/train/train_grpo.py --help` lists every option, and
`src/train/cli.py` explains each one with the run that motivated it.

**Following a run.** Its text log is `logs/<run>.log` (losses, rewards, every evaluation, every
re-anchoring); metrics also go to Weights & Biases. Evaluation lines look like
`[test_eval] step 705 — Pass@1 = 38/100`.

**The run queue.** Long experiments were shell scripts in `runs/queue/`, executed one after the
other by one runner; a finished job moved to `runs/queue/done/`. Each job checked its
prerequisites and ran the CPU tests before starting the GPU. The job scripts were removed when
the project closed (they remain in the git history); the runner works as soon as
`runs/queue/<NN>_<name>.sh` files exist again.

```bash
setsid nohup bash scripts/run_queue.sh >> logs/queue.log 2>&1 &
```

**Evaluating a saved model** on the 100 test tasks:

```bash
bash src/utils/start_vllm_server.sh <full_model_dir>              # vLLM server on port 8001
python src/eval/eval_textcraft.py --model <full_model_dir> --run-name <family>/<name>
kill $(cat /tmp/vllm_server.pid)
```

A LoRA run saves only the adapter: merge it into the base model with `src/utils/merge_lora.py`,
or, for a run with a moving reference (merge mode), rebuild the full model with
`src/utils/merge_anchor_chain.py`. `src/eval/eval_oracle.py` measures pass@k (k tries per task);
`src/analysis/analyze_eval.py` breaks an evaluation down by depth and by error type.

---

## 8. Map of the code

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
│   ├── horizon_schedules.py     Horizon (turns) and Budget (tokens) curricula
│   ├── sampling.py              Depth curriculum and depth rebalancing of the task draw
│   ├── periodic_eval.py         test evaluation every N updates; saves the best model
│   ├── diagnostics.py           GPU memory telemetry
│   └── lr_schedules.py, lr_adaptive.py   learning-rate schedules of early runs (kept for reproducibility)
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
│   ├── merge_anchor_chain.py    rebuild a full model from a moving-reference run
│   ├── label_depths.py          depth (1-4) of every TextCraft task
│   ├── build_fewshot_examples.py  solved examples from the reservoir recipes
│   ├── vllm_supports.py         can the installed vLLM serve this model?
│   └── start_vllm_server.sh     vLLM server for evaluation
└── tests/                       run these before any GPU job
    ├── test_schedules.py        learning-rate, horizon and budget schedules; merge-mode reference
    ├── test_moving_ref.py       frozen-copy reference (mode 'ref')
    ├── test_sampling.py         Depth curriculum, depth rebalancing, sampler shape
    ├── check_queue_jobs.py      every queued job passes valid options
    └── smoke_train.sh           3 GPU updates through the whole recipe
```

**How the training files call each other.** `train_grpo.py` reads the options (`cli.py`), loads
the tasks (`data.py`), builds the TRL configuration (`grpo_config.py`) and the callbacks
(`callbacks.py`, which creates the evaluation, the KL reference and the telemetry), then hands
everything to TRL's `GRPOTrainer`. At every update, TRL calls `rollout.grpo_rollout_func`, which
plays the episodes through `vllm_engine.py` with the caps given by `horizon_schedules.py`;
`sampling.py` decides which tasks are drawn. `periodic_eval.py` reuses the evaluation loop of
`eval/textcraft_common.py`, so that scores during training and offline are comparable.

**Other folders.**

| Folder | Content |
|---|---|
| `setup/` | building the machine: training environment, base-model download, TRL patch, TextCraft requirements. Run once per machine (or after a reset); never imported by the code |
| `scripts/` | running experiments: the queue runner and the end-of-job summary |
| `data/` | task lists: TextCraft train, test, depths, few-shot reservoir; other AgentGym environments for future work |
| `runs/` | one folder per experiment family (configs, evaluation logs); `INDEX.md` is the registry of all runs |
| `archive/` | code no longer used, with a README per folder saying why and what replaced it |
| `external/` | third-party code. `AgentGym` (the TextCraft server) and `MAGELLAN` are git submodules pinned to unmodified upstream commits (`git clone --recursive`); `AgentGym-RL` is a copy of the reference paper's verl code (Apache 2.0, `VERL_LICENSE`, `Notice.txt`); `agentgym_rl_paper/` holds the paper's scripts. `USAGE.md` says which files the project actually uses |
| `logs/` | text output of every run, the queue and the server; the figure scripts read their metrics here (not versioned) |
| `saves/`, `wandb/` | checkpoints, best models, W&B files (not versioned) |
| `docs/` | everything written along the way, in French: `post_rapport/` (final analyses of the last runs: `BILAN_FINAL.md`, `ANALYSE_runs_post_rapport.md`, figures and the scripts that rebuild them), `rapport/` (internship report, LaTeX, French and English, with its figures), `slides/`, `hebdo/` (weekly notes), `RESULTS.md` (results table, one block per run), `WORKLOG.md` (journal up to June), `MAGELLAN_ANALYSE.md` (study of the MAGELLAN autocurriculum), `PLAN_EXPERIENCES.md`, `dashboard/`, `archive/` |

---

## 9. Conventions

- **One change at a time.** Every run differs from its reference run by one named option; the
  header of its job script says which.
- **Test before using the GPU.** The CPU tests take seconds; the queue jobs run them first.
- **Keep what cannot be rebuilt on persistent storage.** Checkpoints, best models with their
  optimizer state, the chain of KL references of a moving-reference run, and logs.
- **Report plateaus, not peaks.** One evaluation on 100 tasks has a standard deviation of up to
  5 points; a best score over dozens of evaluations overestimates the real level.
- **Comments.** Each file starts with a plain-English paragraph; the detailed notes below it are
  in French and cite the runs (exp numbers of `runs/INDEX.md`) that motivated each choice.

## 10. Project status

The project closed in October 2026. Nothing is running; the last runs and what they leave open
(a stronger fixed KL penalty, a re-anchoring period counted in updates, the depth-4 wall that no
run has broken) are listed at the end of
[`docs/post_rapport/BILAN_FINAL.md`](docs/post_rapport/BILAN_FINAL.md).

## References

- Xi et al., *AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making*, ICLR 2026.
- Prasad et al., *ADaPT*, NAACL 2024 — origin of TextCraft.
- Shao et al., *DeepSeekMath*, 2024 — GRPO.
- Hu et al., *LoRA*, ICLR 2022; Lialin et al., *ReLoRA*, ICLR 2024.
- Cui et al., *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*, 2025.
