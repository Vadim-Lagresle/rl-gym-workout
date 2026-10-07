# Project Tracking — Self-Improving LLM Agents by Multi-Turn RL (TextCraft)

*Vadim Lagresle, Criteo AI Lab, April–September 2026. Supervisors: Alberto Lumbreras, Patrick Gallinari, Alain Rakotomamonjy, Sylvain Lamprier. Hosted by the AIAC team (Maxime).*

This page is the condensed logbook: one block per month with the direction, what was done, the results and the decisions. Day-to-day detail lives in the repository (`docs/hebdo/`, `runs/INDEX.md`, `docs/RESULTS.md`).

**Contents**

1. [Context and objective](#1-context-and-objective)
2. [Key results](#2-key-results)
3. [Timeline by month](#3-timeline-by-month)
4. [Ideas explored and their status](#4-ideas-explored-and-their-status)
5. [Open questions and runs in progress](#5-open-questions-and-runs-in-progress)
6. [Working practices and infrastructure](#6-working-practices-and-infrastructure)
7. [Pointers](#7-pointers)

---

## 1. Context and objective

- **Offer.** Autonomous improvement of LLM agents through synthetic data and preference filtering (West-of-N, BOND, SCPO, Coral). Exploratory internship: the student scopes the topic, the supervisors validate.
- **Scope chosen (May).** Once an environment is given, how does a model improve best on it by RL: what makes training stable, what makes it efficient. Environment TextCraft (AgentGym-RL), model Qwen2.5-3B-Instruct, algorithm GRPO, single GPU (B200 192 GB).
- **Target.** Replicate the published 75 % pass@1, understand the failure modes, test curricula and the levers of efficiency.
- **Horizon for Criteo.** Shopping agents trained by RL on a commerce environment; the stabilisation recipe and the difficulty control are what transfers.

## 2. Key results

| Configuration | pass@1 best / plateau | Note |
|---|---|---|
| Qwen2.5-3B, zero-shot | 10.3 | mean of 20 passes (18 on the single historical pass) |
| Full fine-tuning, fixed reference (paper recipe) | 69 / 61 | stable over 247 epochs, 238 GPU h |
| LoRA, fixed reference (grid r × LR × β) | ≤ 39 then collapse | geometric KL drift, all runs |
| ReLoRA moving anchor, G=8 (exp25) | 73 / 70 | 170 epochs, no collapse |
| Control G=16, no curriculum (exp36) | 54 then collapse | unbounded k3 estimator |
| Control G=16 + KL bound (exp43) | 82 / 76 | 80 epochs, entropy 0.13 |
| **Horizon curriculum (exp32)** | **82 / 78** | 30 % fewer GPU h, 40 % fewer env turns to reach 75 |
| Depth curriculum (exp33) / Budget curriculum (exp35) | 80 / 74 · 80 / 75 | |
| MAGELLAN autocurriculum (exp34) | 45 then collapse | epoch 7 |
| AgentGym-RL-3B (paper) | 75 | multi-GPU, verl |
| Qwen3.5-4B, zero-shot | 78.5 | no training; Gemini 3.5 Flash 99 |

## 3. Timeline by month

### April 2026 — onboarding and scoping
- **Direction.** Read the offer's papers, understand Criteo's agentic teams, find a topic that mixes agents and RL.
- **Done.** Kick-off 10/04 (exploratory scope, autonomy, weekly reviews). Literature presentation 17/04 on West-of-N, SCPO, Coral, BOND: all single-turn, no tools, no dataset creation from scratch. Interviews: Clément (ACT), Maxime (AIAC), Flavian and Imad (DeepShopper, 20/04), Jérémy's reading group on Semantic IDs for recommendation (21/04, 28/04). Onboarding: GCP VM, Cursor, Claude Code, Hugging Face and DeepLearning.AI courses. Broad reading on agent trends (evolving environments, protocols).
- **Supervisors' framing.** Alain: agents are viable only if inference is efficient, hence small models. Patrick: West-of-N is the central paper of the offer; the two ideas (synthetic user–assistant dataset, then training on it) must be treated separately, and the first is too ambitious. 17/04 questions raised: online RL versus offline self-training (the offer criticises online RL, the papers rely on it); whether the word "agent" is even needed (only Coral is multi-turn); every paper needs a seed dataset.
- **Decisions.** DeepShopper set aside: too close to recommendation and production for a six-month academic contribution (six RecSys tools, another intern will do RL on it, NeurIPS submission planned; Imad's advice: do not touch the tools, study how the model uses them). Protocols (MCP, A2A) out of scope. Weekly 24/04: stay general on self-improvement, consider restricting to one conference; interests noted for later: multi-turn rewards, long context, environment and task design, difficulty curriculum, SFT + RL mixing. 29/04 Alberto agrees to build on AgentGym-RL; 30/04 Patrick: frame the work tightly, read the OpenReview reviews.
- **Lesson.** Beware of undefined words ("emergent", "multi-agent"): every claim needs a bound or a clean empirical test (Jérémy, 28/04).

### May 2026 — TextCraft chosen, first failures, project restructuring
- **Direction.** Replicate AgentGym-RL on TextCraft; isolate the depth-4 difficulty (0 % for every model, trained or not).
- **Done.** 07/05 TextCraft chosen (pre-coded, hard for frontier models, extensible to the other AgentGym tasks). 12–13/05 hand-made GRPO on one A100 40 GB with degraded hyperparameters (LoRA r=16, G=2, 128 tokens, β≈0.04): 8 to 18 % versus 18 % untrained. Insight: at G=2 about 70 % of groups have zero reward variance and produce no gradient; DeepSeekMath says the same. B200 requested. 18/05 code review with Alberto: new layout (`src/`, `runs/`, `external/`), TRL instead of the paper's verl, W&B, micro-commits, Gerrit. 19–21/05 written experiment plan: paper replication on one GPU, ScalingInter at 3B, joint curriculum (turns, context, response length, depth), LoRA versus full fine-tuning, SCPO data, SFT + RL alternation. 26–28/05 W&B set up, Kimi K2.6 and Toolathlon read.
- **Decisions.** One environment, one mechanism at a time; write experiments before launching them; do not trust AI-generated training code without reading it (reward shaping had been invented silently).
- **Supervisors' feedback.** 22/05 Patrick: finding a niche in a field is work in itself; the intern is not expected to find the topic alone.
- **Doubts recorded.** Curricula are a classic idea (Hassan); Gemini 3.5 Flash reaches 99 % on TextCraft (21/05); is a 3B model enough. Answer kept: the object of study is the training dynamics, not the score.

### June 2026 — the observation-masking bug, code ownership, SNIS idea
- **Direction.** Own the codebase; make GRPO learn at all.
- **Done.** 04/06 root cause of "loss decreases, model does not learn": observations were dropped from the training sequence instead of masked; fixed (report §4.2.1). Deep dive verl versus TRL; Stanford CS224R lectures 1–3; Dr. GRPO read; first Qwen3.5 test. 15/06 contact with Performance Science for RL theory advice. 22/06 Otmane: reuse the group's rollouts to credit each action (self-normalised importance sampling); Chelsea Finn and Dylan Foster recommended. 26/06 three directions written: reasoning ↔ multi-turn transfer, LoRA stabilisation, SNIS.
- **Supervisors' feedback.** 05/06 Alberto: the sweet spot in model size is about 4B (test it); Alain: what is gained or lost from 3B to 0.5B, and more generally the law between model size and gains in time and memory; Patrick: try Gemma 4. 19/06 Patrick: write a mini-paper to expose the ideas to others; look for general ideas rather than solving Minecraft. 26/06: go for Otmane's SNIS idea, knowing it leaves the curriculum study aside for a while.
- **Decisions.** 19/06 reframing: the subject is autonomous improvement, RL is the means; keep reading, start the report early. Store training metrics for convergence analyses. The model-size question was answered only partially, by evaluating Qwen3.5-4B zero-shot in July (78.5 %).

### July 2026 — LoRA lineage, SNIS, stack v2, the end-of-internship question
- **Direction.** Stabilise pure GRPO with LoRA; test SNIS; define a defensible end-of-internship objective.
- **Done.** LoRA warm-start lineage reaches 54 % (peak 58, exp10.8): LoRA stabilises where full training on one GPU did not. SNIS coded and run (exp17): recombining sub-trajectories without causal alignment collapses, the lock is identified (Appendix B). Reasoning-only training does not transfer to interactive control (Appendix D). Snell et al. on test-time compute read (21/07). 22/07 stack v2: TRL 1.9 + vLLM 0.25 colocate + flash-attention, VM migrated; Qwen3.5-4B evaluated at 78.5 % zero-shot. Presentations to Keïn and Sylvain's group; MAGELLAN, BOSS, HERAKLES on the reading list. Adaptive learning-rate schedules tried (staged, reward-adaptive, restore-best).
- **Global review (24/07).** Presentation of the whole internship since day one to the supervisors (`Pres_Juillet.pdf`).
- **Decision (23/07).** End-of-internship objective: quantification and ablation of the improvement levers on one environment, compute-controlled, in the spirit of Snell et al.; stay on Qwen2.5-3B for comparability with the paper. Applications to PhD positions in parallel.
- **Lesson.** `setsid nohup` for every run: closing the IDE session killed runs 45 minutes later.

### August 2026 — replication reference, LoRA grid, moving anchor, curricula
- **Direction.** Establish the full fine-tuning reference, understand LoRA's instability, fix it, then run the curricula.
- **Done.** exp23 lineage: paper recipe in full fine-tuning, 100 then 247 epochs, 69 % best, no collapse, KL linear. exp24 LoRA grid (r ∈ {16, 32, 64} × LR × β, fixed reference): every run collapses, KL grows geometrically ×5–10 per two epochs; rank does not help (12/08). 14/08 silent purge of `/tmp`: checkpoint policy moved to the home disk. 17/08 exp25 moving KL anchor (merge adapter every 4 epochs, reset, purge Adam): 65 % at epoch 100, 73 % after resume; exp30/31 anchor × β square: the reference regime, not β, governs stability. 7B attempt (exp26) abandoned on a bug and memory. 24/08 exp32 Horizon curriculum at G=16 launched; 26/08 record 82 %; exp36 control G=16 without curriculum collapses at epoch 10 (27/08); exp33 Depth (calendar then adaptive stages) and exp35 Budget curricula run, 80 % each after resumes. Report V1 written (intro, deep-RL appendix, experiments).
- **Decisions.** Binary exact reward kept; β = 0.01 kept by lineage continuity; curricula compared at equal recipe and at equal compute (tokens, GPU hours, environment turns).

### September 2026 — freeze, report, defence, then the collapse mechanism
- **Direction.** Freeze experiments (03/09), write and defend; afterwards, explain the G=16 collapse and requalify the method.
- **Done.** Report V2 → V3 (03–07/09), defence mid-September. exp34 MAGELLAN, exp36.1 (anchor every 12 epochs) and exp38 (paper's 256-trajectory batches) collapse; exp39 (G=16, 8 tasks/step) stable at 72; exp40 (anchor every 8 epochs) collapses at epoch 15; exp41 (G=8, anchor every 8) 64. 15/09 token-by-token replay of the collapse: the KL jump comes from the unbounded k3 estimator on forced end-of-turn tokens (ρ ≈ 45, k3 ≈ 10^19); verl bounds k3 to [−10, 10], TRL does not. 16/09 the "moving anchor" is recognised as ReLoRA (accumulated rank, B·A and Adam reset each cycle): three confounded factors. exp43 control G=16 with the bound: no collapse, 82 / 76 in 80 epochs (23/09). exp45 fixed reference with the bound still collapses: the fixed-reference failure is drift, not the estimator. Report V4 (French) and full English translation; appendices F.7 (replay) and F.8 (what the curriculum adds once the collapse is prevented: 30 % fewer GPU hours and 40 % fewer environment turns to reach 75 %, twice the entropy at equal score).
- **Decisions.** Remaining experiments ordered: exp44 Horizon at G=8, exp41 resume, exp46 true LoRA with a frozen-copy moving reference (separates the moving reference from merge-and-restart), then a short LoRA fixed-reference run at LR 1e-6, then dataset rebalancing if time allows. Seeds dropped; MAGELLAN parked.

## 4. Ideas explored and their status

| Idea | Status | Where |
|---|---|---|
| Shaped rewards (format bonus, per-turn bonus) | dropped: reward hacking, policy learns to give up | report §4.2.1 |
| Learning-rate regulation (staged, adaptive, restore-best) | superseded by the moving anchor | Appendix C |
| Moving KL anchor by merge-and-restart (= ReLoRA) | adopted; factors being separated (exp46) | §4.2.4 |
| Horizon / Depth / Budget curricula | adopted; efficiency and exploration margin | §4.3, App. F.8 |
| KL estimator bound (k3 ≤ 10) | confirmed cause of the G=16 collapse | §4.2.5, App. F.7 |
| SNIS trajectory recombination (Otmane) | parked: causal-alignment lock | Appendix B |
| Single-turn planning ↔ multi-turn control transfer | negative preliminary result, parked | Appendix D |
| MAGELLAN autocurriculum | collapsed at G=16; needs a stable learner | Appendix E |
| Few-shot prompting and few-shot + RL | +17 points zero-training; RL variant not rerun | §4.5 |
| SCPO / BOND / SFT-on-successes (offer papers) | not run: exact reward made them unnecessary | §4.5 |
| Qwen 7B, Qwen3.5 training | 7B abandoned (memory, bug); 3.5 evaluated only | §4.5 |
| DeepShopper / generative recommendation (SIDs) | out of scope by decision | April |

## 5. Open questions and runs in progress

1. **exp44 — Horizon at G=8** (running since 23/09): does the curriculum still pay at G=8?
2. **exp41 resume** to 80 epochs: G=8 with anchor every 8 epochs.
3. **exp46 — true LoRA with a frozen-copy moving reference**: is the moving reference the active ingredient, or the reset of B·A and Adam? Read on the intra-window KL.
4. **(E) LoRA, fixed reference, LR 1e-6**, short: is fixed-reference LoRA stable at full fine-tuning speed?
5. **(F) Rebalance the training set by depth** (one depth-4 recipe today).
6. Later: seeds, another AgentGym-RL environment, a WebShop-like environment.

## 6. Working practices and infrastructure

- **Code.** `src/train` (GRPO entry point, rollout loop, schedules, periodic evaluation), `src/eval`, `src/analysis`; external code in `external/`; every run has a config and a registry line. One task at a time, explained before coded; dry run or CPU self-test before GPU.
- **Runs.** Queue runner (`scripts/run_queue.sh`), jobs in `runs/queue/`; checkpoints, best models and anchor chains on the home disk; `/tmp` only for rebuildable artefacts (base model, environment). Pod recreations kill everything and purge `/tmp`: resume jobs are written for each interrupted run.
- **Tracking.** W&B project `rl-gym-workout`; `runs/INDEX.md`; weekly session notes in `docs/hebdo/`; daily GitLab push, weekly Gerrit snapshot.
- **Compute.** Single B200 192 GB (Coder). Full fine-tuning: 12 GB per checkpoint (model 5.8 + 8-bit Adam 6.2). LoRA r=8: 0.18 GB.

## 7. Pointers

- Repository: `rl-gym-workout` (GitLab `v.lagresle/rl-gym-workout`; Gerrit `ai-agentic-commerce-incubation/research/vadim-lagresle/`)
- Report: `docs/rapport/RAPPORT_eng.tex` (EN) · `RAPPORT_V4_fr.tex` (FR) · slides `docs/slides/SOUTENANCE_V3.tex`
- Registry and results: `runs/INDEX.md` · `docs/RESULTS.md` · `docs/PLAN_EXPERIENCES.md`
- W&B: LoRA lineage report — [wandb.ai/v-lagresle-criteo/rl-gym-workout](https://wandb.ai/v-lagresle-criteo/rl-gym-workout/reports/LoRa-Training-3--VmlldzoxNzQ0NDExMg)
- Literature page: *Literature — Self-Improving LLM Agents by Multi-Turn RL* (Confluence)
- Reference paper: AgentGym-RL — [arXiv:2509.08755](https://arxiv.org/abs/2509.08755) · [OpenReview](https://openreview.net/forum?id=ZgCCDwcGwn)
