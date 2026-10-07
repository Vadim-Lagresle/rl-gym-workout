# Literature — Self-Improving LLM Agents by Multi-Turn RL

*Internship of Vadim Lagresle, Criteo AI Lab, April–September 2026. Supervisors: Alberto Lumbreras, Patrick Gallinari, Alain Rakotomamonjy, Sylvain Lamprier.*

This page keeps the papers that actually shaped the project, grouped by the question they answer. Each card gives the link, a two-line summary, the contribution, and what we used it for. The full bibliography (66 entries) is in the report; a short "further reading" list closes the page.

**Contents**

1. [Papers from the internship offer](#1-papers-from-the-internship-offer)
2. [RL for LLMs: foundations and the GRPO family](#2-rl-for-llms-foundations-and-the-grpo-family)
3. [Agentic RL: the reference system and its environment](#3-agentic-rl-the-reference-system-and-its-environment)
4. [Stability, exploration and the limits of RLVR](#4-stability-exploration-and-the-limits-of-rlvr)
5. [Parameter-efficient RL: LoRA and its dynamics](#5-parameter-efficient-rl-lora-and-its-dynamics)
6. [Curriculum learning and autotelic agents](#6-curriculum-learning-and-autotelic-agents)
7. [Self-improvement, synthetic tasks and environments](#7-self-improvement-synthetic-tasks-and-environments)
8. [Benchmarks and evaluation of agents](#8-benchmarks-and-evaluation-of-agents)
9. [Agents for commerce and recommendation](#9-agents-for-commerce-and-recommendation)
10. [Courses and resources](#10-courses-and-resources)
11. [Further reading (not summarised)](#11-further-reading-not-summarised)

---

## 1. Papers from the internship offer

The offer framed the topic as *autonomous improvement of agents through synthetic data and preference filtering*. These papers share one recipe: sample N answers, filter them (reward model, majority vote, belief filtering), learn from the filtered pairs. All are single-turn and without tools; the project moved to multi-turn verifiable rewards, but the sampling-and-filtering logic is the ancestor of GRPO's group.

**West-of-N: Synthetic Preferences for Self-Improving Reward Models** (Pace et al., Google DeepMind, 2024) — [arXiv:2401.12086](https://arxiv.org/abs/2401.12086)
*Summary.* Sample N responses per query, keep the best and worst according to a base preference model, and train the reward model on these synthetic pairs.
*Contribution.* On-policy synthetic preference data improves the reward model as much as extra human data.
*Use for us.* Reference point for "Best-of-N as a training signal"; also the reminder that the offer's setting was *non-verifiable* rewards, which TextCraft is not.

**BOND: Aligning LLMs with Best-of-N Distillation** (Sessa et al., Google DeepMind, 2024) — [arXiv:2407.14622](https://arxiv.org/abs/2407.14622)
*Summary.* Distils the Best-of-N distribution into the policy with a Jeffreys divergence, so one sample at inference has Best-of-N quality.
*Contribution.* J-BOND, an iterative algorithm with an exponential moving-average anchor policy.
*Use for us.* The EMA anchor is the closest published relative of our moving KL reference (their Fig. 5 shows EMA beating a periodic anchor).

**Self-Consistency Preference Optimization (SCPO)** (Prasad et al., Meta, 2024) — [arXiv:2411.04109](https://arxiv.org/abs/2411.04109)
*Summary.* Moves self-consistency from inference to training: the most frequent final answer becomes the chosen response, the least frequent the rejected one, with a DPO loss weighted by the vote margin.
*Contribution.* Matches gold-label training on GSM8K and MATH without any label.
*Use for us.* Candidate for generating training signal on tasks where the verifier is missing; parked because TextCraft gives an exact reward.

**Collaborative Reasoner (Coral): Self-Improving Social Agents with Synthetic Conversations** (Meta, 2025) — [Meta AI publication page](https://ai.meta.com/research/publications/collaborative-reasoner-self-improving-social-agents-with-synthetic-conversations/)
*Summary.* Two agents must agree on an answer; synthetic conversations are filtered by belief tracking and used for DPO.
*Contribution.* Up to +29.5 % on collaborative reasoning, not reproduced on ChatGPT and Gemini.
*Use for us.* Read for the multi-turn angle; its improvement is social, not agentic (no environment, no tools), so it left the scope in April.

**How the four offer papers answer the four challenges of the offer** (from the review of 17/04/2026)

| Challenge | Coral | SCPO | West-of-N | BOND |
|---|---|---|---|---|
| Synthetic generation | Tree sampling of multi-turn dialogues between two agents (beam of 5 per turn) | N answers per problem, plus few-shot generation of new problems from a seed set | N ≤ 64 on-policy answers to unlabelled real queries | N variants per prompt to build the Best-of-N distribution |
| Evaluation and filtering | Belief filtering: an extractor reads each agent's implicit answer and labels every turn against the ground truth | Self-consistency vote: most frequent answer is chosen, least frequent rejected | Teacher reward model picks best and worst, with confidence and likelihood filters | Reward model picks the best of N |
| Learning from preferences | Turn-level DPO pairs | DPO weighted by the vote margin | Student reward model trained on human plus synthetic pairs | Distribution matching with a Jeffreys divergence |
| Iterative training and drift control | Cap on pairs per problem; little gain from re-iteration | Model at iteration t is the reference at t+1; DPO + NLL anchor | Small N and likelihood filtering keep data on-policy | Iterative J-BOND with a moving (EMA) anchor |

**Embarrassingly Simple Self-Distillation (SSD) Improves Code Generation** (2025) — *(link to add)*
*Summary.* Sample at a chosen temperature and truncation, then plain SFT on the model's own outputs, with no verifier or RL.
*Contribution.* Training sharpens the distribution where precision matters ("locks") and keeps diversity where exploration matters ("forks").
*Use for us.* An early hint that self-improvement is largely *sharpening*; see Huang et al. and the pass@k ceiling debate in Section 4.

---

## 2. RL for LLMs: foundations and the GRPO family

**Trust Region Policy Optimization** (Schulman et al., ICML 2015) — [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
*Summary.* Each update stays inside a KL trust region around the previous policy.
*Contribution.* Monotonic-improvement guarantee under a KL constraint.
*Use for us.* The idea behind a KL penalty to a *recent* policy, which our moving anchor implements at the scale of a few epochs.

**Proximal Policy Optimization** (Schulman et al., 2017) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
*Summary.* Replaces the trust-region constraint by a clipped importance ratio.
*Contribution.* The workhorse of RLHF; GRPO keeps its objective and drops the critic.
*Use for us.* Our recipe is strictly on-policy (one update per rollout batch), so the clip is inert; the paper's recipe used four clipped steps per batch.

**Training language models to follow instructions with human feedback (InstructGPT)** (Ouyang et al., NeurIPS 2022) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
*Summary.* SFT, reward model, PPO with a KL penalty to the SFT policy.
*Contribution.* The RLHF pipeline and the fixed-reference KL penalty everybody inherited.
*Use for us.* The fixed reference is exactly what fails with LoRA in our multi-turn regime.

**DeepSeekMath** (Shao et al., 2024) — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
*Summary.* Introduces GRPO: advantage normalised within a group of G samples of the same prompt, no value network.
*Contribution.* Critic-free RL that scales to LLMs; also notes that gradient quality collapses for small groups.
*Use for us.* Our algorithm. The small-group remark explained our first failure at G=2 (70 % of groups without gradient).

**DeepSeek-R1** (DeepSeek-AI, 2025) — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
*Summary.* RL with verifiable rewards at scale, rule-based rewards only (accuracy and format).
*Contribution.* Reasoning emerges from RL alone; learned reward models are rejected for reward-hacking risk.
*Use for us.* The case for keeping a binary exact reward after our shaped-reward runs collapsed.

**Understanding R1-Zero-like training (Dr. GRPO)** (Liu et al., 2025) — [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
*Summary.* Dissects GRPO's normalisation terms and shows a length bias.
*Contribution.* Dividing by response length penalises short wrong answers less than long ones and inflates outputs.
*Use for us.* The lesson that an innocuous term changes behaviour once you follow it through the advantage; same logic as our reward-shaping failure.

**Approximating KL divergence** (Schulman, blog 2020) — [joschu.net](http://joschu.net/blog/kl-approx.html)
*Summary.* Three Monte Carlo estimators of the KL from sampled tokens; k3 = e^ρ − ρ − 1 is unbiased, positive and low-variance.
*Contribution.* The estimator used by TRL and verl.
*Use for us.* k3 is unbounded in TRL and bounded to [−10, 10] in verl; that single difference explained the G=16 collapse (report §4.2.5, Appendix F.7).

---

## 3. Agentic RL: the reference system and its environment

**ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., ICLR 2023) — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
*Summary.* Interleave free-text thoughts with environment actions and observations.
*Contribution.* The Thought / Action / Observation loop.
*Use for us.* The format of every trajectory we train; the disappearance of the "Thought" token is one of our collapse signals.

**AgentGym** (Xi et al., ACL 2025) — [arXiv:2406.04151](https://arxiv.org/abs/2406.04151)
*Summary.* Fourteen text environments behind a unified HTTP API, with trajectories and an evaluation suite.
*Contribution.* A platform rather than a method; makes environments interchangeable.
*Use for us.* We keep only its HTTP layer and the TextCraft server.

**AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making** (Xi et al., ICLR 2026 oral) — [arXiv:2509.08755](https://arxiv.org/abs/2509.08755) · [OpenReview](https://openreview.net/forum?id=ZgCCDwcGwn)
*Summary.* End-to-end RL (PPO, GRPO, REINFORCE++) on five environments, plus ScalingInter-RL, a curriculum on the interaction horizon.
*Contribution.* Qwen2.5-3B from 14 % to 75 % on TextCraft; 7B beats Gemini-2.5-Pro on some tasks; open-source framework.
*Use for us.* The system we replicate and the target score.
*Our reading of the reviews (OpenReview).* Oral deserved for the framework and the open-source effort; the algorithmic novelty is thin (ScalingInter is curriculum learning on one dimension, justified by gradient-norm curves rather than theory); hyperparameter "robustness" is oversold (36.8 to 39.1 across schedules); real-world results stay modest (WebArena 26 %, BrowseComp 8.5 %); GPT-5 as failure-mode judge is fragile; no 3B configuration, no epoch count, no curves, text and code disagree on the horizon (20 vs 30 turns). Contemporary ICLR 2026 posters with more principled credit assignment: ARPO (entropy-driven rollouts), iStar (implicit step rewards), RLVMR (verifiable meta-reasoning rewards), HiPER (hierarchical planning/execution).

**ADaPT: As-Needed Decomposition and Planning with Language Models** (Prasad et al., NAACL 2024) — [arXiv:2311.05772](https://arxiv.org/abs/2311.05772)
*Summary.* Recursive decomposition of tasks when the executor fails; introduces TextCraft.
*Contribution.* TextCraft as a crafting-tree environment where difficulty is the recipe depth.
*Use for us.* The origin of our depth stratification and of the "depth 4 wall".

**A Taxonomy of RL Environments for LLM Agents** (Lee Hanchung, blog 2026) — [blog post](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/)
*Summary.* Five pillars of an RL environment: tasks, harness, verifier, state, configuration.
*Contribution.* Practical guidance: programmatic verifiers over LLM judges, deliberate noise injection, turn-level evaluation.
*Use for us.* The vocabulary of Section 2 of the report ("the environment is what determines what can be learned").

---

## 4. Stability, exploration and the limits of RLVR

**The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** (Cui et al., 2025) — [arXiv:2505.22617](https://arxiv.org/abs/2505.22617)
*Summary.* Policy entropy is consumed as performance rises; the change in entropy is driven by the covariance between action probability and advantage.
*Contribution.* Empirical law R = −a·e^H + b and two entropy-preserving tricks (Clip-Cov, KL-Cov).
*Use for us.* Our reading grid for collapse and for the "who produces gradient" mechanism of the curricula.

**Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** (Yue et al., NeurIPS 2025) — [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)
*Summary.* RL-trained models win at small k but lose to the base model at large k on pass@k.
*Contribution.* The ceiling hypothesis: RL concentrates mass on paths the base model already had.
*Use for us.* Our oracle results (pass@20 of the base model 47 %, pass@1 after RL 82 %) put the hypothesis under tension without refuting it.

**ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs** (Liu et al., NVIDIA, NeurIPS 2025) — [arXiv:2505.24864](https://arxiv.org/abs/2505.24864)
*Summary.* Thousands of RL steps with periodic reset of the reference policy and of the optimizer.
*Contribution.* Long RL does expand the base model's boundary, against Yue et al.
*Use for us.* Closest precedent to our moving anchor; their trigger is validation stagnation, ours is a fixed period.

**Defining and Characterizing Reward Hacking** (Skalse et al., NeurIPS 2022) — [arXiv:2209.13085](https://arxiv.org/abs/2209.13085)
*Summary.* Formal definition: a proxy reward is hackable if increasing it can decrease the true reward.
*Contribution.* Shows that non-trivial unhackable proxies essentially do not exist.
*Use for us.* Names what our shaped rewards did: the policy learned to give up early and pocket the format bonus.

**Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters** (Snell et al., 2024) — [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
*Summary.* Compares parallel sampling with verifiers and sequential revision, per difficulty level.
*Contribution.* A compute-optimal allocation between the two, conditional on a difficulty oracle.
*Use for us.* The model for our compute-controlled comparison and for the "scaling agentic test-time compute" idea left open.

**Self-Improvement in Language Models: The Sharpening Mechanism** (Huang et al., 2024) — [arXiv:2412.01951](https://arxiv.org/abs/2412.01951)
*Summary.* Self-improvement works because a model verifies better than it generates; training sharpens toward its own high-likelihood outputs.
*Contribution.* Theory of when SFT-style and RL-style sharpening succeed.
*Use for us.* Theoretical companion of the entropy story: sharpening is the mechanism, exhaustion is the risk.

---

## 5. Parameter-efficient RL: LoRA and its dynamics

**LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., ICLR 2022) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
*Summary.* Train a low-rank product B·A added to frozen weights.
*Contribution.* Orders of magnitude fewer trainable parameters and optimizer memory.
*Use for us.* r=8 on seven projections: 0.5 % of parameters, optimizer 6 GB → 0.12 GB, checkpoints 12 GB → 0.18 GB.

**LoRA Without Regret** (Schulman et al., Thinking Machines, blog 2025) — [thinkingmachines.ai](https://thinkingmachines.ai/blog/lora/)
*Summary.* In RL, LoRA matches full fine-tuning even at very low rank, because policy gradient learns "about one bit per episode".
*Contribution.* Recipe: learning rate about 10× that of full fine-tuning, roughly independent of rank.
*Use for us.* The recipe collapsed in our multi-turn setting (fixed reference, geometric KL drift); the blog's remark on the "less favourable optimisation dynamics" of the B·A product is one of our two hypotheses.

**ReLoRA: High-Rank Training Through Low-Rank Updates** (Lialin et al., ICLR 2024) — [arXiv:2307.05695](https://arxiv.org/abs/2307.05695)
*Summary.* Periodically merge the adapter into the base weights, reset it, and partially reset the optimizer.
*Contribution.* Accumulated rank grows with the number of cycles; high-rank training at low-rank cost.
*Use for us.* What our "moving anchor" turned out to be, identified after the fact; exp46 separates the moving reference from the merge-and-restart.

---

## 6. Curriculum learning and autotelic agents

**Curriculum Learning** (Bengio et al., ICML 2009) — [ACM DL](https://dl.acm.org/doi/10.1145/1553374.1553380)
*Summary.* Present training examples in order of increasing difficulty.
*Contribution.* The original definition and the continuation-method reading.
*Use for us.* Our Depth curriculum is the closest to this definition.

**Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey** (Narvekar et al., JMLR 2020) — [arXiv:2003.04960](https://arxiv.org/abs/2003.04960)
*Summary.* A curriculum is a sequence of tasks or samples organised to learn a target problem.
*Contribution.* Taxonomy: what to order (tasks) versus how much to give the agent (means).
*Use for us.* Our three regimes span both levers: Horizon and Budget act on the means, Depth on the tasks.

**Automatic Curriculum Learning for Deep RL: A Short Survey** (Portelas et al., IJCAI 2020) — [arXiv:2003.04664](https://arxiv.org/abs/2003.04664)
*Summary.* Methods that shape the learning trajectory by proposing tasks adapted to the agent's capabilities.
*Contribution.* The definition of ACL we quote; learning progress as the standard signal.
*Use for us.* Bridge between our hand-set curricula and MAGELLAN.

**MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents** (Gaven, Carta, Romac, Colas, Lamprier, Sigaud, Oudeyer, 2025) — [arXiv:2502.07709](https://arxiv.org/abs/2502.07709) · recommended by Sylvain Lamprier, code available
*Summary.* An LLM agent predicts its own learning progress per goal from its representations and samples goals accordingly.
*Contribution.* Autocurriculum that generalises to unseen goals.
*Use for us.* Ported and run on TextCraft: it stopped proposing depth 4 as predicted, then collapsed at epoch 7 at G=16; an autocurriculum needs a stable learner.

**Grounding Large Language Models in Interactive Environments with Online RL (GLAM)** (Carta et al., ICML 2023) — [arXiv:2302.02662](https://arxiv.org/abs/2302.02662)
*Summary.* Online RL of an LLM policy in BabyAI-Text.
*Contribution.* Shows functional grounding through interaction; the lineage MAGELLAN builds on (with SAC-GLAM).
*Use for us.* Background for the autotelic line; not used directly.

**HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents** (Carta, Romac, Gaven, Oudeyer, Sigaud, Lamprier, 2025) — [arXiv:2508.14751](https://arxiv.org/abs/2508.14751)
*Summary.* A high-level policy picks reachable sub-goals; a low-level policy compiles them into skills.
*Contribution.* Open-ended skill acquisition with hierarchical autotelic control.
*Use for us.* Outlook: sub-goal choice as the next step after a fixed curriculum.

**BOSS: Bootstrap Your Own Skills** (Zhang et al., CoRL 2023) — [arXiv:2310.10021](https://arxiv.org/abs/2310.10021)
*Summary.* An LLM proposes chains of known skills of increasing length; the agent learns them.
*Contribution.* LLM-guided skill bootstrapping without task rewards.
*Use for us.* Outlook, same family as HERAKLES.

---

## 7. Self-improvement, synthetic tasks and environments

**STaR: Bootstrapping Reasoning With Reasoning** (Zelikman et al., NeurIPS 2022) — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
*Summary.* Generate rationales, keep those leading to the right answer, fine-tune, iterate.
*Contribution.* The simplest self-improvement loop; rationalisation for failed problems.
*Use for us.* The SFT-on-successful-trajectories idea we kept in the plan ("SFT + RL mix"), never run.

**Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST-EM)** (Singh et al., TMLR 2024) — [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
*Summary.* Expectation-maximisation view of self-training with a binary verifier.
*Contribution.* Self-generated data beats human data at scale; overfitting after a few iterations.
*Use for us.* Same as STaR, with the warning on iteration count.

**Self-Challenging Language Model Agents** (Zhou, Levine, Weston, Li, Sukhbaatar, NeurIPS 2025) — [arXiv:2506.01716](https://arxiv.org/abs/2506.01716)
*Summary.* The agent writes its own tasks as code (instruction, verifier, example solution, failure cases), then trains on them.
*Contribution.* The "Code-as-Task" filter removes impossible tasks and lax verifiers; beats PAE on multi-tool environments where exploration matters.
*Use for us.* The environment as an object to generate: the frontier we point to in the outlook.

**Why AI Systems Don't Learn and What to Do About It** (Dupoux, LeCun, Malik, 2026) — [arXiv:2603.15381](https://arxiv.org/abs/2603.15381)
*Summary.* Current systems lack the loop of a learner that observes and a learner that acts, orchestrated by a meta-controller.
*Contribution.* A cognitive-science reading of what "learning" should mean for AI systems.
*Use for us.* Frames our planning/interaction alternation idea (Appendix D of the report).

**DreamGym: Learning Agents in Synthesized Experience** (Chen et al., 2025) — [arXiv:2511.03773](https://arxiv.org/abs/2511.03773)
*Summary.* A reasoning-based world model synthesises transitions for agent RL.
*Contribution.* RL without touching the real environment for most of training.
*Use for us.* Outlook: environment generation as the follow-up of environment scaling.

**MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023) — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
*Summary.* Hierarchical memory management to exceed the context window.
*Contribution.* Non-parametric improvement through memory.
*Use for us.* Deliberately left aside; complementary to weight updates.

**A Survey of Self-Evolving Agents** (Gao et al., 2025) — [arXiv:2507.21046](https://arxiv.org/abs/2507.21046)
*Summary.* Map of self-evolution along what, when and how to evolve.
*Contribution.* Vocabulary and taxonomy of the field.
*Use for us.* Check that the project sits inside the field (it does: parametric evolution by RL on an environment).

---

## 8. Benchmarks and evaluation of agents

**Survey on Evaluation of LLM-based Agents** (Yehudai et al., 2025) — [arXiv:2503.16416](https://arxiv.org/abs/2503.16416)
*Summary.* Organises agent evaluation by capability (planning, tool use, self-reflection, memory) and by application (web, software, science, conversation), plus development frameworks and gym-like environments.
*Contribution.* Fig. 1 is a map of the field; trends: harder and live benchmarks, granular and cost-aware metrics, Agent-as-a-Judge.
*Use for us.* Where to look for ideas by neighbouring domain; the "environment turns as a cost" axis of our comparison echoes its cost-and-efficiency direction.

**ALFWorld** (Shridhar et al., ICLR 2021) — [arXiv:2010.03768](https://arxiv.org/abs/2010.03768)
*Summary.* Text counterpart of ALFRED household tasks.
*Contribution.* The standard sibling of TextCraft for text-based embodied agents.
*Use for us.* Comparison point in the AgentGym-RL results.

**WebArena** (Zhou et al., 2023) — [arXiv:2307.13854](https://arxiv.org/abs/2307.13854)
*Summary.* Self-hosted realistic websites (shopping, forum, code, maps) with functional evaluators.
*Contribution.* The realistic web benchmark where AgentGym-RL reaches 26 %.
*Use for us.* The environment class a Criteo shopping agent would train in.

**τ-bench** (Yao, Shinn, Razavi, Narasimhan, 2024) — [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)
*Summary.* Tool-agent conversations with a simulated user and policy compliance; pass^k measures reliability.
*Contribution.* Reliability as a metric, not only success.
*Use for us.* Same spirit as our pass@k oracle and "plateau, not peak" reporting.

**The Tool Decathlon (Toolathlon)** (Li et al., ICLR 2026) — [arXiv:2510.25726](https://arxiv.org/abs/2510.25726)
*Summary.* Long-horizon tool-use tasks over real software; difficulty measured by the number of turns of a reference model.
*Contribution.* A hard, realistic benchmark for multi-tool agents.
*Use for us.* Illustrates the "who defines difficulty" problem behind the autocurriculum question.

---

## 9. Agents for commerce and recommendation

**Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via RL** (Zhang et al., ICLR 2026) — [arXiv:2507.17842](https://arxiv.org/abs/2507.17842)
*Summary.* RL for shopping-behaviour simulation with a five-term weighted reward.
*Contribution.* Shows RL applied to e-commerce behaviour.
*Use for us.* Counter-example on reward design: coefficients from 0.1 to 0.5 and a ×1000 scale, ablated for presence only.

**Generative recommendation with LLMs: SIDReasoner and GRLM** — [arXiv:2601.06798](https://arxiv.org/abs/2601.06798) · [arXiv:2603.23183](https://arxiv.org/abs/2603.23183) (from Jérémy's reading group, April 2026)
*Summary.* Two ways to put items into an LLM: Semantic IDs (RQ-VAE tokens, aligned then trained with GRPO) versus Term IDs (five native keywords, plain SFT).
*Contribution.* SIDReasoner: explicit reasoning transfers across domains; GRLM: +50 % cross-domain recall, near-zero hallucination, scaling 0.6B → 14B.
*Use for us.* The recommendation side we chose *not* to pursue; a good embedding space may be worth more than an agent with sixty tools.

**Kimi K2.5: Visual Agentic Intelligence** (Kimi Team, 2026) — [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)
*Summary.* Frontier open model with sub-agent orchestration; tech report is explicit that agentic scores depend on tools, token budget and context management.
*Contribution.* Orchestration at scale and honest footnotes on evaluation conditions.
*Use for us.* Orchestration left out of scope; the footnote practice is one we followed (evaluation conditions stated).

**DeepShopper** (Criteo internal, Flavian & Imad) — *(internal Confluence link)*
A shopping assistant over the Amazon catalogue with several tools. Discussed in April as an application frame, set aside in favour of an academic environment.

---

## 10. Courses and resources

- **Stanford CS329A — Self-Improving AI Agents** (Chowdhery, Mirhoseini, fall 2025) — [cs329a.stanford.edu](https://cs329a.stanford.edu) · [video playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOyk-YyLIvnBC4TaOofeGzq)
- **Stanford CS224R — Deep Reinforcement Learning** (Finn) — [cs224r.stanford.edu](https://cs224r.stanford.edu) — MDP/POMDP, policy gradients, PPO; the basis of Appendix A of the report
- **Automated Design of Agentic Systems** (Clune, talk) — meta-learning of agent architectures, left out of scope
- **RLHF Book** (Lambert) — [rlhfbook.com](https://rlhfbook.com)
- **Monte Carlo Theory, Methods and Examples** (Owen, 2013) — [statweb.stanford.edu/~owen/mc](https://artowen.su.domains/mc/) — chapter 9, importance sampling; basis of the SNIS derivation (Appendix B)
- **Hugging Face LLM course**, **DeepLearning.AI agent courses** — onboarding, April 2026
- **TRL** — [github.com/huggingface/trl](https://github.com/huggingface/trl) · **vLLM** — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · **verl** (AgentGym-RL's trainer)

---

## 11. Further reading (not summarised)

Collected during the internship; relevant but not used in the report.

**Multi-turn agent RL, contemporaries of AgentGym-RL.** ARPO ([arXiv:2507.19849](https://arxiv.org/abs/2507.19849)) · RAGEN ([arXiv:2504.20073](https://arxiv.org/abs/2504.20073)) · WebRL ([arXiv:2411.02337](https://arxiv.org/abs/2411.02337)) · Search-R1 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516)) · Agent-R1 · iStar · RLVMR · HiPER · Demystifying RL in agentic reasoning ([arXiv:2510.11701](https://arxiv.org/abs/2510.11701)) · Synthetic data and multi-step RL for tool use ([arXiv:2504.04736](https://arxiv.org/abs/2504.04736)) · Procedural environment generation for tool-use agents ([arXiv:2506.11045](https://arxiv.org/abs/2506.11045)) · Agentic Context Engineering ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618)).

**Reward models and process supervision.** Let's Verify Step by Step ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) · Math-Shepherd ([arXiv:2312.08935](https://arxiv.org/abs/2312.08935)) · Self-Rewarding LMs ([arXiv:2401.10020](https://arxiv.org/abs/2401.10020)) · Position: hidden costs of RLVR ([arXiv:2509.21882](https://arxiv.org/abs/2509.21882)) · Rubric-based rewards, "Mock worlds, real skills" ([arXiv:2601.22511](https://arxiv.org/abs/2601.22511)).

**Reasoning and test-time compute.** Quiet-STaR ([arXiv:2403.09629](https://arxiv.org/abs/2403.09629)) · rStar-Math ([arXiv:2501.04519](https://arxiv.org/abs/2501.04519)) · Large Language Monkeys ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) · TTRL ([arXiv:2504.16084](https://arxiv.org/abs/2504.16084)) · Reflexion ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) · SFT memorizes, RL generalizes ([arXiv:2501.17161](https://arxiv.org/abs/2501.17161)) · Learning dynamics of LLM finetuning ([arXiv:2407.10490](https://arxiv.org/abs/2407.10490)).

**Synthetic tasks and self-play.** Proposer-Agent-Evaluator, PAE ([arXiv:2412.13194](https://arxiv.org/abs/2412.13194)) · AgentSynth · AutoPlay · SWE-RL self-play · IntellAgent ([arXiv:2501.11067](https://arxiv.org/abs/2501.11067)).

**Commerce and recommendation.** RecoWorld ([arXiv:2509.10397](https://arxiv.org/abs/2509.10397)) · ShoppingBench · AdNanny · eCeLLM · Agent4Rec, SimUser (simulation-centric recommendation).

**Models and post-training reports.** Kimi k1.5 ([arXiv:2501.12599](https://arxiv.org/abs/2501.12599)) · Kimi K2 ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)) · Tülu 3 ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) · Qwen3 technical report ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) · Magistral ([arXiv:2506.10910](https://arxiv.org/abs/2506.10910)) · DPO survey ([arXiv:2410.15595](https://arxiv.org/abs/2410.15595)) · SimPO ([arXiv:2405.14734](https://arxiv.org/abs/2405.14734)).

**Read and set aside.** *A narrowing window to understand AI* (interactional and behavioural opacity taxonomy; the loss-of-control thesis was not found convincing) · MCP / A2A protocols (engineering, not research) · POET ([arXiv:1901.01753](https://arxiv.org/abs/1901.01753)) as background for environment co-evolution.
