# archive/train/ — training variants that are no longer run (refactor of 2026-09-23)

| File | What it was | Why it left `src/` |
|---|---|---|
| `magellan.py` | Port of the MAGELLAN autocurriculum (Gaven et al. 2025): tasks sampled by predicted learning progress (exp34, exp42) | Collapsed at G=16 (report Appendix E); parked for lack of time. The weighted sampler and the depth curricula it contained now live in `src/train/sampling.py`. |
| `snis.py`, `train_grpo_snis.py` | Recombination of sub-trajectories by self-normalised importance sampling (exp17) | Collapsed: multi-turn sub-trajectories cannot be spliced without causal alignment (report Appendix B). |
| `rollout_plan.py` | "Plan mode": the model writes the whole action sequence in one completion, replayed in the environment (exp21) | Single-turn planning does not transfer to interactive control (report Appendix D). |

These files import modules that have since moved (`src.train.schedules` was split into
`horizon_schedules`, `lr_schedules`, `lr_adaptive`, `kl_anchor`) and are kept for reading, not
for running. To run them exactly as they were, check out the branch
`backup/pre-refacto-2026-09-23`.
