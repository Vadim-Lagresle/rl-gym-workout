# archive/setup/ — superseded environment recipes

| File | What it was | Replaced by |
|---|---|---|
| `setup.sh`, `requirements-agentgym-rl.txt` | Conda envs of the first GCP VM (verl, AgentGym-RL, spring 2026) | `setup/setup_agentgym_rl_v2.sh` |
| `setup_trl_b200.sh`, `requirements-trl-b200.txt` | TRL 1.3 / vLLM 0.9.1 env for glibc 2.28 (patched vLLM), until July 2026 | `setup/setup_agentgym_rl_v2.sh` (TRL ≥ 1.9, vLLM ≥ 0.25, glibc 2.39) |
| `setup_verl_h100.sh` | Pinned recipe to run the paper's verl code on 8×H100 (Nebius, August 2026, never obtained) | — |
