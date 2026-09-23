# archive/scripts/ (2026-07-16)

| Script | Raison |
|---|---|
| `run_exp7.1_chain.sh` | orchestration one-shot terminée (eval exp7 ckpt564 → smoke → lancement exp7.1, juin 2026) |
| `merge_lora_originaux/` | scripts de merge de la lignée exp10.x, sauvés de `logs/` (gitignoré) — voir son README ; remplacés par `src/utils/merge_lora.py` |

## Ajouts du 2026-09-23

| Fichier | Rôle | Pourquoi archivé |
|---|---|---|
| `run_curriculum_staged.sh` | curriculum par stages de profondeur, un run par stage (exp9) | remplacé par `--depth-schedule-epochs` / `--depth-schedule-auto` dans un seul run |
| `run_benchmark_baselines.sh` | évaluations de référence en série (mai 2026) | one-shot terminé |
| `backup_latest_ckpt.sh` | copie périodique du dernier checkpoint vers le home | les checkpoints vont désormais directement sur le home (`--output-root saves/trl_grpo_ckpt`) |
| `after_exp43_start_queue.sh`, `start_queue_when_env_ready.sh` | guetteurs qui démarraient la file après un run ou après la reconstruction de l'env | one-shot, liés à des incidents de septembre |
