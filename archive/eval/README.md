# archive/eval/ — pourquoi chaque script est ici (2026-07-16)

| Script | Raison | Remplaçant |
|---|---|---|
| `eval_baseline.py` | patron vLLM in-process abandonné (la stack est serveur vLLM + KV cache) ; référence des chemins `scratch/` disparus | `src/eval/eval_textcraft.py --backend vllm` |
| `eval_lora.py` | même patron in-process ; l'éval LoRA se fait désormais en mergeant l'adapter (`src/utils/merge_lora.py`) puis éval standard | `merge_lora.py` + `eval_textcraft.py` |
| `compare_runs.py` | **cassé** (`REPO_ROOT` résolvait vers `src/` → cherchait `src/runs/`, toujours vide) et limité aux runs v2/v3 de mai | `src/analysis/compare_dashboard.py` |
| `view_log.py` | chemin de logs codé en dur sur `src/eval/eval_logs` (n'existe plus) | lire le JSON, ou `analyze_eval.py` |
| `analyze_gemini.py` | analyse par depth recalculée via subprocess conda (le fichier depth JSON existe maintenant) ; bug `DATASET_PATH` non défini | `src/analysis/compare_dashboard.py` |
| `summarize_eval.py` | sous-ensemble strict d'`analyze_eval.py` (CSV + taux de succès) | `src/analysis/analyze_eval.py --eval-dir <dir>` |
| `auto_eval.sh` | workflow d'éval verl 4-GPU (stack legacy exp1-6) | stack TRL : `eval_textcraft.py` |
| `eval_fullft.py` (arrivé à l'étape 4) | boucle épisode HF dupliquée | `eval_textcraft.py --backend hf` |
| `eval_vllm.py` (étape 4, renommé) | devenu le tronc de `eval_textcraft.py` | `eval_textcraft.py --backend vllm` |
