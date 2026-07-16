# archive/analysis/ — pourquoi chaque script est ici (2026-07-16)

| Script | Raison | Remplaçant |
|---|---|---|
| `check_context_length.py` | redéfinissait son propre `analyze_episode` ; le comptage de tokens et l'alerte fenêtre de contexte existent dans `analyze_eval.py --tokenizer <path>` | `src/analysis/analyze_eval.py` |
| `plot_exp10_collapse.py` | figure one-shot (anatomie du collapse exp10, données parsées de `logs/exp10_grpo_lora_b200.log`) — figure déjà générée : `docs/dashboard/10_exp10_collapse.png` | — (historique) |
| `plot_reasoning_pur.py` | figure one-shot exp16 à scores codés en dur (antérieure aux sweeps qui ont affiné ces chiffres) — `docs/dashboard/09_reasoning_pur_par_depth.png` | — (historique) |
