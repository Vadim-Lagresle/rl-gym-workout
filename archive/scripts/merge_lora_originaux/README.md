# Scripts de merge LoRA originaux (lignée exp10.x)

Scripts jetables écrits pendant les runs warm-start, sauvés ici depuis `logs/`
(gitignoré — ils auraient été perdus au premier clone). Ils documentent la
lignée exacte du meilleur modèle du projet :

| Script | Base | Adapter | Sortie (modèle mergé) |
|---|---|---|---|
| `merge_exp10p5.py` | Qwen2.5-3B + 35 % (exp10.3) | best exp10.5 (51/100, step 368) | `qwen25_3b_exp10p5_step368_51pct` |
| `merge_exp10p7.py` | merge 51 % | best exp10.7 (53/100, step 92) | `qwen25_3b_exp10p7_step92_53pct` |
| `merge_exp10p8.py` | merge 53 % | best exp10.8 (58 pic / 54 re-éval, step 368) | `qwen25_3b_exp10p8_step368_58pct` |

**Remplacés par** [`src/utils/merge_lora.py`](../../../src/utils/merge_lora.py)
(version paramétrée `--base --adapter --out`) — 2026-07-16.
