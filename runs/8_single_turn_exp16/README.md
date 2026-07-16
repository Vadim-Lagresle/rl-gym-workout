# exp16 — planification single-turn (« reasoning pur »)

## Le protocole (2 phases, aucun entraînement)

exp16 isole la capacité de **planification one-shot** du modèle de sa capacité de
**contrôle réactif multi-tour** :

1. **Phase 1 — plan** (`plans/`, script `src/eval/single_turn/collect_single_turn_plans.py`) :
   recettes + goal → le modèle rédige un plan de craft en texte libre. Un seul appel LLM,
   pas d'interaction avec l'environnement.
2. **Phase 2 — extraction + replay** (`replay_logs/`, script
   `src/eval/single_turn/replay_single_turn_plans.py`) : un extracteur LLM (T=0)
   convertit le plan en actions JSON, rejouées **déterministiquement** dans TextCraft.
   Métrique = pass@1 « oracle » du plan.

## Les 4 axes de variation (lisibles dans les noms de dossiers)

| Axe | Valeurs | Marqueur |
|---|---|---|
| Modèle | Qwen2.5-3B base · Qwen3.5-4B (HF) · best RL exp10.8 | `3b`/`base` · `4b`/`qwen35_4b` · `best58`/`b58` |
| Température phase 1 | 0.7 · 1.0 · balayage 0.0→1.0 pas 0.1 | `temp07` · `temp1` · `t00`…`t10` |
| Extracteur | informé (recettes+goal) vs aveugle | préfixe `blind_` |
| Protocole | point isolé (1 tirage) vs balayage systématique | préfixe `sweep_` |

⚠ `best58` désigne le best d'exp10.8 (pic mono-tirage 58/100 au step 368) ; sa valeur
honnête en re-éval indépendante est **54/100** (`5_lora_warmstart/verif_best58_multitour`).

## Sous-dossiers

- **`core/`** — 6 runs à température unique (plans + replay complets, config.yaml présent) :
  `exp16_single_turn_reasoning` (3B, T0.7 → 20), `exp16_qwen35_4b` (4B, T0.7 → 54),
  `exp16_3b_temp1` (14), `exp16_4b_temp1` (49), `exp16_best58_temp07` (19), `exp16_best58_temp1` (14).
- **`blind/`** — 5 ablations : mêmes plans, extracteur privé des recettes et du goal.
  Écart informé−aveugle = part du score « réparée » par l'extracteur : 3B 20→12 (~40 %),
  4B 54→46 (~15 %) → les plans du 4B sont bien plus auto-suffisants.
- **`sweep_base/`, `sweep_b58/`, `sweep_4b/`** — balayage de température (1 tirage/T,
  extracteur informé, pas de config.yaml ; scores dans `docs/RESULTS.md` §sweeps et
  recalculables depuis `replay_logs/`). `sweep_4b_t07` est **incomplet** (35 plans,
  0 replay — avorté pour libérer le GPU).

## ⚠ Verdicts périmés dans les configs de core/

Les sweeps (juillet, référence actuelle) **invalident** deux conclusions écrites dans
les `config.yaml` de `core/` (audit : `docs/hebdo/10juillet/audit_exp16_reasoning_pur.md`) :

1. « Pénalité de température 0.7→1.0 » (config `exp16_3b_temp1`) → **faux** : la bande
   T=0.0→1.0 est plate (bruit mono-tirage ±5) ; le 20 à T0.7 était un tirage chanceux.
2. « Le GRPO n'a rien transféré à la planification (19≈20) » (config `exp16_best58_temp07`)
   → **faux** : sur 11 températures, best58 (moy 19.5) > base (moy 12.3), +7.2 pts,
   test des signes p=0.002. Transfert réel mais **modeste** (+7 single-turn vs +36
   multi-tour) : le gain RL est surtout réactif, pas de la planification a priori.

Conclusions d'ensemble : le mur depth 4 est un mur de *planification* (0/3 partout,
même Qwen3.5-4B) ; et le multi-tour (77) > oracle single-turn (54) pour le 4B — le
feedback de l'environnement rend la tâche PLUS facile que la planification parfaite.
