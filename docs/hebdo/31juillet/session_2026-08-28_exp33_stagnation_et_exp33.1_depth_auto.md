# Session 2026-08-28 — mort d'exp33 (purge + stagnation), paliers auto-déclenchés, lancement exp33.1

## Constat sur exp33 (paliers calendaires)

Morte dans une purge pod à ~01:48 (ep ~22/80) — mais le diagnostic de Vadim
précède la purge : le run n'allait pas remonter. Deux défauts symétriques des
paliers statiques, tous deux documentés pour le rapport (§4.3) :
1. **Palier saturé maintenu** : reward train à 1.0 des ep ~6 à 12 sur d<=1 —
   8 epochs sans le moindre gradient GRPO, entropie descendue à ~0.19.
2. **Palier bloquant sans échéance utile** : sur d<=2, reward médian FIGÉ à
   0.25 pendant 10 epochs (ep 12→22), éval 26-29, entropie saine (0.3→0.49)
   mais aucune progression.
Artefacts : best 31/100 @ step 282 (+optimizer) et chaîne d'ancres sur le home.
Job 44 parqué dans `skipped/` (ne pas relancer la version statique).

## La règle auto-déclenchée (décision Vadim) et son implémentation

**Règle : palier suivant si le reward train MOYEN sur la dernière epoch
complète ≥ 0.8 ; sinon passage forcé après 10 epochs au palier.**

- `magellan.py` : nouvelle classe `DepthAutoScheduleProvider` — masque uniforme
  depth<=palier identique au provider calendaire ; l'avancement vit dans
  `on_log` (duck-typing TrainerCallback) : fenêtre glissante de
  `steps_per_epoch` rewards (93 à G=16), vidée à chaque passage (cooldown
  d'une epoch de mesures fraîches), butée à la profondeur max.
- `train_grpo.py` : flags `--depth-schedule-auto`, `--depth-auto-threshold`
  (0.8), `--depth-auto-max-epochs` (10) — exclusifs avec le schedule
  calendaire et le goal-sampler ; `steps_per_epoch = len(rows)·G/64`.
- Selftest (test 11) : pas de passage sur fenêtre incomplète, passage au
  succès, cooldown, passage forcé au cap, butée finale — VERT sur l'env
  reconstruit. `add_callback` de transformers accepte le duck-typing (vérifié
  dans la source).

## Relance

Env v2 + Qwen reconstruits (~10 min, wheel), runner relancé, serveur TextCraft
redémarré par ensure_stack. **exp33.1_depth_auto lancée à 07:58** (job 44b) :
selftests pré-vol passés, `[depth-auto] seuil 0.8 sur 93 steps, cap 10
epochs/palier`, palier d<=1 (109/374), ancre mobile armée, 114 Go GPU.
Attendu : premier passage de palier vers l'ep ~3-4 (exp33 atteignait 0.97 de
reward dès l'ep 2-4) — soit ~8 epochs gagnées d'entrée vs la version
calendaire.

## File au 28/08 08:15

exp33.1 (EN COURS) → exp34_magellan → exp35_budget → exp36.1_anchor12 →
exp37_fewshot. Gel 2/09 : ~5 jours — arrêts au plateau obligatoires ;
exp36.1/exp37 sacrifiables. Rapport : v1 complète (abstract → conclusion),
figures et analyses sur logs au gel.
