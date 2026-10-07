# Session 2026-08-24 — arrêt exp31.1 (divergence), clôture phase 1, lancement exp32 (curriculum horizon)

## Décision

L'ablation d'ancre (phase 1) n'apporte plus rien : exp31.1 (ancre fixe + β0.0001)
a divergé à l'epoch ~9. On arrête, on documente, et on passe à la **phase 2
(curriculums, exp32-35)** — le cœur du plan (`docs/PLAN_EXPERIENCES.md`). On
reviendra sur l'ablation seulement si le calendrier le permet.

## Bilan exp31.1 (run du 24/08, 09:47 → 16:47, tué au step ~423/9200)

- Trajectoire éval : 13 → 16 → 22 → 24 → 32 → **34 (step 282, ep ~6)** → 25 → 25 → 26.
- Divergence KL runaway : kl 0.003→0.213 jusqu'à l'ep 6, puis **906 (ep 7) →
  1572 (ep 8) → 2.25×10¹⁴ (ep 9)** ; entropie 1.3 → 0.02 ; completions moyennes
  8k tokens (génération détruite), loss ~10¹¹. Même signature que le collapse
  β0.001 d'exp24, en plus violent.
- Sauvegardes (home) : best 34 + optimizer (`saves/trl_grpo/exp31.1_r8_fixedanchor_b00001_best`),
  dernier checkpoint copié de /tmp avant le kill
  (`..._ckpt376` — déjà en divergence, kl ~1572, gardé pour analyse seulement).

## Verdict phase 1 — le carré ancre × β est complet (figure 1 du rapport)

| | β 0.01 | β 0.001 | β 0.0001 |
|---|---|---|---|
| **Ancre fixe** | exp30 : pic 39 (ep 7) puis collapse ep ~11 | exp24 r16 : pic 35 puis collapse | exp31.1 : pic 34 puis DIVERGENCE ep ~7-9 |
| **Ancre mobile /4 ep** | exp25/25.2 : **65-73, stable** | exp31 : 53, stable (plateau, arrêté ep ~60) | — |

**Lecture** : l'ancre mobile est *nécessaire* (aucune valeur de β ne sauve
l'ancre fixe) et *suffisante* à β0.001 comme à β0.01 pour empêcher le collapse ;
β0.01+mobile reste la recette (53 vs 65-73). La KL n'est un stabilisateur
efficace que vers une référence *proche* de la politique courante.

Au passage, résultat majeur enregistré : **exp25.2_from65 a terminé ses 60 epochs
proprement le 21/08 avec un best de 73/100 (steps 2491/2632/2726) — RECORD du
projet**, à 2 points de l'objectif papier (75).

## Actions de la session

1. Copie du dernier ckpt /tmp → `saves/trl_grpo/exp31.1_r8_fixedanchor_b00001_ckpt376` (184 Mo).
2. `kill` du train (SIGTERM, PID 127111) → le job 42b s'est terminé proprement
   (exit 0), `append_results.sh` a rempli `docs/RESULTS.md`, le runner de file a
   enchaîné **automatiquement** sur le job 43.
3. **exp32_horizon lancée à 16:47** (job 43) : recette exp25 + horizon
   progressif `'10:0,20:8,30:20'`, N=8. **AVORTÉE à 16:59 sur décision de
   Vadim** (avant la 1re éval — aucun artefact, log archivé en
   `logs/exp32_horizon.log.aborted_v1_n8`), deux corrections :
   - **horizon trop rapide** → paliers repoussés : `'10:0,20:15,30:30'`
     (20 tours à l'epoch 15, 30 à l'epoch 30) ;
   - **flags GPU fossiles** → application du verdict du bench exp25.1 (20/08,
     resté lettre morte dans les jobs 43-46) : **N=16 rollouts/prompt** à
     64 traj/step constant (4 prompts/step, GA=64) + `--vllm-gpu-util 0.5`
     = config D du bench, 46.9 s/it vs 57.1 (N=8/0.17) — meilleure estimation
     d'avantage GRPO (groupes mixtes ~2× plus fréquents sur les items
     difficiles) pour un temps MOINDRE.

   **Relancée en v2 à 16:59 (job 43b)**. Vérifié au démarrage : schedule
   `[(0,10),(15,20),(30,30)]`, `64 traj/step, 4 prompts/step, N=16`, ancre
   mobile armée, rollouts plafonnés à 10 tours, premiers rewards +1.
   ⚠ Caveat scientifique assumé : exp32 diffère maintenant de la baseline
   exp25 par DEUX facteurs (horizon + N) — l'effet propre du curriculum
   d'horizon n'est plus isolé.
4. **Harmonisation (décision Vadim, même jour)** : les jobs 44-46 (exp33/34/35)
   passent aussi à N=16 + gpu-util 0.5 → phase 2 homogène. Nouveau job
   **47_exp36_n16** ajouté en queue de file : recette exp25 stricte SANS
   curriculum, seul delta N=16 — effet propre de N vs exp25 ET contrôle qui
   lève le double-delta d'exp32. Promue le soir même en tête de file (job
   renommé 47→43z : passera JUSTE APRÈS exp32, avant les autres curriculums)
   — exp32 monte vite (67/100 @ step 2679, ep ~29 à 93 steps/ep N=16) et il faut trancher tôt si
   ça vient du curriculum d'horizon ou du N. On reste à N=16 (pas 32) : la contrainte
   n'est pas le GPU (concurrence fixée à 64 épisodes quel que soit N ; util
   0.6 s'est montré CONTRE-productif au bench : 61.5 s/it) mais statistique —
   à 64 traj/step, N=32 = 2 items/update (gradient dominé par 2 tâches) ;
   monter GA à 128 doublerait le s/it (~2× moins d'epochs/jour avant le gel).
4. Docs : ligne exp25.2 + section phase 1 + repères dans `runs/INDEX.md`,
   `config.yaml` d'exp31.1, interprétation dans `docs/RESULTS.md`.

## File restante (runs/queue/)

44_exp33_depth → 45_exp34_magellan → 46_exp35_budget1024. Rappels du plan :
exp33 suppose `data/train/textcraft_train_with_depth.json` à jour ; exp34
(autocurriculum ALP) a du code à valider à sec AVANT que le job n'arrive en tête
de file ; exp32 est en « 200 ep, arrêt manuel » — c'est donc un arrêt par Vadim
qui déclenchera exp33.

## État disque (24/08, après la session)

Home 35 Go : 14 Go utilisés / **21 Go libres**. `saves/trl_grpo/` = 4.3 Go :

| Artefact | Contenu | Taille |
|---|---|---|
| `exp25.2_from65_best` | **best 73 @ 2726 (RECORD)** + optimizer 115 Mo | 183 Mo |
| `exp25.2_from65_anchors` | chaîne cycles 1-15 + chain.jsonl | 858 Mo |
| `exp25_r8_anchor4ep_best` | best 65 @ 4794 — **SANS optimizer** (perdu purge 20/08) | 69 Mo |
| `exp25_r8_anchor4ep_anchors` | chaîne cycles 1-27 | 1.6 Go |
| `exp31_r8_anchor_b0001_best` | best 53 @ 2679 + optimizer | 183 Mo |
| `exp31_r8_anchor_b0001_anchors` | chaîne cycles 1-15 | 858 Mo |
| `exp30_r8_fixedanchor_best` | best 39 @ 329 + optimizer | 183 Mo |
| `exp31.1_..._best` | best 34 @ 282 + optimizer | 183 Mo |
| `exp31.1_..._ckpt376` | dernier ckpt (divergent, analyse seulement) | 184 Mo |
| `saves/wheels/` | wheel flash-attn (rebuild env v2 en ~10 min) | 232 Mo |

`saves/keep_best/` n'existe plus (archivé 17/08, README dans `docs/archive/`).
`/tmp` : Qwen2.5-3B présent ; `/tmp/trl_grpo_runs/` contient encore le dossier
exp31.1 (volatil, ckpt-376 déjà copié sur le home — supprimable) et accueille
les checkpoints périodiques d'exp32. Budget best exp32 : ~183 Mo
(adapter+optim LoRA r8) + ~860 Mo de chaîne d'ancres → large.
