# Analyse approfondie — Oracle pass@k (base vs best RL)

**Expérience** : « oracle » / best-of-K (discussion 2026-06-16). Pour chaque item du test
set TextCraft (100 items), on tire **N=20 trajectoires indépendantes** (température 1.0,
≤30 tours) et on mesure le **pass@k** (estimateur non biaisé de Chen et al. 2021) en
fonction de k=1→20, globalement et **par profondeur de craft (depth)**.

**But** : estimer la **marge exploitable par le RL**. pass@1 = ce que le modèle fait de
façon fiable ; pass@20 = ce qu'il sait faire « parfois » (borne supérieure optimiste, un
oracle qui garderait la meilleure des 20 tentatives). L'écart pass@20 − pass@1 = marge RL.
Le pass@k par depth tranche : **mur de fiabilité** (pass@20 élevé, RL-able) vs **mur de
capacité** (pass@20 ≈ 0 → aucune graine pour le RL, cf. Dr. GRPO).

Modèles comparés :
- **base** : `models/Qwen2.5-3B-Instruct` (non entraîné). Données : `runs/oracle_baseline/oracle_passk.json`.
- **best** : `saves/trl_grpo/exp7.3_grpo_pur_we6_best` (GRPO, meilleur checkpoint = 32/100). Données : `runs/oracle_best32/oracle_passk.json`.

Distribution du test set : 31 items depth 1, 41 depth 2, 25 depth 3, 3 depth 4.
Figure : `passk_curves.png` (courbes pass@k par depth, base vs best).

---

## 1. pass@k global

| k | base | best | Δ (RL) |
|---|---|---|---|
| 1 | 10.3 | 32.0 | +21.7 |
| 2 | 17.1 | 41.3 | +24.2 |
| 5 | 28.6 | 52.3 | +23.7 |
| 10 | 37.8 | 59.1 | +21.3 |
| 20 | 47.0 | 65.0 | +18.0 |

Le RL améliore à la fois pass@1 (+22) **et** pass@20 (+18) → il a déplacé la frontière
d'atteignabilité, pas seulement la fiabilité. Mais la marge globale restante du best est
encore large : 32 (fiable) vs 65 (atteignable).

---

## 2. pass@k par depth (le cœur)

**Baseline Qwen2.5-3B :**
| depth | items | @1 | @2 | @5 | @10 | @20 |
|---|---|---|---|---|---|---|
| 1 | 31 | 27 | 44 | 69 | 84 | 94 |
| 2 | 41 | 4 | 8 | 17 | 29 | 44 |
| 3 | 25 | 0 | 0 | 0 | 0 | **0** |
| 4 | 3 | 0 | 0 | 0 | 0 | **0** |

**Best (RL, 32) :**
| depth | items | @1 | @2 | @5 | @10 | @20 |
|---|---|---|---|---|---|---|
| 1 | 31 | 79 | 92 | 99 | 100 | **100** |
| 2 | 41 | 18 | 30 | 50 | 65 | **78** |
| 3 | 25 | 1 | 2 | 3 | 6 | **8** |
| 4 | 3 | 0 | 0 | 0 | 0 | **0** |

---

## 3. Distribution des succès/20 par item (la « forme » de la marge)

Nombre d'items dans chaque tranche de `succès/20` :

| depth | 0/20 | 1–5/20 | 6–15/20 | 16–20/20 | | (base → best) |
|---|---|---|---|---|---|---|
| 1 base | 2 | 13 | 16 | 0 | | la masse se décale vers la droite |
| 1 best | 0 | 0 | 12 | 19 | | (19 items quasi toujours résolus) |
| 2 base | 23 | 17 | 1 | 0 | | 23 items jamais résolus |
| 2 best | 9 | 21 | 11 | 0 | | seulement 9 jamais ; 11 souvent |
| 3 base | 25 | 0 | 0 | 0 | | **tous à zéro** |
| 3 best | 23 | 2 | 0 | 0 | | 2 items débloqués (sporadiques) |
| 4 base | 3 | 0 | 0 | 0 | | tous à zéro |
| 4 best | 3 | 0 | 0 | 0 | | **toujours tous à zéro** |

Lecture : à depth 2, le best a 21 items en « 1–5/20 » (résolus mais rarement) → c'est la
réserve typique que le RL peut amplifier. À depth 3, seuls 2 items quittent le « 0/20 », et
de justesse (3/20 et 1/20).

---

## 4. Frontière d'atteignabilité (items résolus ≥ 1 fois / 20)

| depth | base | best | Δ |
|---|---|---|---|
| 1 | 29/31 | 31/31 | +2 |
| 2 | 18/41 | 32/41 | **+14** |
| 3 | 0/25 | 2/25 | +2 |
| 4 | 0/3 | 0/3 | 0 |

Le RL a rendu **14 items depth-2 supplémentaires** atteignables (et 2 depth-3 qui ne
l'étaient jamais). C'est une vraie expansion de capacité, pas qu'une meilleure fiabilité.

---

## 5. Marge RL restante sur le best (pass@20 − pass@1)

| depth | pass@1 | pass@20 | marge |
|---|---|---|---|
| 1 | 79 | 100 | +21 (presque saturé) |
| 2 | 18 | 78 | **+60 (énorme)** |
| 3 | 1 | 8 | +7 (quasi nul) |
| 4 | 0 | 0 | 0 (rien) |

---

## 6. Conclusions

1. **Depth 2 = problème de FIABILITÉ → terrain du RL.** Le best sait résoudre 78% des
   items depth-2 « parfois » mais seulement 18% de façon fiable (marge +60). C'est là que
   davantage / un meilleur RL doit payer. Priorité n°1 pour pousser le score.

2. **Depth 3-4 = mur de CAPACITÉ → le RL pur est insuffisant.** Le best ne dépasse jamais
   8% à depth 3 (4 succès sur 500 tirages, 2 items) et **0% à depth 4** (0/60). La baseline
   est à **0 partout** (0/500 et 0/60). Point décisif (**Dr. GRPO**) : pass@k = 0 ⇒ aucune
   trajectoire gagnante ⇒ **aucun gradient** pour GRPO ⇒ le RL pur ne peut **pas** bootstrap
   ces profondeurs. → il faut **injecter** des succès : curriculum depth 1→2→3, SFT sur
   traces expertes, CoT structuré, ou un meilleur modèle de base.

3. **Nuance importante** : le RL n'a pas qu'amélioré la fiabilité, il a **légèrement déplacé
   la frontière** à depth 3 (0→2 items atteignables). Donc depth 3 n'est pas un mur infini —
   il est juste à la limite de ce que ce modèle peut produire. Un curriculum qui amène le
   modèle au niveau depth-2 fiable pourrait suffire à faire entrer depth-3 dans la zone
   « sporadique mais non nulle » exploitable par le RL.

4. **Borne supérieure optimiste** : pass@20 surestime le gain RL réel (oracle parfait,
   exploration imparfaite en pratique, risque de casser depth 1-2 en poussant depth 3).
   À lire comme un *plafond de potentiel*, pas une prédiction.

---

## Reproduire

```bash
# serveur vLLM sur le modèle, puis :
python src/eval/eval_oracle.py --model <chemin_servi> --run-name <name> --n-samples 20
```
Artefacts : `runs/<name>/oracle_passk.json` (pass@k global + par depth + successes_per_item).
Détail du script : `docs/hebdo/19juin/session_2026-06-16.md` §5 ; lecture du json : idem.
