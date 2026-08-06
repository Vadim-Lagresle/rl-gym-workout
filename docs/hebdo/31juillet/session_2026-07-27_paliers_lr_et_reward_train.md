# Session 2026-07-27 — Paliers LR/beta (exp20) : verdict et piste reward-adaptive

## Contexte

`exp20_staged_lr_v2` (GRPO LoRA, few-shot k=10, depuis zéro) a appliqué la recette
d'exp19.2 (LR 5e-6, beta 0.01) avec un **schedule par paliers** :
LR et beta divisés par **3 toutes les 3 epochs** (`StagedLrBetaCallback`), sur
21 epochs (~966 steps). Résultat final : **58/100 @ step 800** (éval multi-tour
périodique), sans collapse KL (contrairement à exp19.2 à LR constant 5e-6, best
45/100 puis dérive).

Comparaison directe avec exp19.2 (même recette, LR constant) :
- exp19.2 : 45/100 @ step 300 → collapse KL ~step 350-450 ;
- exp20 : progression plus lente mais **meilleur pic** (58 vs 45) et run complet.

## Verdict sur les paliers LR ÷3 / 3 epochs

**Ce qui marche** : le schedule **stabilise** l'entraînement (pas de runaway KL)
et permet d'**aller chercher de meilleures perfs test** qu'un LR constant à 5e-6
sur la durée du run.

**Ce qui pose question** : en regardant le **reward train** (`rewards/textcraft_reward/mean`
sur wandb / logs de rollout), l'impression est que **chaque palier casse une dynamique
d'apprentissage en cours** — le reward moyen semble repartir ou stagner juste après
la division du LR, comme si on « gelait » le modèle au mauvais moment plutôt qu'au
bon. Les paliers sont **pilotés par le temps** (epochs), pas par l'état d'apprentissage :
on divise à t=3, 6, 9… epochs qu'il y ait encore de la marge de progrès ou non.

Les 7 transitions `[lr-beta-stage]` d'exp20 (epochs 0→3→6→…→18) coïncident avec
des phases où le Pass@1 test continue de monter par à-coups (pic 58 au step 800,
alors que le LR est déjà à ~7e-9) — donc le schedule n'est pas catastrophique ;
mais le signal **train** suggère qu'on paie un coût en dynamique à chaque coupure.

## Piste : LR adaptatif au reward train (pas au calendrier)

**Idée** : faire **décroître le LR en fonction du reward train**, plutôt qu'au
calendrier fixe.

Critère proposé (à affiner) :
- calculer une **moyenne roulante** du reward train (mean sur les rollouts du step) ;
- estimer sa **dérivée** (pente sur la fenêtre) ;
- si la dérivée **décroît** (ou reste ≤ 0) pendant **plusieurs steps consécutifs**,
  **diviser le LR** (et éventuellement beta, comme exp20) d'un facteur fixe (ex. ÷3).

**Fenêtre de la moyenne roulante** : à calibrer pour être à la fois :
- assez longue pour lisser le bruit du batch GRPO (64 trajectoires/step, tirage
  aléatoire des items → variance inter-steps) ;
- assez courte pour réagir avant qu'une phase d'apprentissage ne s'épuise.

Piste de dimensionnement : **~¼ d'epoch** comme fenêtre. Sur exp20/exp21,
374 items / 8 prompts/step ≈ **47 steps/epoch** → fenêtre ≈ **12 steps**,
soit ~768 trajectoires dans la moyenne (12 × 64). Ça devrait donner une courbe
plus cohérente et **moins sensible à la difficulté des items tirés** qu'une
moyenne sur 1-2 steps.

**Seuil « plusieurs steps »** : même ordre de grandeur qu'à définir (ex. dérivée
≤ 0 sur 2 fenêtres consécutives, ou pente < ε pendant N steps).

**Implémentation (codée le soir même — exp22)** : `RewardAdaptiveLrCallback`
dans `schedules.py`, branché via `--lr-adaptive`. Paramètres retenus après
discussion : fenêtre roulante **1 epoch** (~47 steps ≈ 3 000 trajectoires,
std de la moyenne ≈ 0.009 — préféré au ¼ d'epoch envisagé le matin, trop
bruité pour servir de référence best), **check toutes les ½ epochs**,
**patience 3** checks consécutifs sous le best (tolérance eps 0.005), coupe
**÷3** (LR et beta, comme exp20), best réinitialisé après coupe (cooldown
naturel de 1.5 epoch minimum entre deux coupes), plancher 1e-9. Application
LR/beta factorisée dans `apply_lr_beta` (partagée avec `StagedLrBetaCallback`).
Validé par selftest CPU (`python -m src.train.schedules`) + smoke GPU 2 steps.
Le « restart du best ckpt à chaque coupe » évoqué par Vadim n'est pas
implémenté (sélection test-set dans la trajectoire + restauration mi-run
risquée pour un run de nuit) — option future.

**exp22_reward_adaptive_lr** lancé dans la foulée (nuit du 27/07) : recette
exp20 à l'identique, seule la décroissance change, 100 epochs (~60 h, stop
manuel prévu). L'ablation propre paliers calendaires (exp20) vs
reward-adaptive (exp22) est donc en cours, même LR initial 5e-6, même facteur ÷3.
Fiche : `runs/10_fewshot_rl/exp22_reward_adaptive_lr/config.yaml`.

**Suite (2026-07-28)** : exp22 arrêté — les coupes sans restore figeaient la
dérive post-pic (test 49 → 22-26). Le « restart du best » est finalement
implémenté (best **train**, jamais test) dans exp22.1 →
`session_2026-07-28_exp22_verdict_et_restore_best.md`.

## Liens

- Code actuel des paliers : `src/train/schedules.py` (`StagedLrBetaCallback`)
- Run de référence : `runs/10_fewshot_rl/exp20_staged_lr_v2/config.yaml`
- exp21 reprend les mêmes paliers calendaires (LR 7e-6) — utile comme témoin
  pendant qu'on réfléchit au callback reward-adaptive.
