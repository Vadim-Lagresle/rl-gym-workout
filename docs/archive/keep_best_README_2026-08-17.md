# keep_best — adapters LoRA à conserver (déplacés depuis saves/trl_grpo/ le 2026-07-29)

Dossier de mise en sûreté : tous les « best » adapters du projet, regroupés pour
survivre à tout ménage disque dans `saves/trl_grpo/`. Chaque dossier est un
adapter PEFT (~126 Mo) qui se plugge sur `models/Qwen2.5-3B-Instruct`.
Les fiches détaillées sont dans `runs/` (configs `.yaml`, chemins d'origine
`saves/trl_grpo/<nom>`).

| Dossier | Sélection | Score | Contexte |
|---|---|---|---|
| `exp10.8_warmstart53_lr_div3_best` | best test | **54/100 (re-éval indép.)** | best du projet, lignée warm-starts |
| `exp20_staged_lr_v2_best` | best test | 58/100 @ step 800 (1 éval, re-éval à faire) | paliers LR calendaires ÷3/3ep |
| `exp22.1_restore_best_best` | best test | 53/100 @ step 850 | LR adaptatif + restore best train |
| `exp22.1_restore_best_besttrain` | best TRAIN (rolling mean 0.4752 @ step 1679) | — | + `.besttrain_info` |
| `exp22.2_rank64_median_best` | best test | 47/100 @ step 50 (avant spike KL) | r=64 à LR 5e-6 — trop agressif |
| `exp22.2_rank64_median_besttrain` | best TRAIN (rolling mean, adapter r=64) | — | + `optimizer.pt` + `.besttrain_info` |
| `exp22.3_rank64_lr_div4_best` | best test | 49/100 @ steps 350-500 | r=64 à LR 1.25e-6 — stable, décroche après epoch 11 |
| `exp22.3_rank64_lr_div4_besttrain` | best TRAIN (rolling mean 0.4783 @ step 851) | — | + `optimizer.pt` + `.besttrain_info` |
| `exp22.4_rank64_staged_div2_best` | best test | 50/100 @ step 250 | calendaire ÷2/4ep — même décrochage post-pic |
| `exp22.5_stage10ep_restore_best` | best test | 41/100 @ steps 400-500 | **bras ZERO-SHOT** de l'ablation few-shot (paliers 10 ep ÷2 + restore du best de palier) |
| `exp22.5_stage10ep_restore_besttrain` / `_stagebest` | best TRAIN global / de palier (r=64) | rolling mean 0.369 | + `optimizer.pt` + `.besttrain_info` |
| `exp22.6_stage10ep_fewshot_best` | best test | **49/100 @ step 250** | **bras FEW-SHOT k=10** de la même ablation — recette identique à exp22.5 |
| `exp22_reward_adaptive_lr_best` | best test | 49/100 @ step 200 | LR adaptatif sans restore |
| `exp19.2_scratch_5e-6_best` | best test | 45/100 @ step 300 | LR constant 5e-6 (collapse KL ensuite) |
| `exp19_fewshot_rl_k10_best` | best test | 43/100 @ step 400 | few-shot RL k=10, LR 7.33e-7 |
| `exp21_pure_reasoning_best` | best test single-turn | 37/100 single-turn, 9/100 multi-tour | reasoning pur (plan one-shot) |
| `exp10.5/10.6/10.7_*_best` | best test | 35-51 | étapes intermédiaires lignée exp10 |

Resté dans `saves/trl_grpo/` : `exp17_snis_v1_best` (5,8 Go — modèle COMPLET,
pas un adapter ; exp17 = échec 15/100, candidat à suppression).

## Ménage du 2026-08-05 (place pour les bests full-FT d'exp23)

`exp23_verl_repro` est un run **full-FT** : son best est un modèle HF complet de
5,8 Go (pas un adapter), et le swap atomique du callback exige ce même volume
libre à chaque amélioration du Pass@1. Pour dégager la marge, les deux artefacts
d'entraînement d'exp22.6 (1,4 Go chacun) ont été supprimés ; leurs valeurs sont
conservées ici pour la traçabilité :

- `_besttrain` supprimé — best TRAIN global : `step=322 epoch=7.00 rolling_mean=0.4681 lr=1.25e-6 beta=1e-2 palier=0`
- `_stagebest` supprimé — best du palier 1 : `step=506 epoch=11.00 rolling_mean=0.4620 lr=6.25e-7 beta=5e-3 palier=1`

Seul `exp22.6_stage10ep_fewshot_best` (adapter, 49/100) est archivé. Le run était
mort (purge pod à l'epoch 11,2), donc aucune reprise ne dépendait de ces optimizers.

## Ménage du 2026-08-06 (place pour le best + optimiseur d'exp23.1, 100 epochs)

`exp23.1` sauvegarde désormais le best test full-FT **avec** son état d'optimiseur
(~12 Go au total : 5,8 modèle + ~6,2 moments Adam 8-bit). Pour dégager la marge
(validé explicitement) :

- Les 4 `optimizer.pt` (914 Mo chacun) des runs LoRA clos ont été supprimés :
  `exp22.2_rank64_median_besttrain`, `exp22.3_rank64_lr_div4_besttrain`,
  `exp22.5_stage10ep_restore_besttrain`, `exp22.5_stage10ep_restore_stagebest`.
  Les **adapters** et `.besttrain_info` de ces dossiers restent intacts ; seuls
  les moments Adam (inutiles : runs morts, jamais repris) sont partis.
- `saves/trl_grpo/exp23_verl_repro_best` (modèle complet 5,8 Go, 40/100 @ step
  1175) a été supprimé : exp23.1 est la même recette prolongée à 100 epochs et
  doit le re-dériver vers l'epoch 25-27. Résultats documentés dans
  `runs/2_grpo_fullft/exp23_verl_repro/config.yaml`.
