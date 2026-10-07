# Session 2026-08-17 (après-midi) — ancre KL mobile (exp25), full-FT 7B (exp26), purge disque

Suite de la matinée (lecture de la grille exp24 : β0.001 monte à 31-35 puis collapse,
β0.01 stable mais plafonne 27-29). Décisions de Vadim : tester l'hypothèse « la KL
fixe tire trop » avec une ancre mobile en r8, puis répliquer la recette exp23 sur le
7B ; chaîner le tout sans intervention après le job 24 en cours.

## 1. Purge disque (validée) — 30 → 15 Go utilisés

- `exp23.4_best` : optimizer.pt supprimé (8.1 Go) ; **poids 5.9 Go conservés jusqu'au
  téléchargement sur le Mac** (commande rsync fournie), suppression ensuite.
- Bests exp24 lus supprimés (r64 ×3, r32) ; conservés : le 35/100 (r16/β0.001) et le
  run en cours (r16/β0.01).
- `saves/keep_best/` supprimé (5.5 Go, runs clos exp10-22) — README préservé dans
  `docs/archive/keep_best_README_2026-08-17.md`.
- Jobs 25-28 de la grille parqués dans `runs/queue/skipped/` (β lu avec r16 ce soir,
  arm 1e-6 dépriorisée).
- Politique 7B (contrainte physique) : best = poids seuls sur le home, ckpts sur /tmp.

## 2. exp25 — LoRA r8, ancre KL mobile (job 30)

TRL 1.10 a `sync_ref_model` (TR-DPO) mais **NotImplementedError avec PEFT** → nouveau
`MovingAnchorCallback` (schedules.py) en « merge-and-restart », validé par selftest CPU :

1. snapshot de l'adapter (`<run>_anchors/cycle<k>`, chaîne reconstructible par
   `src/utils/merge_anchor_chain.py`) ;
2. fusion dans la base, flag merged oublié (jamais défusionné) ;
3. adapter réinitialisé EN PLACE (A=kaiming, B=0) → politique inchangée, mêmes
   Parameters → optimizer valide sans reconstruction ;
4. moments Adam LoRA purgés (ancien paysage).

L'ancre TRL-PEFT (« adapter désactivé ») devient la politique du dernier ré-ancrage :
la KL mesure la distance au dernier point de confiance, plus à Qwen nu.

Config : r8/α32 (blog : petit rang suffit — vérifié exp24), LR 3e-6 **constant**,
β 0.01 (validé), ré-ancrage /4 ep, **batch 64 on-policy** (retour demandé — caveat
gros batch LoRA du blog ; plus de buffer 256), 15 epochs. Succès = >30 sans collapse.

## 3. exp26 — full-FT 7B, recette exp23, 150 epochs (jobs 31-33)

- Job 31 : **pass@20 du 7B nu** sur le test set (eval_oracle, résumable).
- Job 32 : smoke mémoire 2 steps (OOM détecté avant d'engager 150 ep) puis le run :
  recette exp23 à l'identique (LR 1e-6, β 0.001, buffer 256, entropie 0.001,
  zero-shot), `--vllm-gpu-util 0.2`, best poids seuls via nouveau
  `--best-delete-before-save` (pic disque = 1 modèle). **Durée : 7-11 jours GPU.**
- Job 33 : pass@20 du best exp26 → étude avant/après (la frontière de support
  bouge-t-elle, ou le RL fiabilise-t-il seulement ? — lecture 1 bit/épisode).

## 4. Modifications de code

- `schedules.py` : `MovingAnchorCallback` + selftest 6 (invariance politique, ancre
  déplacée, B=0, moments purgés, chaîne écrite) — tous les selftests passent.
- `train_grpo.py` : `--moving-anchor-every-epochs` (garde-fous : LoRA seul, β>0,
  exclusif des paliers LR à restore), `--best-delete-before-save`.
- `periodic_eval.py` : mode delete-before-save dans TestEvalCallback.
- `ensure_qwen_tmp.sh` : 2e argument = repo HF (7B téléchargeable).
- `start_vllm_server.sh` : timeout d'attente paramétrable (`WAIT_ITERS`, 8 min pour 7B).
- `src/utils/merge_anchor_chain.py` : reconstruction base ⊕ cycles ⊕ adapter.
- Jobs de file 30-33 avec smokes intégrés (le run réel ne part que si le smoke passe).

File après le job 24 (r16/β0.01, fin ~ce soir) : 30 (exp25) → 31 (pass@20 base) →
32 (exp26, ~1 semaine) → 33 (pass@20 best). Le runner survit aux purges (rebuild).
