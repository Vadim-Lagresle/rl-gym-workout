# Session 2026-08-26 — exp32 : RECORD 82/100 (objectif papier dépassé), lancement exp36, refonte file + plan rapport

## L'événement : exp32_horizon termine sur un record à 82/100

Arrêt volontaire au plateau à 17:17 (ep ~80/200, **2 j 00 h 17 de GPU**, ~23 s/step
de moyenne — les épisodes courts du début de curriculum accélèrent tout le run).

- **Best : 82/100, atteint DEUX fois (steps 6110 et 7238)** — record du projet
  (+9 sur exp25.2) et **au-dessus de l'objectif papier AgentGym-RL-3B (75)**.
- Trajectoire : ~67 @ ep ~29 → 82 @ ep ~66 → plateau final 75-82 (d'où l'arrêt).
- ⚠ CORRECTION D'ÉCHELLE (26/08 soir) : avec N=16, un epoch TRL = 374×16/64 =
  **93 steps** (vérifié : 20 ré-ancrages /4 ep = ep ~80 ; total exp36 = 7440
  steps pour 80 ep). Un epoch N=16 contient donc 2× plus de trajectoires qu'un
  epoch N=8 (exp25 : 46 steps/ep) — comparer exp25 vs runs N=16 en STEPS ou en
  trajectoires, pas en epochs. Durée réelle d'un run N=16 de 80 ep à 30 tours :
  ~7440 steps ≈ 3.5-4 j → arrêts au plateau indispensables avant le gel.
- 20 ré-ancrages (ancre mobile /4 ep) sans le moindre collapse ; KL/entropie saines.
- Config : recette exp25 + horizon progressif 10/20/30 (ep 0/15/30) + **N=16**
  à 64 traj/step constant + gpu-util 0.5 (verdict bench exp25.1).
- Artefacts (home) : best+optimizer (`exp32_horizon_best`, .best_info step=7238),
  dernier ckpt (`exp32_horizon_ckpt7426`, 191 Mo), chaîne d'ancres (20 cycles).
- ⚠ Lecture : double delta vs exp25 (horizon ET N) — **exp36_n16, lancée dans la
  foulée (17:17), tranche** : ≈82 → le gain venait de N ; ≈73 → du curriculum.

## Décisions et changements du jour (matin/après-midi, avant l'arrêt)

1. **80 epochs par défaut** pour tous les jobs restants (43z/44/45/46) — les 200
   ep « arrêt manuel » ne servaient à rien en pratique.
2. **exp33 (job 44)** : paliers depth recalés pour 80 ep → `'1:0,2:12,3:25,4:37'`.
3. **exp35 (job 46) redéfinie en CURRICULUM de budget de sortie** (option A) :
   nouveau mécanisme `--max-completion-schedule-epochs '256:0,512:15,1024:35'`
   — code miroir de ScalingInter : `parse_max_completion_schedule_epochs` +
   `current_max_completion` dans `schedules.py` (+ cas au selftest, VERT),
   flag + 2 garde-fous dans `train_grpo.py` (max du schedule ≤
   --max-completion-length ; vLLM in-process requis), lecture du palier à
   chaque tour dans `rollout.py:121`. Sans schedule : comportement inchangé.
4. **PLAN_RAPPORT.md réécrit** sur les remarques de Vadim — le fil rouge est
   RENVERSÉ : le sharpening (Huang 2024 / pass@k-plafond Yue 2025) devient
   l'hypothèse adverse à tester, la question du rapport étant « un curriculum
   peut-il PERCER les murs de difficulté ? ». Contributions restructurées
   (guide de bonnes pratiques GRPO multi-tour / étude des curriculums / analyse
   inter-curriculums), GRPO seul en intro (DRL → annexe A + cours Finn/Levine),
   SNIS et test-time réduits à « autres pistes », critique chiffrée du dataset
   (train d1=109/d2=221/d3=43/**d4=1** vs test 31/41/25/**3** — vérifié ;
   re-stratifier = travail futur n°1), AgentGym-RL = **5 envs** (pas 14 —
   vérifié dans les scripts du papier).
5. Note méthodo « compute contrôlé » ajoutée au §4.4 du plan : comparer à
   epochs égales (données) ET à tokens générés égaux (`num_tokens` cumulé),
   + heures GPU + tours d'env — tout est déjà loggé.

## État de la file après la session

exp36_n16 (EN COURS, 80 ep) → exp33_depth → exp34_magellan → exp35_budget
(curriculum 256→1024). Tous à N=16 + gpu-util 0.5 + 80 ep. Gel GPU le 2/09 :
~6 jours pour 3-4 runs → arrêts au plateau comme aujourd'hui.

## Disque (après la session)

Home : 17 Go utilisés / 18 Go libres. Nouveaux artefacts : `exp32_horizon_best`
(183 Mo, optimizer inclus), `exp32_horizon_ckpt7426` (191 Mo),
`exp32_horizon_anchors` (20 cycles, ~1.1 Go). Budget par run restant ~1.2 Go
(best+ckpt+ancres) → OK pour la fin de la phase 2.
