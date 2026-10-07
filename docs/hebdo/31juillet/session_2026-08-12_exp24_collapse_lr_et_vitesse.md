# Session 2026-08-12 — exp24 : collapse au LR « 10× du blog » et enquête vitesse

## 1. exp24_r64_lr1e-5_b0.001 : collapse, pas crash

Le run tournait encore ce matin (step ~760/4400, epoch ~17) mais la politique était
morte : évals 20 → **24 (pic, steps 94-141)** → chute monotone → **0/100 dès step 658**.
Les rollouts finaux : 0 action valide, 30 rounds brûlés par épisode, ~1800 tokens de
bruit générés par trajectoire (vs ~600 sains). Tué à 10:51 ; la file a enchaîné seule
sur le job suivant (exit 143 → `done/`, démarrage 04).

**Lecture** : la prédiction « LR LoRA = 10× full-FT » du blog Thinking Machines ne
transfère pas telle quelle sur notre setup. Leurs benchs RL sont du single-turn math à
petits batches ; nous : multi-tour 30 rounds, reward sparse 0/1, buffer 256 — leur
propre caveat (« LoRA tolère moins bien les gros batches ») joue sans doute contre le
10×. À 1e-5, la KL s'emballe et la politique quitte la zone de récupération avant la
3e epoch (même mécanique qu'exp22.2 : LR × capacité r=64 trop agressif).

## 2. Enquête vitesse (« pourquoi 15 ep en 20 h ? »)

Chiffres mesurés (mtimes des saves de best + fiches exp23) :

| Contexte | Vitesse |
|---|---|
| exp24 r64, DÉBUT du run (steps 0-141) | **~74 steps/h** |
| exp23 scratch full-FT (fiche : « ~56 s/step ») | ~64 steps/h |
| exp23.4 warm-startée (3055 steps / 22.25 h) | ~137 steps/h |
| exp24 r64, moyenne sur 22.5 h (collapse) | ~34 steps/h |

Conclusions :
1. **Aucune régression de stack** : au départ, exp24 allait PLUS VITE qu'exp23 scratch.
   vLLM 0.26.0 = exactement la version d'exp23.3/23.4 ; les jobs de la grille ont les
   mêmes arguments que la recette exp23 (hors LoRA). Rien dans les modifs du 11/08
   (file, épinglage vLLM, fix hf CLI) ne touche le chemin chaud.
2. **Le ralentissement EST le collapse** : génération 3× plus longue + 30 rounds
   systématiques → le rollout de 256 trajectoires domine tout. Cercle vicieux :
   politique morte = steps lents = GPU gaspillé sur du bruit.
3. Le souvenir « avant, 100-150 epochs/jour » vient d'exp23.4, warm-startée sur une
   politique efficace (épisodes courts). Depuis scratch, on n'a jamais dépassé
   ~35-60 epochs/jour. **Run LoRA sain de 15 ep ≈ 9-11 h.**

### Complément (question « c'est le buffer 256 ? ») — non, mesures tqdm des logs

| Run | Config | s/it | min/ep |
|---|---|---|---|
| exp22.5 zero-shot r64 | 64 traj/step, max_rounds 20 | 32 | ~25 |
| exp20 few-shot r16 | 64 traj/step, max_rounds 20 | 44 | ~34 |
| exp22.6 few-shot r64 | 64 traj/step, max_rounds 20 | 54 | ~42 |
| exp22.3 few-shot r64 | 64 traj/step, max_rounds 20 | 84 | ~65 |
| exp23 scratch full-FT | buffer 256, max_rounds 30 | ~56 | ~41 |
| exp24 r64 3e-6 (en cours) | buffer 256, max_rounds 30 | 45 | ~33 |
| exp23.4 warm-startée | buffer 256, max_rounds 30 | 17 | ~12 |

Le buffer 256 ne change pas le volume par epoch (toujours 64 traj/step, 2992 traj/ep,
coût de rollout amorti ~0,5 s/traj). Les vrais facteurs : (a) max_rounds 20→30 (+50 %
de plafond d'épisode, payé plein tarif par une politique de scratch) — l'essentiel de
l'écart 32→45 s/it ; (b) la compétence de la politique (exp23.4 : 17 s/it, épisodes
courts) ; (c) le few-shot k=10 (~10k tokens de prefill/tour) rendait exp22.3/22.6 PLUS
LENTES que la grille actuelle. Levier possible non appliqué (casse la parité avec la
barre full-FT mesurée à max_rounds 30) : repasser max_rounds à 20 (~-30 %).

## 3. Grille restructurée (décisions Vadim)

- **15 epochs par run** (le 100 ep du 11/08 était une erreur de calibrage — l'écran
  suffit pour comparer les combos).
- Arm (1e-5, β0.001) **arrêtée** : r64 a répondu (collapse) ; r16/r32 parqués dans
  `runs/queue/skipped/` (l'indépendance au rang prédit le même sort).
- **Nouvelle arm LR 1e-6** (ratio 1× = LR full-FT) × 3 rangs — jobs 10-12.
- L'arm (1e-5, β0.01) reste en FIN de file (13-15) : teste si l'ancre KL forte
  contient 1e-5.
- File restante : 3e-6 × {64,16,32} → 1e-6 × {64,16,32} → 1e-5/β0.01 × {64,16,32},
  soit 9 runs ≈ 4 jours GPU. `exp24_r64_lr3e-6_b0.001` relancé à 10:51 (12/08).

Hypothèse de travail mise à jour : l'optimum LoRA est probablement entre 1e-6 et 3e-6
sur notre tâche (ratio 1-3×, pas 10×).

## 4. Analyse des LoRA historiques : la grille relue en « chaleur effective » α·LR

La mise à jour effective d'un LoRA ∝ (α/r)·r·LR = **α·LR** (r termes rang-1, préfacteur
α/r). Sous l'ancienne convention α=2r, monter le rang chauffait donc le run — ce qui
unifie tous les constats du projet :

| α·LR | Constat |
|---|---|
| ≥ 3e-4 | collapse : exp10 (32×1.5e-5, β0.001), exp22.2 (128×5e-6, β0.01), exp24 job 1 (32×1e-5, β0.001) |
| ~1.6e-4 | zone d'apprentissage : exp19.2, exp20 palier 0, exp22.3/22.6 (128×1.25e-6) — avec β0.01 ET refroidissement ensuite |
| 2-7e-5 | polissage : exp10.8 (54/100, 32×7.3e-7), paliers tardifs exp20 |

Autres écarts grille vs recettes gagnantes : β 0.001 (verl) vs **0.01 historique**
(garde-fou né du collapse d'exp10, que le job 1 a rejoué), zero-shot + 30 rounds vs
few-shot + 20, LR constant vs refroidissement (aucun bon score LoRA à LR constant :
54 et 58 = chaînes de warm-starts / paliers → la grille mesure la pente initiale,
pas le plafond).

**Swap validé par Vadim** : l'arm de fin de file (1e-5, β0.01, α·LR=3.2e-4 — condamnée
par l'historique) devient **(3e-6, β0.01)** : ablation propre « β 0.01 vs 0.001 à
chaleur égale (9.6e-5) » contre l'arm 04-06 en cours.
