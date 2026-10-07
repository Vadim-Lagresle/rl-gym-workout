# Session 2026-08-18 — bilan de la nuit : exp25 concluante, record 39 en r16, bug 7B

La file a enchaîné seule : job 24 (r16/β0.01) → job 30 (exp25 ancre mobile, smoke +
ré-ancrage réel validés) → job 31 (pass@20 7B nu) → job 32 (7B full-FT, tué ce matin).

## exp24_r16_lr3e-6_b0.01 (fin du rejeu) — RECORD LoRA zero-shot : 39/100

Évals : 7, 17, 20, 23, 26, 29, **39**(step 329, ep ~7), 31, 29, 32, 32, **10**(ep ~12),
24, 25. Montée monotone jusqu'au record, puis instabilité tardive avec récupération
partielle. β0.01 ne protège donc pas totalement r16 en ancre fixe — mais le best 39
est sauvé. À β égal : r16 (39) >> r64 (29) — le petit rang gagne encore.

## exp25_r8_anchor4ep — l'ancre mobile tient sa promesse

Évals : 12, 13, 11, 18 | 27, 31, 30, 24 | 30, 26, 32, 31 | 30, **35** (step 658, FINAL).
Ré-ancrages exécutés aux epochs 4/8/12 (steps 184/368/552), aucune déstabilisation
post-ancre. **Best = dernière éval : le run était encore ascendant à l'arrêt des
15 epochs.** Lecture : 35 = record de la grille égalé, avec r8 (plus petit rang testé),
batch 64 on-policy, et SANS le collapse de β0.001 ni le plafond net de β0.01 fixe.
L'hypothèse « la KL fixe tire trop » en sort renforcée ; envie de suite : mêmes
hyperparamètres, budget plus long (le cut à 15 ep a probablement laissé des points).
Nuance : r16/β0.01 fixe a pointé plus haut (39) mais en zigzag ; l'ancre mobile est
plus lente mais plus propre. r8-mobile vs r16-fixe confond rang et ancre — une
ablation r16-mobile trancherait.

## Oracle pass@20 — Qwen2.5-7B nu (job 31)

pass@1 33.2 %, pass@5 54.9 %, pass@20 68 %. Par depth (@20) : d1 94 %, d2 76 %,
d3 32 %, **d4 0 % (3 items)**. Le 7B nu ~= 2× le 3B nu en pass@1 (33 vs 18), grosse
marge RL sur d2/d3… et le mur depth 4 tient toujours, même à 20 tirages (stat faible :
3 items).

## exp26 7B full-FT — TUÉ (bug de pipeline, pas un résultat scientifique)

Chronologie : smoke 2 steps OK (on-policy, 128 tokens) → run réel : **buffer 1
parfait (256 traj, actions valides ~1.0, reward train 0.40 — 2× le 3B débutant,
1071 tokens/traj)** → dès le buffer 2 : 0 action valide sur 30 rounds, 14k tokens/traj
(cap 512×30 rempli de bruit), reward 0 partout, KL 1.0 → 6.5e4 → 1e7, entropie 6.6,
828 s/step (ETA 35 j). Éval step 47 : 0/100 (30 format errors/ép) alors que le même
modèle nu fait 33 % via le serveur → politique HF réellement détruite ensuite
(boucle KL sur tokens hors-distribution, advantage nul partout : le seul signal
restant est beta·KL + entropie).

Un update à LR 1e-6 ne détruit pas un 7B : suspicion = **1re synchronisation des
poids HF → vLLM colocate après les 4 updates du buffer** (mode steps-per-generation),
qui casse la génération. Piste structurelle : le 7B a `tie_word_embeddings=false`
(lm_head séparé) — toute la stack n'a jamais synchronisé que des Qwen à embeddings
liés (0.5B/3B). Le smoke on-policy (sync après CHAQUE step, 128 tokens, max-len
16384) était sain → le bug est spécifique à la config réelle (buffer et/ou max-len
32768 / pression mémoire).

Actions : run tué (exit 143), job 33 (pass@20 du best) parqué, « best 0 % » (15 Go)
supprimé, **repro minimal lancé** (`repro7b_buffer` : config réelle, 5 steps — le
rollout du buffer 2 dit si ça reproduit). Prochain pas selon résultat :
- reproduit → bisecter (steps-per-generation off / max-len 16384) puis inspecter la
  sync TRL (lm_head non lié) ;
- pas reproduit → suspecter l'éval sync_before_eval / interaction mémoire du run long.

## Décision de mi-journée : abandon du 7B, cap sur LoRA 3B (soutenance < 1 mois)

Vadim tranche : plus le temps pour le 7B — il faut stabiliser LoRA sur le 3B et
passer au curriculum. Actions :
- **7B abandonné** : repro tué + purge totale demandées (modèle /tmp 15 Go, logs,
  jobs 32/33) — le pass@20 du 7B nu (33.2/68, d4=0) reste acquis dans `runs/7_oracle/`.
- **exp25 prolongée à 200 epochs** (`runs/queue/40_exp25_resume_200ep.sh`) : reprise
  EXACTE du checkpoint-690 (adapter + optimizer, epoch 15.0, retrouvé intact sur /tmp)
  sur la base ancrée cycle 3 reconstruite par `merge_anchor_chain.py`. Nouveau flag
  `--moving-anchor-initial-cycle` (schedules.py + train_grpo.py + selftest) : sans lui,
  le compteur de cycles repartant à 0, le callback aurait enchaîné 3 ré-ancrages
  parasites dès l'epoch 15.02 (et purgé les moments Adam repris). `--best-init-score
  0.35` protège le best existant ; branche scratch intégrée si /tmp purgé.

Incident infra : le shell du workspace Cursor/Coder est tombé en panne vers 11h50
(aucune commande n'aboutit, GPU lu à 0 Mo) — la purge 7B et le lancement du job 40
sont EN ATTENTE de la récupération de l'environnement. Le code et le job sont prêts.

## État du disque après ménage

9.7 Go utilisés / 35. Le best 69 (exp23.4) est archivé sur le Mac de Vadim et
supprimé du serveur ; bests restants : exp24 r16 β0.001 (35), r16 β0.01 (39),
exp25 (35) + chaîne d'ancres.
