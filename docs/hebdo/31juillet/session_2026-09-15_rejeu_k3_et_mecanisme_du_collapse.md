# Session 2026-09-15 — analyse des collapses, étape 1 (logs) et étape 2 (rejeu k3 token par token)

## État du matériel et du registre au matin
- GPU libre depuis le 11/09 13:44 : la reprise d'exp41 (job 53b) est morte au pas 2777 (ép. 60) par
  recréation du pod (/tmp purgé, log coupé sans erreur). Nouvelle purge de /tmp le 15/09 ~08:47.
  exp42 (job 54) n'a jamais démarré. ckpt-2726 + optimizer et 7 cycles d'ancre sur le home, best 64 @ 2632.
- Lignes périmées dans runs/INDEX.md (non corrigées, à valider) : exp36.1 = collapse à l'ép. 5 sans
  atteindre sa 1re ancre, tuée par la purge du 03/09 (best 44) ; exp39 = 80 ép. terminées le 09/09,
  best 72 @ 3525 ; exp41 = reprise morte à l'ép. 60, best 64.
- Env v2 reconstruit (`logs/setup_env_20260915.log`), Qwen retéléchargé, serveur TextCraft relancé.

## Étape 1 — ce que les logs disent (figure `fig_collapse_entropy_kl.pdf`, slide B7b)
Source : `docs/rapport/figures/collapse_entropy_kl.json` (un enregistrement par pas pour 9 runs),
script `plot_collapse_entropy_kl.py`.
- Matrice sans curriculum complète : G=8 tient à ancre 368 pas (exp41, 64) ; G=16 ne tient qu'à
  ancre 184 pas (exp39, 72) ; G=16 à ancre 372/368/1116 pas casse (exp36 ép. 10, exp40 ép. 15,
  exp36.1 ép. 5 ; exp34 MAGELLAN ép. 7).
- Panneau (a) : KL soutenue > 1 seulement sous entropie 0,5. Mais la réciproque est fausse :
  Horizon vit 20 ép. à entropie 0,22-0,27, KL 1e-3. Entropie basse = nécessaire, pas suffisante.
- Panneau (b) : 18 pics isolés (KL > 1 après 10 pas sains) : 12 à entropie > 0,5, tous absorbés
  (max 1,25 dans les 10 pas suivants) ; 6 à entropie < 0,3, tous en escalade (30 à 8e4).
  Cas Horizon pas 253 : KL 137 avec reward normal, 0,03 au pas suivant → un pic porté par quelques
  tokens, pas une dérive de la politique.

## Étape 2 — rejeu k3 token par token (`src/analysis/replay_k3_tokens.py`, job 55)
Protocole : politique et référence exactes reconstruites par `merge_anchor_chain.py`
(référence = Qwen ⊕ cycles ≤ pas, politique = référence ⊕ adapter) ; génération par
`rollout.collect_episodes` via un faux trainer (token-identique à l'entraînement) ; recalcul HF de
log π et log π_ref sur les tokens d'action ; ρ = log π_ref − log π, k3 = e^ρ − ρ − 1.
64 tâches × 4 épisodes, seed 0, mêmes tâches pour tous. Sorties : `runs/15_replay_k3/<tag>.json`.

| Checkpoint | rôle | reward | k3 moyen (= KL TRL) | entropie | part k3 top 1 % | part ρ>0 | tokens ρ>5 | médiane long. tour | tours tronqués |
|---|---|---|---|---|---|---|---|---|---|
| exp40 pas 705 | malade, 5 pas avant l'explosion | 0,48 | 0,10 | 0,21 | 64 % | 30 % | 1 (max ρ 5,75) | 12 | 1 / 4701 |
| exp34 pas 611 | MAGELLAN, ½ ép. avant collapse | 0,48 | 0,013 | 0,78 | 64 % | 34 % | 0 | 34 | 0 |
| exp32 pas 7238 | Horizon, sain à entropie basse | 0,86 | 0,0010 | 0,32 | 54 % | 58 % | 0 | 33 | 0 |
| exp41 pas 2632 | G=8, sain | 0,64 | 0,0011 | 1,18 | 39 % | 57 % | 0 | 39 | 0 |
| exp40 pas 846 | post-explosion (KL 1e17 au log) | 0,25 | 7e14 | 0,003 | 100 % | 100 % | 488 (max ρ 45,5) | 513 | 493 / 761 |

Lectures :
1. **Écart vLLM/HF hors de cause** : |log π_HF − log π_vLLM| médian 1e-5 à 3e-3 nat, 1 à 11 tokens
   sur ~10^5 au-dessus de 2 nats, et sur ces tokens référence et politique HF sont d'accord (ρ petit).
2. **Régime dégénéré (pas 846) : 100 % du k3 vient des `<|im_end|>` FORCÉS.** 65 % des tours sont des
   boucles « Action: Action: … » tronquées à 512 tokens ; `collect_episodes` ajoute `<|im_end|>` avec
   env_mask = 1 (gradient actif, suffixe verl). La politique lui donne 1e-20, la référence 0,9 :
   ρ = 45, k3 = 6e19 par token. Ce n'est pas un tirage (p_vLLM = 1 est l'artefact du 0.0 posé).
3. **Pas 705 (amorçage)** : KL 100× celle des runs sains, portée à 70 % par le côté linéaire (433
   tokens ρ < −5 : politique sûre là où la référence ne l'est pas — « Action » direct, tours de 12
   tokens, 81 % des tours sans « Thought »). Côté exponentiel : les plus gros k3 (300) sont des
   « Thought » tirés à p = 0,3 % là où la référence donne 0,6-1,0 (ρ 4,5-5,75). L'explosion du log
   (2 400 aux pas 710-715) demande ρ ≈ 18 sur un token : continuation de cette tendance entre 705 et
   710, non observée directement (pas de checkpoint). Une seule troncature au pas 705, bénigne (ρ 0,25).
4. **Différence verl / TRL** : verl borne k3 à [−10, 10] par token
   (`external/AgentGym-RL/verl/agent_trainer/ppo/core_algos.py:381`) ; TRL ne borne pas
   (`trl/trainer/grpo_trainer.py:3193`). Le commentaire de `train_grpo.py:558` (« même estimateur »)
   est faux sur ce point. Dans la stack du papier, un token ne peut pas peser plus de 10.
5. Chaîne proposée : sharpening (entropie ↓, « Thought » supprimé, tours courts) → un token éliminé
   est parfois retiré, ρ 5 → 18, k3 exponentiel non borné → gradient clippé dominé par ce token →
   format cassé, boucles, troncatures → `<|im_end|>` forcés à ρ 45 → 1e19. Deux mécanismes, un seul
   défaut d'estimateur.

## Pistes (à valider par Vadim)
- Borner k3 par token comme verl (clamp à 10) : patch minimal du trainer TRL (sous-classe ou hook).
- Exclure les `<|im_end|>` forcés du terme KL (ils restent dans la perte de politique si on veut le
  comportement verl), ou les masquer tout court.
- Surveillance : longueur médiane de tour et fraction de tours sans « Thought », plus précoces que la KL.
- Le sharpening lui-même (entropie) n'est pas traité par ces parades : elles évitent la divergence,
  pas le plafonnement.

## Fichiers touchés
- créés : `src/analysis/replay_k3_tokens.py`, `runs/queue/55_replay_k3_tokens.sh`,
  `runs/15_replay_k3/{exp40_s705,exp34_s611,exp32_s7238,exp41_s2632,exp40_s846}.json`,
  `docs/rapport/figures/{collapse_entropy_kl.json,plot_collapse_entropy_kl.py,fig_collapse_entropy_kl.pdf/png}`,
  `docs/slides/figures/fig_collapse_entropy_kl.pdf`, ce document.
- modifiés : `docs/slides/SOUTENANCE_V2.tex` (slide B7b, avant B8).
- /tmp (volatil) : `/tmp/models/replay/<tag>_{ref,policy}` (10 modèles complets).
