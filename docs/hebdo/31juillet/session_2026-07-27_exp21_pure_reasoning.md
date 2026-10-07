# Session 2026-07-27 — Mode « pure reasoning » (single-turn RL) : code + lancement d'exp21

## Contexte

- exp20_staged_lr_v2 est terminé (966 steps, pas de collapse, **best 58/100 @ step 800**,
  adapter sur `saves/trl_grpo/exp20_staged_lr_v2_best`) — clôture propre encore à faire
  (re-éval indépendante du 58, config/INDEX/RESULTS).
- La VM a redémarré entre-temps : `/tmp` purgé → env v2 reconstruit via
  `setup/setup_agentgym_rl_v2.sh` (~10 min grâce au wheel flash-attn persisté dans
  `saves/wheels/`), serveur TextCraft relancé. Mêmes versions qu'avant
  (vLLM 0.25.1, TRL 1.9.0, flash-attn 2.8.3, torch 2.11).
- Demande Vadim : nouvelle expérience d'entraînement en **raisonnement pur** — le modèle
  donne en one-shot toute la recette séquentielle de crafting, on exécute exactement
  cette séquence dans l'environnement et on rétropropage le reward par GRPO. Même config
  qu'exp20, LR de départ un poil plus grand.

## Le paradigme single-turn (et pourquoi PAS le pipeline exp16)

En multi-tour (exp7→20), le modèle joue jusqu'à 20 tours avec une observation entre
chaque action. Ici : **une seule complétion** = `Thought:` (raisonnement libre) puis
`Actions:` (une commande TextCraft par ligne). La séquence est :

1. **parsée par regex déterministe** dans la complétion (`get N item` /
   `craft N item using …` / `inventory`, tolérant aux numérotations et puces) ;
2. **rejouée telle quelle** dans l'env (préfixe `"Action: "` par commande, arrêt au
   goal ; une action invalide ne stoppe pas le replay) ;
3. reward sparse 0/1 de l'env → GRPO sur la complétion entière (`env_mask` tout à 1 :
   il n'y a AUCUN token d'observation dans la complétion, contrat TRL trivial).

Piège évité : le pipeline d'éval exp16 (`src/eval/single_turn/`) passait par un
**second appel LLM** pour extraire les actions d'un plan en texte libre. Inutilisable
en RL : le reward dépendrait de l'extracteur (qui, en variante « informée », réparait
même les plans) et non des tokens générés. Ici le modèle doit émettre lui-même la
syntaxe exécutable — crédit attribuable à 100 %.

## Code (patron SNIS : tout se branche sur le `main()` existant)

| Fichier | Quoi |
|---|---|
| `src/train/rollout_plan.py` (nouveau, ~340 l.) | règles `PLAN_RULES` + ack, `load_plan_fewshot` (mêmes exemples résolus qu'exp18/19/20 reformatés « tâche → Thought + plan complet »), `build_plan_prompt_rows` (même marqueur `<ITEM_IDX:n>`), `parse_plan_actions`, `replay_actions`, `plan_rollout_func`, `run_plan_test_eval` (éval périodique single-turn, même protocole que le training), selftest parsing |
| `src/train/train_grpo.py` | flag `--plan-mode` : bascule few-shot / dataset / rollout / éval ; garde-fous (incompatible SNIS, max-rounds-schedule, filtres depth) |
| `src/train/periodic_eval.py` | `TestEvalCallback(eval_fn=…)` : fonction d'éval interchangeable, save-best et logging `eval/` inchangés |

## Validation avant GPU (leçon CLAUDE.md : jamais de run sans dry-run)

- Selftest parsing : 3 cas (plan propre, plan décoré 1./2)/Step 4:/Action:, prose sans
  action) — OK via `python -m src.train.rollout_plan`.
- Replay d'un plan connu-bon (complétion idéale de l'exemple few-shot idx 223, parsée
  par NOTRE regex puis rejouée) → reward 1.0, 0 invalid.
- Prompt rendu : **~1 830 tokens** avec k=10 (vs ~20k en dialogue multi-tour) →
  `vllm_max_len` défaut 16384 largement suffisant.
- Smoke GPU 2 steps (N=2, GAS=8, 16 items, éval 8 items) : rollouts parsés/rejoués,
  grad_norm 0.12, KL saine, éval single-turn 3/8 avec `parse_fail=0`, save-best
  adapter OK. Artefacts smoke supprimés.

## exp21_pure_reasoning — lancé

`runs/11_reasoning_rl/exp21_pure_reasoning/` · wandb `l4rwppam` · PID 397853 ·
config = exp20 (LoRA r=16 depuis zéro, N=8, 64 traj/step, k=10, paliers LR/beta
÷3 / 3 epochs, 21 epochs = 966 steps, éval/50 steps sur 100 items) avec :

- **LR départ 7e-6** (vs 5e-6 exp20 — « un poil plus grand »), beta 0.01, suit le LR ;
- `max_completion_length` **1024** (raisonnement + plan profond en un tour) ;
- éval périodique **single-turn** (le Pass@1 loggé n'est PAS comparable aux évals
  multi-tour des autres runs — protocole différent).

Premiers signaux (steps 1-14) : reward mean **0.35-0.42** dès le départ (le few-shot
single-turn amorce bien : plein de graines pour GRPO), ~**22 s/it** (vs 47 s/it exp20 —
le single-turn supprime les ~20 rounds de génération), ETA **~6 h**. Éval single-turn
elle aussi très rapide (~1 batch de génération + replays).

## Question scientifique

Le mur depth 3-4 du multi-tour vient-il de la **planification** (alors le single-turn
butera au même endroit) ou de la **gestion du dialogue long** (alors le single-turn
peut faire mieux) ? Le profil `eval/pass1_d{1..4}` de ce run donnera la réponse,
à croiser avec exp16 (éval-only) et exp20 (multi-tour, même recette RL).

## Surveillance

- `[rollout-plan]` : `n_actions` (taille des plans) et `episode_reward` par batch ;
- `eval/parse_fail` doit rester ~0 ; `completions/mean_length` → effondrement = alerte ;
- KL : mêmes seuils qu'exp20 (sain ~0.001, dérive > 0.05 = alerte) — LR départ plus haut ;
- paliers `[lr-beta-stage]` attendus aux epochs 3/6/9/… (7 au total).

## Reste à faire (repris de la semaine dernière)

1. Clôturer exp20 (re-éval indépendante du best 58, config/INDEX/RESULTS).
2. `--adapter-init` (reprise depuis un best adapter sans merge ni nouvelle ancre KL).
3. Verdict exp21 → décider de la suite (curriculum single-turn ? mix des deux modes ?).
