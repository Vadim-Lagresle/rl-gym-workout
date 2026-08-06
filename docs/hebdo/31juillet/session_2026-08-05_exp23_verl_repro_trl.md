# Session 2026-08-05 — exp23 : la recette verl du papier traduite sur TRL

## Contexte

L'analyse comparative du 04/08 (verl AgentGym-RL vs notre stack TRL) a identifié 6
différences majeures entre leur run à 75/100 et notre lignée LoRA plafonnée à ~50.
Vadim demande une expérience qui réplique leur logique au plus près sur TRL :
full-FT, LR/beta constants (les leurs), vrai PPO, max_rounds 30, bonus d'entropie,
zero-shot, batch 256. C'est exp23_verl_repro (`runs/2_grpo_fullft/exp23_verl_repro/`).

## Correction du rapport du 04/08 — la mécanique mini-batch de verl

Le rapport disait « mini-batches de 8, ~32 updates clippées par batch de 256 ».
C'est FAUX : `ppo_mini_batch_size=8` compte des PROMPTS, et le fork le multiplie
par `rollout.n` (`agent_fsdp_workers.py:116` : `ppo_mini_batch_size *= n`) →
le mini-batch réel est 8 × 8 = **64 trajectoires**, soit **4 updates d'optimiseur
clippées** par batch de 256 rollouts. Coïncidence heureuse : 64 trajectoires par
update, c'est exactement notre step actuel (grad_accum=64) — seule la taille du
buffer de génération change.

## Ce que TRL 1.9.2 offre nativement (audit du source installé)

1. **`steps_per_generation`** (grpo_config.py:518) : buffer de rollouts découplé du
   pas d'optimiseur. `generation_batch_size = per_device_bs × world × steps_per_generation`.
   Avec `steps_per_generation=256` et `grad_accum=64` : une génération de 256
   trajectoires consommée en 4 optimizer steps.
2. **old_per_token_logps automatiques** (grpo_trainer.py:2562-2576) : dès que
   `grad_accum % (steps_per_generation × num_iterations) != 0` (ici 64 % 256 ≠ 0),
   TRL fait un forward sous le modèle PRÉ-updates et fige les logps de référence
   du ratio → le clipping PPO (`epsilon=0.2`, défaut = clip_ratio verl) devient
   réel dès la 2e update du cycle. Indépendant de `vllm_importance_sampling_correction`
   (qu'on garde à False, parité lignée exp10).
3. **`entropy_coef`** (grpo_config.py:866) : `loss -= coef × entropie moyenne par
   token actif` — même signe et même forme que verl (`dp_actor.py:253` :
   `policy_loss = pg_loss − entropy_loss × 0.001`). Rien à sous-classer.
4. **env_mask honoré partout** : le `tool_mask` (grpo_trainer.py:2214, 2877) masque
   les tokens obs/template dans la loss ET dans le ratio — le vrai PPO est
   compatible avec notre contrat multi-tour sans modification.

## Code (2 arguments CLI, aucun changement d'algo)

- `src/train/train_grpo.py` : `--steps-per-generation` (0 = défaut TRL = grad_accum,
  sémantique historique inchangée) et `--entropy-coef` (0.0 = off par défaut) ;
  validation des divisibilités + print « vrai PPO » au lancement ; les deux champs
  passés à `GRPOConfig`.
- `max_rounds=30` au train : aucun code — `--max-rounds-schedule '30:0'` existant.
  (Nuance conservée en commentaire : l'appendice B.3 du papier dit 20 tours pour le
  GRPO pur, mais le script du 75 tourne `rounds_ctrl fixed 30` — on suit le script.)

## Smoke test GPU (logs/smoke_exp23.log)

Full-FT, buffer 32 traj / grad_accum 8 (4 updates par cycle), entropy_coef 0.001,
8 steps. Tout est visible dans les métriques :
- update 1 de chaque cycle : `clip_ratio/* = 0` (on-policy, ratio ≡ 1) ;
- updates 2-4 : `clip_ratio/low_mean` et `high_mean` non nuls → clipping ACTIF sur
  les mini-batches off-policy ;
- `entropy_coef: 0.001` loggé, entropie 0.6-0.9, KL ~0.001, pas d'OOM.

## exp23_verl_repro — lancé 2026-08-05 ~11h20 (PID 2059898, setsid)

| Paramètre | Valeur | Source verl |
|---|---|---|
| Fine-tuning | full-FT (adamw_bnb_8bit) | FSDP fp32 (écart assumé) |
| LR / beta KL | 1e-6 / 0.001 constants | policy_lr / kl_loss_coef (low_var_kl ≡ k3 TRL) |
| Buffer / mini-batch | 256 traj / 64 traj (4 updates, eps 0.2) | train_bs 32×8 / ppo_mini 8×8 |
| Entropie | 0.001 | entropy_coeff |
| max_rounds train | 30 (= éval) | rounds_ctrl fixed 30 |
| Few-shot | 0 (zero-shot) | prompt fixe |
| Epochs | 30 (1320 steps, 44/epoch) | total_epoches 30 |
| vllm_max_len | 32768 | max_model_len |

Éval périodique 47 steps / 100 items ; best test (modèle FULL ~5.8 Go) sur le home ;
checkpoints périodiques sur /tmp (volatils). Durée estimée 24-40 h — c'est le run
qui teste l'hypothèse « le mur train ~0.47-0.50 vient du principe même de l'adapter ».

## Clôture exp22.6 (morte sur purge pod, step 516/2300, epoch 11.2)

Best test **49/100 @ step 250**, best train 0.4681 (epoch 7) ; 1re frontière de
palier (epoch 10) exécutée proprement (restore poids+Adam, LR ÷2). L'ablation
few-shot/zero-shot est lisible malgré la mort : **49 vs 41 test, 0.468 vs 0.369
train** → les exemples du prompt valent ~+8 pts, et le mur ~0.47 est bien un
invariant de la lignée few-shot LoRA.
