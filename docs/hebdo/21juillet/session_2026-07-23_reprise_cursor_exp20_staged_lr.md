# Session 2026-07-23 — Reprise (Cursor) : vérification migration v2, schedule LR/beta par paliers, lancement exp20

Contexte : la session précédente (Claude Code) s'est arrêtée en plein travail pour
cause de crédit épuisé — pile au début de la tâche « relancer une expé from scratch
avec LR ÷3 toutes les 3 epochs et beta KL adapté ». Reprise depuis Cursor.
Ce document fait le pont : ce qui était déjà fait, ce qui a été fait aujourd'hui.

## 1. Vérification de l'état laissé par la migration (tout est bien là)

| Vérification | Résultat |
|---|---|
| Commit migration `1298b95` (feat(v2)) | présent, dépôt propre |
| glibc | **2.39** (VM CentOS Stream 10) |
| Env v2 `/tmp/envs/agentgym-rl-v2` | présent, **vllm 0.25.1 / trl 1.9.0 / flash-attn 2.8.3.post1 / torch 2.11.0+cu130 / peft 0.19.1** |
| GPU / trainings actifs | libre / aucun |
| Serveur TextCraft (36005) | opérationnel |

Rien à reconstruire : l'env v2 a survécu sur /tmp depuis le 22/07 (rappel : s'il est
purgé, `bash setup/setup_agentgym_rl_v2.sh` le refait en ~10 min grâce au wheel
flash-attn conservé dans `saves/wheels/`).

## 2. Ce qui manquait : le schedule LR/beta par paliers (jamais codé)

La session coupée s'était arrêtée sur la toute première commande de cette tâche
(un grep de `self.beta` dans TRL). Fait aujourd'hui :

### Vérifications préalables sur TRL 1.9.0 installé (faisabilité)

- **beta est mutable en cours de run** : `GRPOTrainer` relit `self.beta` à CHAQUE
  calcul de loss (`grpo_trainer.py:2626, 3069`). La seule capture à l'init
  (`beta=self.beta`, ligne 1028) concerne le chemin Liger (`use_liger_kernel`),
  que l'on n'utilise pas. Assigner `trainer.beta` suffit donc.
- **LR mutable aussi** : le run utilise `lr_scheduler_type="constant"` → un
  `LambdaLR` avec lambda≡1, dont le LR effectif est `base_lrs[i] × 1` recalculé à
  chaque `scheduler.step()`. Il faut donc modifier `base_lrs` (en déballant
  l'éventuel wrapper accelerate) EN PLUS des `param_groups` de l'optimizer, sinon
  le scheduler écraserait le changement au step suivant.

### Implémentation : `schedules.StagedLrBetaCallback` (+ CLI)

- `src/train/schedules.py` : nouveau `StagedLrBetaCallback(TrainerCallback)` —
  à `on_step_begin`, palier = `int(state.epoch // every_epochs)` (stateless →
  correct après un `--resume-from-checkpoint`) ; si le palier change :
  `lr = base_lr / factor^palier`, `beta = base_beta / factor^palier` (ou constant
  avec `scale_beta=False`), appliqués au scheduler + optimizer + `trainer.beta`,
  avec une ligne de log `[lr-beta-stage]`.
- `src/train/train_grpo.py` : arguments `--lr-stage-every-epochs` (0 = off),
  `--lr-stage-factor` (défaut 3), `--lr-stage-keep-beta` (ablation : ne divise
  que le LR). Le callback reçoit `trainer_ref` après construction (même pattern
  que `TestEvalCallback`).

**Choix scientifique (validé par Vadim, question posée en session)** : beta suit
le LR (÷3 aux mêmes paliers) — lecture littérale de la demande. Les alternatives
(beta constant = parité exp10.3 ; beta ×3 = force de rappel absolue constante)
restent accessibles via `--lr-stage-keep-beta` ou un facteur custom.

### Validation (avant tout lancement long, comme d'habitude)

- À sec : `py_compile` des deux fichiers, `--help` OK.
- **Smoke GPU dédié au schedule** : 2 items / 1 prompt/step → 2 steps/epoch,
  6 steps = 3 epochs avec `--lr-stage-every-epochs 1`. Les 3 paliers se déclenchent
  aux bonnes epochs (`[lr-beta-stage] palier=0/1/2`, lr 5e-6 → 1.667e-6 → 5.556e-7,
  beta 1e-2 → 3.33e-3 → 1.11e-3), et le **LR effectif loggé par le Trainer suit**
  (décalage d'affichage d'au plus 1 step — négligeable sur des paliers de ~140
  steps). `kl` loggé (donc chemin beta≠0 actif). Exit 0, artefacts nettoyés.

## 3. Lancement : exp20_staged_lr_v2 (premier run LONG de la stack v2)

Config complète : `runs/10_fewshot_rl/exp20_staged_lr_v2/config.yaml`. En résumé :
même recette qu'exp19.2 (Qwen2.5-3B nu, GRPO LoRA r=16, few-shot k=10, N=8,
64 traj/step, éval/50 sur 100 items, best sur le home) mais **21 epochs (~980
steps) avec LR 5e-6 et beta 0.01 divisés par 3 toutes les 3 epochs** (7 paliers).

Double objectif :
1. **Endurance de la stack v2** : la sync colocate en mémoire n'a été validée que
   par des smokes d'1 step — ce run la teste sur des centaines de steps (+ vitesse
   à comparer aux 246 s/it d'exp19.2).
2. **Anti-collapse par schedule** : exp19.2 a collapsé (KL ×3500) vers l'epoch
   ~7.5, encore à 5e-6. Ici, à cette epoch le LR sera déjà à ~5.6e-7. Repères :
   dépasser le best honnête 45/100 (exp19.2 step 300) sans divergence.

Surveillance (voir config) : lignes `[lr-beta-stage]` aux epochs 0/3/6/…,
`kl` wandb (alerte si > ~0.05 soutenu), `completions/mean_length`, s/it.

### Observations au lancement (premiers steps, avant de laisser tourner)

- wandb `9fc7abct`, PID 2010377, 966 steps prévus, palier 0 confirmé dans le log.
- Premier rollout : **24/64 récompensés** (≈ baseline promptée 30 % ; exp19.2 : 23/64).
- **Vitesse : 47,3 s/it contre 246 s/it sur exp19.2** (×5,2) — le gain concret de la
  migration (sync colocate en mémoire au lieu de l'écriture 5,8 Go/step + recréation
  moteur, + vLLM 0.25.1 sm100 natif). ETA ~12h40 pour le run complet, contre ~66 h
  qu'auraient pris 966 steps sur l'ancienne stack.
- `kl` = 0.00078 au step 2 (régime sain), complétions 616-651 tokens (idem exp19.2 sain).

## 4. Pendant que exp20 tourne : audit des hyperparamètres `GRPOConfig` figés en dur

Revue pédagogique de `src/train/train_grpo.py` en Cursor (dossier `src/train/`
disséqué fichier par fichier). Question de départ : pourquoi `lr_scheduler_type`
reste `"constant"` même quand on demande explicitement autre chose — aucun flag
CLI ne permettait de le changer, la valeur était figée en dur dans le code.

### Origine et audit des 4 hyperparamètres GRPO figés en dur (aucun CLI)

| Paramètre | Origine | Couplage avec autre chose | Décision |
|---|---|---|---|
| `lr_scheduler_type="constant"` | `17d2d55` (11 juin, audit verl→TRL, repro Table 6 papier) — jamais reproposé depuis | **Oui** : `StagedLrBetaCallback` (codée ce matin, section 2) suppose ce mode (réécrit `base_lrs` en supposant un multiplicateur de schedule ≡ 1) | **Exposé** via `--lr-scheduler-type` |
| `loss_type="dapo"` | même commit du 11 juin, mais double justif. : repro papier + anti-dérive TRL entre versions | aucun | laissé en l'état |
| `num_iterations=1` | corrigé le 11 juin (pas arbitraire : le fork verl du papier ignore lui-même `ppo_inner_epochs=2`) | **Oui** : couplé à `vllm_importance_sampling_correction=False` (ratio PPO ≡ 1, skip du forward `old_logprobs`) | laissé en l'état |
| `temperature=1.0` / `top_p=1.0` | hérité du tout premier script jetable (`scratch/07_trl_grpo_textcraft_smoke.py`, 6 mai) — sans rapport avec l'audit du 11 juin | aucun connu | laissé en l'état (hors périmètre demandé) |

### Changement appliqué : `--lr-scheduler-type` (portée volontairement limitée à ce seul paramètre)

- `src/train/train_grpo.py` (`build_parser`) : nouvel argument `--lr-scheduler-type`,
  choix `constant` / `constant_with_warmup` / `linear` / `cosine` /
  `cosine_with_restarts`, défaut `"constant"` → **aucun changement de
  comportement** pour un run qui ne passe pas le flag.
- `GRPOConfig(lr_scheduler_type=...)` : `"constant"` en dur → `args.lr_scheduler_type`.
- Garde-fou (`main()`, juste avant la construction de `StagedLrBetaCallback`) :
  `SystemExit` si `--lr-stage-every-epochs` est utilisé avec
  `--lr-scheduler-type` ≠ `"constant"` — les deux mécanismes ne peuvent pas
  cohabiter (cf. couplage dans le tableau ci-dessus).
- Validation à sec : `py_compile` + `--help` OK sur l'env v2 (`/tmp/envs/agentgym-rl-v2`),
  aucune erreur de lint.

### Pas d'impact sur exp20_staged_lr_v2 (déjà en cours)

Un process Python déjà lancé ne relit jamais son code source modifié sur disque ;
et de toute façon exp20 ne passe pas le nouveau flag, donc la valeur effective de
`lr_scheduler_type` reste `"constant"` avant/après ce changement — pas de
divergence de comportement. Confirmé dans le log : `[lr-beta-stage]` actif dès
le step 0 (palier 0, lr=5e-6, beta=1e-2), toujours au palier 0 au step 109
(epoch 2.37 ; palier 1 attendu vers step ~138).
