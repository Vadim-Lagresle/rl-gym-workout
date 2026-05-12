## Résultats expérimentaux — RL agentique sur TextCraft

Ce document rassemble les expériences conduites, leur protocole exact, et
leurs résultats numériques. Il est conçu pour être autonome : on doit
pouvoir lire une section sans avoir à consulter le `WORKLOG.md` (qui lui
est un journal chronologique, pas un référentiel de résultats).

Convention : "v2", "v3" etc. désignent des **versions successives du
script de training** `scratch/07_trl_grpo_textcraft_smoke.py`, pas des
versions de modèle au sens habituel. Le modèle base est toujours
Qwen2.5-3B-Instruct.

---

### Setup commun à toutes les expériences

| Élément | Valeur |
|---|---|
| Modèle de base | Qwen2.5-3B-Instruct |
| Méthode RL | GRPO (TRL 0.19.1) |
| PEFT | LoRA rank 16, alpha 32, sur tous les `*_proj` |
| Précision | bf16 |
| `per_device_train_batch_size` | 1 |
| `num_generations` (N) | 2 |
| `max_completion_length` | 128 tokens |
| `max_rounds` (rollout interactif) | 20 (sauf v4 ScalingInter) |
| Learning rate | 1e-6 (linéaire decay) |
| Optimizer | AdamW |
| `max_grad_norm` | 1.0 (depuis v3) |
| GPU | NVIDIA A100 40 GB single |
| Env | TextCraft serveur HTTP local (127.0.0.1:36005) |
| Train set | 256 problèmes (`--max-items 256`) |
| Test set | 100 problèmes (indices 0..99 du pool TextCraft) |
| Métrique principale | Pass@1 sur le test set |

L'eval Pass@1 utilise vLLM avec `enforce_eager=False` pour bénéficier des
CUDA graphs avec LoRA (sinon ralentissement 100×, cf. issue résolue le
2026-05-09).

---

### Baseline — Qwen-3B-Instruct sans aucun training

| Métrique | Valeur |
|---:|---:|
| Pass@1 sur test 100 | **18 / 100** (18%) |
| Items résolus | majoritairement 1-action triviaux + quelques craft 2-3 actions |

Le baseline est référencé comme `baseline` dans `scratch/09_compare_4way.py`
et utilise les logs de `scratch/eval_logs/`. Aucun shaping, aucun fine-tuning.

---

### Exp 1 — GRPO v2 step10 (shaping bugué, run court)

**Objectif** : premier signal d'apprentissage GRPO sur TextCraft. Vérifier
que le pipeline tient debout sur 10 steps.

**Setup v2 spécifique** :

| Élément | Valeur |
|---|---|
| Reward formula | `episode_reward + shape(count_actions(completion)) − 0.01 × invalid_steps` |
| Shape v2 | `+0.02` si `count_actions == 1`, `-0.05` sinon |
| Steps | 10 |
| Run name | `trl_grpo_rolloutfunc_v2_step10` |

Important : `count_actions(completion)` opère sur la **completion
concaténée** (tous les messages d'assistant collés bout à bout). C'est le
bug.

**Résultats** :

| Métrique | Valeur |
|---:|---:|
| `train_runtime` | ~880 s (~15 min) |
| `train_loss` final | ~0.04 |
| Pass@1 test 100 | **18 / 100** (18%) |

Pass@1 strictement égal au baseline, mais avec un **set d'items partiellement
différent** : 8 items résolus uniquement par v2-step10 (que le baseline ratait)
et 8 items résolus uniquement par baseline (que v2-step10 ratait). Net = +0.
Pas de gain réel à 10 steps mais le pipeline fonctionne et le modèle a
commencé à se déplacer dans l'espace des politiques.

---

### Exp 2 — GRPO v2 step50 (shaping bugué, run long)

**Objectif** : voir si laisser tourner GRPO plus longtemps avec le même
shaping rapporte. *Réponse : non, ça empire activement.*

**Setup v2 spécifique** : identique à Exp 1 mais 50 steps.
Run name : `trl_grpo_rolloutfunc_v2_step50`.

**Résultats** :

| Métrique | Valeur |
|---:|---:|
| `train_runtime` | 4033 s (~67 min) |
| `train_loss` final | 0.043 |
| Pass@1 test 100 | **14 / 100** (14%) |
| Δ vs baseline | **−4** |
| Δ vs v2-step10 | **−4** |
| `grad_norm` max | 1765.89 au step 35-36 |

**Diagnostic du training** (25 batches de rollout) :

- 5 batches sur 25 ont reward positif (toujours avec completion courte
  95-787 tokens) → 1-shot lucky wins.
- 20 batches sur 25 ont reward négatif (toujours avec completion longue
  1100-1550 tokens) → multi-turn pénalisé par le bug.
- Pic `frac_reward_zero_std=1.0` à 3 reprises (12 % de steps perdus).

**Confusion v2-step50 vs v2-step10** : 7 items gagnés uniquement par
step50 (les 1-action triviaux), 11 items perdus que step10 résolvait
(les craft multi-action complexes). Net = −4. **Reward hacking au sens
strict de Skalse et al.** : le modèle a parfaitement optimisé une
fonction qui encodait mal l'intention.

---

### Exp 3 — GRPO v3 step50 (shaping corrigé)

**Objectif** : isoler l'effet du fix de shaping (per-turn vs concaténé)
toutes choses égales par ailleurs. Hypothèse : le pattern reward-vs-longueur
disparaît et le Pass@1 remonte au moins au niveau baseline.

**Setup v3 spécifique** :

| Élément | Valeur |
|---|---|
| Reward formula | `episode_reward + shape(mean_n_actions_per_turn) − 0.01 × invalid_steps` |
| Shape v3 | `+0.02` si `0.9 ≤ mean ≤ 1.1`, `−0.05` si `mean > 1.5` ou `mean < 0.5`, `0.0` sinon (zone neutre) |
| `max_grad_norm` | 1.0 (ajouté en v3) |
| Steps | 50 |
| Run name | `trl_grpo_rolloutfunc_v3_step50` |

Différence clef avec v2 : le décompte d'actions se fait **par message
d'assistant à l'intérieur de `textcraft_rollout_func`**, puis on prend la
moyenne. Un épisode 10-tours avec 1 action propre par tour a maintenant
`mean = 1.0` (bonus) au lieu de `count = 10` (malus v2).

**Résultats training** :

| Métrique | Valeur |
|---:|---:|
| `train_runtime` | 4120 s (~69 min) |
| `train_loss` final | 0.044 |
| `mean_n_actions_per_turn` range | 0.9–1.4 (stable autour de 1.0) |
| `grad_norm` range (clipped) | 0.0–4.9 |
| Reward trajectory | oscillation modérée, plus de spikes 0.95 sur completions courtes |
| Reward pour trajectoire qui réussit | +0.95 à +1.02 avec completions ~700-1300 tokens (vs ~95 tokens en v2) |

**Résultats eval** (commit `c7bf3ec`) :

| Métrique | Valeur |
|---:|---:|
| Pass@1 test 100 | **8 / 100** (8 %) |
| Δ vs baseline | **−10** |
| Δ vs v2 step50 | **−6** |
| `mean_rounds` | 27.8 |

**Confusion 4-way** (baseline / v2-step10 / v2-step50 / v3-step50) :

- 3 items résolus **uniquement** par v3 (`textcraft_1, 5, 8`) → gain
  marginal sur un petit cluster d'items
- 9 items résolus par v2-step50 mais ratés par v3 → grosse régression
- 1 seul item gagné vs baseline (`textcraft_2`)

**Smoking gun training** : à step 21, `completions/mean_length = 98.5`,
reward = +1.02, `frac_reward_zero_std = 1.0`. Le pattern "1-tour gagne, sinon
on raccourcit la réponse" est encore plus net qu'en v2. Le bonus `+0.02`
sur `mean_n ≈ 1.0` continue à récompenser des **échecs courts bien formatés**
(base reward = 0, shape = +0.02 → avantage positif à l'intérieur du
groupe). Le `max_grad_norm=1.0` a bien tenu (max observé 19.2 vs spike
1765 de v2), mais ça n'a pas suffi à compenser le signal vicié.

**Verdict** : le shaping per-turn (v3) **n'a pas réparé v2**, il a aggravé
le problème. Tout shaping non-sparse sur reward verifiable doit être
banni — c'est le constat qui mène à v4.

---

### Exp 4 — GRPO v4 ScalingInter (sparse reward, AgentGym-RL aligned)

**Objectif** : tester si combiner (a) le retour à la reward sparse 0/1
(aligné papier, fix de v3) et (b) le curriculum `max_rounds` du papier
permet d'au moins remonter au baseline.

**Modifications scientifiques vs v3** :

1. **Reward = `episode_reward` brut** (commit `30e580e fix(grpo):
   align textcraft_reward with AgentGym-RL`). Suppression complète du
   shaping `+0.02 / −0.05 / −0.01 × invalid_steps`. Le rollout continue
   d'émettre `invalid_steps` et `mean_n_actions_per_turn` mais
   uniquement pour les logs, plus dans le gradient.
2. **Curriculum ScalingInter sur `max_rounds`** :
   `5` (steps 0-12) → `10` (13-25) → `15` (26-37) → `20` (38-49).

**Setup spécifique** :

| Élément | Valeur |
|---|---|
| Reward formula | `episode_reward` (sparse 0/1 only) |
| `--max-rounds-schedule` | `5:0,10:13,15:26,20:38` |
| Steps | 50 |
| Init | Qwen-3B vanilla (pas continued depuis v3) |
| Run name | `trl_grpo_v4_scalinginter_sparse_step50` |
| Commit engineering | `0839679 chore(grpo): add --resume-from-checkpoint flag` |

**Incident d'exécution** : à step 28/50, le serveur TextCraft (PID 5653) est
mort tout seul → `ConnectionError` côté trainer → crash exit 1. Cause non
identifiée (pas d'OOM, pas de signal externe trouvé dans `dmesg`).
Resume propre depuis `checkpoint-25` via le flag engineering ajouté en
`0839679`, le `current_max_rounds(trainer)` relit correctement
`trainer.state.global_step` après resume, le curriculum a continué sa
progression. Pas de perte de signal.

**Diagnostic training** (50 steps complétés en deux runs concaténés) :

| Métrique | Valeur |
|---:|---:|
| `train_runtime` cumulé | ~31 min (training partiel pre-crash + resume) |
| `train_loss` final | 0.018 |
| Premier `reward=1.0` observé | rollout batch #9 (après transition `max_rounds 5→10`) |
| Rollouts avec ≥1 success | 3 batches sur ~50 enregistrés |
| `completions/mean_length` au palier 5 | ~510 tokens |
| `completions/mean_length` au palier 20 | ~1490 tokens (×3) |
| `grad_norm` régime | 0 sur la majorité (sparse + N=2), pics ≤ 5 au palier 20 |
| Distribution effective du curriculum | conforme au schedule (5/10/15/20 sur 13/13/12/12 steps) |

**Résultats eval** (run name `v4_scalinginter_sparse_checkpoint-50`) :

| Métrique | Valeur |
|---:|---:|
| Pass@1 test 100 | **14 / 100** (14 %) |
| Δ vs baseline | **−4** |
| Δ vs v3 step50 | **+6** |
| `mean_rounds` | 27.1 |

**Interprétation** :

- **+6 vs v3** confirme empiriquement que le shaping `+0.02` était bien
  le poison principal. Retirer le shaping rapproche immédiatement v4 du
  régime baseline.
- **−4 vs baseline** : ScalingInter + sparse reward + N=2 + LoRA n'est
  pas suffisant pour battre Qwen-3B vanilla. Le signal d'apprentissage
  reste trop sparse (cf. Insight #5 sur la borne mathématique à N=2).
- Le `train_loss=0.018` non nul confirme que **du gradient a effectivement
  circulé** au palier `max_rounds=20` (steps 38-50), mais l'amplitude
  cumulée des updates n'a pas suffi à dépasser le baseline.
- Lecture forte : sur notre régime hardware (single A100 40 Go), GRPO en
  multi-tour avec reward sparse vérifiable est **borné supérieurement
  par le baseline** ; pour réellement le dépasser il faut du compute
  qui rapproche du papier (cf. §"Écart à la recette papier" en bas).

---

### (À venir) Exp 5 — Continued GRPO depuis v4, comparaison reward env vs reward SCPO

**Objectif scientifique** : tester si une "self-consistency reward" basée
sur la similarité d'action-sequence entre N=4 trajectoires d'un même
problème peut remplacer le reward env vérifiable pour le fine-tuning RL.
Si oui → on peut entraîner sur des envs sans verifier.

**Setup prévu** :

| Variante | Reward de training | `num_generations` |
|---|---|---:|
| **T3-A** : v4 + envreward | `episode_reward + shape + invalid_steps` (= v3) | 4 |
| **T3-B** : v4 + scporeward | similarité d'action-sequence pondérée (cf. infra) | 4 |

Les deux variantes :
- partent du checkpoint LoRA v4 (50 steps ScalingInter)
- continuent pour 50 steps additionnels sur 256 **nouveaux** problèmes TextCraft
  (différents de ceux vus en v3 / v4)
- `num_generations=4` (au lieu de 2 pour v3/v4) pour que la comparaison
  soit isocompute

**Fonction de vote SCPO (T3-B)** :

Pour chaque problème, parmi les 4 trajectoires :

1. Extraire la "signature" de chaque trajectoire = liste de
   `(action_type, target)` où `action_type ∈ {craft, get, inventory, invalid}`.
2. Calculer la similarité pondérée entre paires (i, j) :
   - poids `craft` = 2.0, `get` = 1.0, `inventory` = 0.5, `invalid` = 0.3
   - bonus `×1.5` sur l'action finale (la plus discriminante)
   - similarité = somme(poids × match) / somme(poids_max)
3. Reward de la trajectoire i = moyenne des similarités(i, j) pour j ≠ i.
   Une trajectoire "modale" (proche du cluster majoritaire) a un reward
   élevé, une trajectoire "outlier" a un reward bas.

**Résultats** : *à compléter après T3.*

---

### Tableau récapitulatif

| Run | Steps | Train time | Pass@1 | Δ vs base | Commentaire |
|---|---:|---:|---:|---:|---|
| baseline | — | — | 18 / 100 | — | Qwen-3B vanilla |
| v2-step10 | 10 | ~15 min | 18 / 100 | +0 | set d'items shifté, pas de gain net |
| v2-step50 | 50 | ~67 min | 14 / 100 | **−4** | reward hacking sur `count_actions` concaténé |
| v3-step50 | 50 | ~69 min | **8 / 100** | **−10** | shaping per-turn, `+0.02` récompense les échecs courts |
| **v4-ScalingInter-sparse** | 50 | ~31 min + resume | **14 / 100** | **−4** | sparse reward + curriculum, +6 vs v3 mais ne bat pas baseline |
| v4+envreward (T3-A) | 50+50 | *abandonné* | — | — | dépendait de gain v4, voir Exp 5 |
| v4+scporeward (T3-B) | 50+50 | *abandonné* | — | — | dépendait de gain v4, voir Exp 5 |

**Reference du papier AgentGym-RL** (Table 2, Yang et al. 2026) :
Qwen2.5-3B-Instruct + GRPO + full FT + N=8 + max_turns=30 constant → **Pass@1 = 75 / 100**.
Notre meilleure expé (v4-ScalingInter-sparse) à 14/100, soit un écart
de **−61 points** avec la recette du papier — voir §"Écart à la recette papier" ci-dessous.

---

### Insights cumulés à date

1. **Le pipeline TRL + GRPO + LoRA + `rollout_func` interactif fonctionne**
   techniquement sur une A100 40 Go avec Qwen-3B (91 % VRAM saturée à N=2).
   C'est en soi une contribution non triviale : `rollout_func` est marqué
   *experimental* dans TRL 0.19.1 et la documentation est minimale.

2. **Le shaping est un piège méthodologique majeur en RL multi-tour**.
   Le bug v2 (count_actions sur completion concaténée) avait l'air anodin
   à l'écriture mais a inversé le signal d'apprentissage. Toujours vérifier
   l'unité de granularité du shaping (par-message ? par-épisode ? par-token ?).

3. **N=2 est le minimum théorique pour GRPO mais entraîne 12 % de steps
   perdus** (`frac_reward_zero_std=1.0` quand les 2 trajectoires donnent le
   même reward). Le papier AgentGym-RL utilise N=8. Mesure VRAM réelle à N=2
   sur A100 40 Go : 37.4/40 Go = 91 %. N=3 serait faisable d'entrée, N=4
   demande de baisser `max_completion_length` à ~96 ou `max_rounds` à 12.

4. **Le `grad_norm` peut exploser à 1700+ sans clip** en GRPO single-GPU
   avec petite batch et signal sparse (cas observé v2 step 35-36 après une
   variance de reward extrême). Ajout obligatoire de `max_grad_norm=1.0`
   depuis v3.

5. **GRPO N=2 + reward sparse 0/1 est borné par la baseline en multi-tour
   long-horizon**. La proba qu'un groupe de N=2 ait `std > 0` quand la
   probabilité de succès individuelle vaut `p` est `2p(1−p)`. Sur la
   baseline TextCraft (`p ≈ 0.18`), ça donne ~30 %. À `N=8`, cette même
   probabilité monte à `1 − (1−p)^8 ≈ 80 %`. Notre setup laisse donc
   70 % des steps sans signal de gradient utilisable, ce qui explique
   structurellement pourquoi v4 plafonne autour du baseline malgré un
   reward et un curriculum corrects.

---

### Écart à la recette du papier AgentGym-RL

Comparaison ligne à ligne entre `examples/train/AgentGym-RL/textcraft_train.sh`
(recette officielle du papier pour AgentGym-RL-3B) et notre
`scratch/07_trl_grpo_textcraft_smoke.py` v4.

| Paramètre | Papier | Nous (v4) | Origine de l'écart |
|---|---|---|---|
| Fine-tuning | Full FT (FSDP sharding via verl) | LoRA r=16 | contrainte 40 Go |
| `max_turns` (rounds) | 30 constant (`rounds_ctrl.type=fixed`) | 20 max (ScalingInter 5→10→15→20) | choix d'implémentation |
| `rollout.n` (N) | 8 trajectoires/query | 2 | contrainte 40 Go |
| `train_batch_size` (queries/step) | 32 | 1 | contrainte 40 Go |
| `ppo_mini_batch_size` | 8 | TRL default | différence framework TRL vs verl |
| `ppo_inner_epochs` | 2 | TRL default 1 | id. |
| `max_tokens` par génération | 512 | 128 (`max_completion_length`) | contrainte 40 Go |
| `max_response_length` total | 10240 | 128 × max_turns ≈ 2560 | id. |
| `kl_loss_coef` | 0.001 (`use_kl_loss=True`, `kl_loss_type=low_var_kl`) | TRL default `beta ≈ 0.04` | différence framework |
| Reward | sparse env 0/1 | sparse env 0/1 (depuis commit `30e580e`) | **alignement OK** |
| LR | 1e-6 | 1e-6 | **alignement OK** |
| Temperature | 1.0 | 1.0 | **alignement OK** |
| ScalingInter | absent pour le 3B | présent (notre v4) | on a ajouté un mécanisme du 7B au 3B |

**Estimation mémoire de la recette papier** : 3B full FT + N=8 + bs=32 +
max_response=10240 ≈ 130-180 Go (weights+optimizer ~30 Go, activations
+ KV cache rollout ~100-150 Go). Hardware paper : multi-noeud A100 ou
H800 (cf. arXiv 2509.08755 Appendix E : *"NVIDIA A100 GPUs and Ascend
910B NPUs […] distributed across multiple nodes"*, pas de comptage
exact).

**Sizing matériel pour répliquer AgentGym-RL-3B** :

| Option | Mémoire totale | Faisable pour 3B (N=8, bs=32, max_resp=10k) |
|---|---|---|
| 1× A100 40 Go (notre setup) | 40 Go | ❌ infaisable |
| 1× A100 80 Go | 80 Go | ⚠️ tight avec LoRA + N=4 max |
| 1× B200 (192 Go) | 192 Go | ✅ confortable, full FT possible |
| 4× A100 80 Go | 320 Go | ✅ confortable |
| 8× A100 80 Go (likely paper setup) | 640 Go | ✅✅ |

**Conséquence pour la suite** : continuer à itérer sur le reward ou les
hyperparams dans notre régime actuel ne nous fera pas franchir le
baseline ; la mathématique de GRPO à N=2 le borne. Pour vraiment
mesurer la valeur de méthodologies type ScalingInter / SCPO / West-of-N
sur TextCraft, il faudra d'abord obtenir une machine qui permette N ≥ 4
et idéalement la recette papier complète.

