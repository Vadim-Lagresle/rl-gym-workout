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

**Résultat eval** : *en cours (T1, lancé 2026-05-12 ~10:30, ETA ~11:20)*.

> *Note : cette section sera complétée dès que le worker T1 livre le Pass@1
> v3 step50 et la matrice de confusion 4-way.*

---

### (À venir) Exp 4 — GRPO v4 ScalingInter

**Objectif** : reproduire le curriculum max_rounds du papier AgentGym-RL,
mesurer le gain de la stratégie "court d'abord puis assouplir" vs le run
v3 à max_rounds=20 fixe.

**Setup prévu** :

- 4 paliers de `max_rounds` : 5 (steps 0-12), 10 (13-25), 15 (26-37), 20 (38-50)
- Tout le reste identique à v3
- Run name : `trl_grpo_v4_scalinginter_step50`
- Baseline d'ablation `max_rounds=10` fixe à lancer en nuit pour comparaison séparée

**Résultats** : *à compléter après T2.*

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

### Tableau récapitulatif (à compléter)

| Run | Steps | Train time | Pass@1 | Δ vs base | Commentaire |
|---|---:|---:|---:|---:|---|
| baseline | — | — | 18 / 100 | — | Qwen-3B vanilla |
| v2-step10 | 10 | ~15 min | 18 / 100 | +0 | set d'items shifté, pas de gain net |
| v2-step50 | 50 | ~67 min | 14 / 100 | **−4** | reward hacking, modèle apprend "1-shot or abandon" |
| **v3-step50** | 50 | ~69 min | *en cours* | *en cours* | fix shaping per-turn |
| v4-ScalingInter | 50 | *prévu* | *à mesurer* | *à mesurer* | curriculum 5→10→15→20 |
| v4+envreward | 50+50 | *prévu* | *à mesurer* | *à mesurer* | continued GRPO standard |
| v4+scporeward | 50+50 | *prévu* | *à mesurer* | *à mesurer* | continued GRPO avec vote SCPO |

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

