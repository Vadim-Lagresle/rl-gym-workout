## Résultats expérimentaux — RL agentique sur TextCraft

Ce document rassemble les expériences conduites, leur protocole exact, et
leurs résultats numériques. Il est conçu pour être autonome : on doit
pouvoir lire une section sans avoir à consulter le `WORKLOG.md` (qui lui
est un journal chronologique, pas un référentiel de résultats).

> **Note réorganisation 2026-07-16** : `runs/` est désormais classé par famille
> (`0_baselines/ … 9_legacy_pre_b200/`, voir `runs/INDEX.md`). Les chemins
> `runs/exp…` mentionnés dans les sections historiques ci-dessous s'y retrouvent
> préfixés par leur famille (ex. `runs/exp16_*` → `runs/8_single_turn_exp16/{core,blind,sweep_*}/exp16_*`).
> Côté scripts : `eval_vllm.py`/`eval_fullft.py` → `src/eval/eval_textcraft.py`
> (`--backend vllm|hf`), pipeline exp16 → `src/eval/single_turn/`.

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
| Métrique principale | **Taux de réussite** sur le test set ; souvent noté *Pass@1* dans les scripts quand **un seul rollout indépendant** est tiré par problème (voir §Exp 6 — ce n’est pas la limite de tours LLM↔env). |

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

### Exp 6 — Réplication papier upstream verl sur 4× A100 40 GB (Qwen-3B, GRPO, sans ScalingInter)

**Objectif scientifique** : valider empiriquement si le code upstream
`AgentGym-RL` (= verl + l'extension multi-tour vLLM des auteurs) est
*directement réutilisable* tel quel sur notre matériel, et mesurer le
Pass@1 obtenu. C'est la première fois qu'on quitte notre stack
TRL+LoRA+rollout_func custom pour faire tourner exactement la recette
de l'article. On choisit volontairement **`rounds_ctrl.type=fixed`**
(pas de ScalingInter) pour avoir une baseline GRPO pure comparable à la
ligne *AgentGym-RL-3B = 75/100* de la Table 3 du papier.

**Réutilisabilité du code upstream — verdict avec preuves** :

| Vérification | Statut | Détails |
|---|---|---|
| Math GRPO offline (advantages, loss_mask, prompt train/eval) | ✅ | 3/3 PASS dans `scratch/smoke_verl_test.py` (cf. Exp 5 *Audit verl*) |
| Eval `main_generation` 1× A100 40 GB | ❌ | SIGSEGV NCCL à `world_size=1`, doc Exp 5 |
| Eval `main_generation` 4× A100 40 GB | ⚠️ non re-testé (le training prime) |
| **Training `main_ppo` 4× A100 40 GB**, Qwen-3B, recette papier | ✅ **après 3 workarounds** | cf. ci-dessous |

Le code est réutilisable **moyennant trois workarounds non-triviaux** qui
n'apparaissent nulle part dans le README upstream :

1. **NCCL gIB sur image GCP A100**.
   `/etc/profile.d/env.sh` source `/usr/local/gib/scripts/set_nccl_env.sh`
   qui force `NCCL_NET=gIB` + une dizaine de tuner flags InfiniBand pour
   les clusters A3-Ultra. Sur une box 4× A100 40 GB single-node, ce
   plugin tente de charger `libibverbs` (non installé), puis crash
   silencieux NCCL → `ray.exceptions.ActorDiedError` avec
   `SYSTEM_ERROR exit code 2` immédiatement après la ligne
   `NCCL version 2.20.5+cuda12.4`. Symptôme **identique** à celui qu'on
   avait observé en single-GPU et attribué au "code path verl pas testé".
   *C'était en fait un bug d'image GCP, pas un bug verl.*
   **Fix** : avant `python3 -m verl.agent_trainer.main_ppo`, exécuter :
   ```bash
   unset NCCL_NET NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC \
         NCCL_IB_* NCCL_NVLS_CHUNKSIZE NCCL_P2P_NET_CHUNKSIZE
   export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' \
                              | grep -v '/usr/local/gib' | paste -sd: -)"
   export NCCL_NET=Socket
   export NCCL_IB_DISABLE=1
   ```
   Le NVLink intra-node fonctionne quand même par-dessus le contrôle TCP.

2. **`load_format=dummy_dtensor` obligatoire pour le training**.
   L'enum `LoadFormat` dans `verl/third_party/vllm/vllm_v_0_6_3/config.py`
   ne contient **pas** `safetensors` (que notre script local
   `textcraft_train.local.sh` forçait pour contourner le crash NCCL
   en single-GPU). Valeurs valides : `auto`, `hf`, `dtensor`, `megatron`,
   `dummy_hf`, `dummy_megatron`, `dummy_dtensor`. En training, on garde
   le défaut `dummy_dtensor` : vLLM init avec poids aléatoires puis se
   fait patcher en place par le `FSDPVLLMShardingManager` à chaque batch
   de génération.

3. **VRAM 40 GB demande 3 concessions vs la recette papier**. Sans ces
   trois flags, OOM à la première backward (37.66 GB déjà occupés,
   manque 4.6 GB) :
   - `actor_rollout_ref.model.enable_gradient_checkpointing=True`
   - `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`
     (push les states AdamW sur CPU — économie ~10 GB par GPU)
   - `actor_rollout_ref.rollout.gpu_memory_utilization=0.4` (down de 0.7)
   - `data.max_response_length=4096` (down de 10240)

**Setup spécifique** :

| Élément | Papier (textcraft_train.sh upstream, AgentGym-RL-3B) | Notre Exp 6 |
|---|---|---|
| Modèle | Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct ✅ |
| Hardware | cluster (probablement ≥ 8× A100 80 GB) | 4× A100 **40 GB** single-node |
| `rollout.n` (N) | 8 | **8** ✅ |
| `train_batch_size` | 32 | **8** ⚠️ |
| `ppo_mini_batch_size` | 8 | 8 ✅ |
| `ppo_epochs` | 2 | 2 ✅ |
| `max_response_length` | 10240 | **4096** ⚠️ |
| `max_tokens` / turn | 512 | 512 ✅ |
| `rollout.tensor_model_parallel_size` | 1 | 1 ✅ |
| `kl_loss_coef`, `kl_loss_type` | 0.001, low_var_kl | 0.001, low_var_kl ✅ |
| `lr` | 1e-6 | 1e-6 ✅ |
| `rounds_ctrl.type` | fixed | fixed ✅ |
| `rounds` | 30 | 30 ✅ |
| Fine-tuning | full FT + FSDP | full FT + FSDP ✅ |
| Gradient checkpointing | non (probable) | **oui** ⚠️ |
| Optimizer offload to CPU | non | **oui** ⚠️ |
| Total training steps | non capé (`total_epochs=30` → ≈ 12 × 30 = 360 steps avec bs=32) | **50 steps** ⚠️ (budget compute du run) |

Script reproductible : `examples/train/AgentGym-RL/textcraft_train.4gpu.sh`.
Lancement : `bash examples/train/AgentGym-RL/textcraft_train.4gpu.sh`
(50 steps par défaut ; `SMOKE=1` pour un test 1-step ; `TOTAL_STEPS=N`
pour override).

**Smoke test (1 step) — VRAI succès** (run dir
`saves/agentgym_rl_4gpu/smoke_4gpu_v4`) :

| Métrique | Valeur |
|---|---:|
| `critic/task_score/mean` step 1 | **0.016** (1 succès / 8 rollouts × 8 GPUs = ~1/64) |
| `critic/task_score/max` step 1 | 1.0 |
| `actor/grad_norm` step 1 | 0.421 |
| `actor/pg_loss` step 1 | 0.044 |
| `actor/kl_loss` step 1 | 0.001 |
| `actor/entropy_loss` step 1 | 1.157 |
| `response_length/mean` step 1 | 2065 tokens |
| `timing_s/step` | **284 s** (gen 248 s + ref 10 s + update 22 s) |
| GPU memory used (FSDP+optim CPU offload+vLLM 40 %) | ~23 GB / 40 GB par GPU |

Pas d'OOM, pas de crash NCCL, pas d'erreur Ray. Le pipeline est stable
et la marge VRAM (~17 GB libres / GPU) laisse de la place pour scaler.

**Résultats du run 50 steps + eval sur `global_step_50`** (merge FSDP → HF,
`scratch/03_eval_qwen.py`, vLLM standard, **2026-05-13**) :

| Métrique | Papier (AgentGym-RL-3B, ~120 steps, Table 3) | Mesure Exp 6 (50 steps, 4× A100 40 GB) |
|---|---:|---:|
| **Succès / 100** (un rollout par item) | **75 / 100** | **38 / 100** (38 %) |
| Libellé script | *Pass@1* | identique : **1 essai complet / problème**, pas « 1 tour » |
| Tours max LLM↔env (eval) | 30 | **30** (`MAX_ROUNDS` dans `03_eval_qwen.py`) |
| `mean_rounds` (moyenne sur 100 épisodes) | — | **21.2** (agrégat `scores_summary.json`) |
| `train_loss` final (dernier step loggué) | non communiqué | extraire depuis `run.log` local si besoin |
| Somme `timing_s/step` (49 steps loggués) | — | **≈ 9617 s** (~2 h 40) de boucles step chronométrées |

**Artefacts versionnés** (dans le dépôt) :

- `scratch/eval_logs_4gpu_global_step_50/textcraft_*.json` — trajectoires complètes.
- `scratch/eval_logs_4gpu_global_step_50/scores_table.csv` — tableau par item.
- `scratch/eval_logs_4gpu_global_step_50/scores_summary.json` — totaux.
- `scratch/training_curves.png`, `scratch/training_losses.png` — courbes depuis `run.log` (fichier log sous `saves/`, non versionné).

**Clarification « Pass@1 » vs 30 rounds** : le papier autorise jusqu’à **30 interactions** par épisode (comme notre eval). Le **@1** signifie **un seul tirage stochastique complet par item de test**, au sens *HumanEval-style* (k programmes / k trajectoires **indépendantes** pour le même exercice). Ce n’est **pas** « une seule action » ni « un seul tour » : une trajectoire peut utiliser jusqu’à 30 tours ; le compte **38/100** est le nombre de problèmes résolus avec **une** trajectoire chacun.

**Bornes attendues** (inchangées par rapport à la rédaction précédente) :

- Le papier atteint 75/100 à la fin de **~30 epochs** ≈ 360 steps avec
  `train_batch_size=32`. Au step 50 avec notre `train_batch_size=8`,
  on a vu seulement `50 × 8 = 400 queries` (vs `50 × 32 = 1600` chez
  le papier au même step). On est donc à **~7×** moins d'exposition à
  data train.
- Ordre de grandeur réaliste pour notre 50-step run : Pass@1 ≈ **30-50 / 100**
  (entre baseline 18 et papier final 75), avec la pente la plus rapide
  en début de training. Pour atteindre 75/100 il faudrait probablement
  300-360 steps ≈ 28 h sur ce hardware.

**Constat (50 steps)** : **38/100** dépasse nettement le baseline TRL **18/100**,
ce qui confirme que la recette verl (N=8, full FT, GRPO upstream) exploite
mieux le signal que notre stack TRL+LoRA contrainte — tout en restant **sous**
le **75/100** papier (moins de steps, `train_batch_size` réduit, `max_response_length` abaissé, cf. tableau setup Exp 6).

---

### Analyse des résultats Exp 6 par profondeur de crafting (session 2026-05-18)

Le test set TextCraft (100 items) contient des problèmes de 4 niveaux de
difficulté (« depth ») déterminés par la profondeur minimale dans l'arbre de
crafting Minecraft. La distribution est : **31 items depth 1, 41 depth 2,
25 depth 3, 3 depth 4** (vérifiée via `agentenv_textcraft.CraftingTree.item_recipes_min_depth`).

#### Résultats par depth — Exp 6 vs papier AgentGym-RL-3B

| Depth | Items | Nos succès | Notre taux | Papier 3B | Écart |
|---|---|---|---|---|---|
| 1 | 31 | 27 | **87.1 %** | 100 % | −13 % |
| 2 | 41 | 10 | **24.4 %** | 90.2 % | −66 % |
| 3 | 25 | 1 | **4.0 %** | 28.0 % | −24 % |
| 4 | 3 | 0 | **0.0 %** | 0.0 % | = |
| **Total** | **100** | **38** | **38.0 %** | **75.0 %** | **−37 %** |

Vérification : 31×1.00 + 41×0.902 + 25×0.28 + 3×0 = 75 ✅ (cohérent avec Table 3 papier).

Depth 4 : **mur universel**. Même GPT-4o, o3, o4-mini font 0/100. Seuls
Gemini-2.5-Pro, Qwen3-8B/32B et ScalingInter-7B passent 1 item sur 3
(33.33 %). Les 3 items depth 4 du test set sont : `polished_granite_slab`,
`polished_andesite_stairs`, `lodestone`.

#### Gap hardware — pourquoi 38 et non 75

Le train set contient **374 items**. Avec `train_batch_size=8`, une epoch
= ceil(374/8) ≈ 47 steps. Nos **50 steps ≈ 1 epoch**. Le papier fait
`total_epochs=30` avec `train_batch_size=32` → 30 × ceil(374/32) ≈ **360
steps ≈ 30 epochs**. L'évaluation du papier est au checkpoint `global_step_150`
≈ **12.5 epochs**. Résumé :

| | Papier (ckpt éval step 150) | Nous (50 steps) |
|---|---|---|
| Epochs vues | ~12.5 | ~1 |
| Queries totales | 150 × 32 = **4 800** | 50 × 8 = **400** |
| Ratio | **12× plus** | — |

En plus de l'exposition data, deux contraintes matérielles dégradent la
qualité d'apprentissage par step :

- **`max_response_length=4096`** (vs 10240 papier) : 4096 ÷ 512 tokens/tour
  ≈ **~8 tours max par épisode pendant le training**. Les items depth 2+
  qui nécessitent 10-20 tours n'ont jamais produit de reward positif pendant
  l'entraînement → pas de signal de gradient sur ces trajectoires.
- **`max_model_len=8192`** (vs 32768 papier eval) : les conversations longues
  sont tronquées en KV-cache dès 8192 tokens total, soit ~16 tours × 512
  tokens. Impacte les épisodes qui vont au bout du timeout (30 tours).

#### Patterns d'échec identifiés par analyse des logs (session 2026-05-18)

Quatre logs examinés manuellement :
`textcraft_0` (depth 1 ✅), `textcraft_12` (depth 1 ✗),
`textcraft_173` (depth 2 ✅), `textcraft_140` (depth 2 ✗).

**Pattern 1 — Distraction par recettes alternatives (depth 1)**
Exemple : `textcraft_12`, goal = `sugar`. Le modèle a obtenu `1 sugar_cane`
dès le tour 1 (recette directe disponible : `craft 1 sugar using 1 sugar_cane`),
mais a ignoré cette recette au profit d'une alternative visible dans le
contexte (`craft 3 sugar using 1 honey bottle`). Il a passé 28 tours à
chercher une honey bottle introuvable, puis a halluciné des items inexistants
(`mayonnaise_pancake`, `sweetcherry_jam`, `scallion_soup`) avant d'échouer.
**Cause** : trop de recettes dans le contexte → le modèle choisit la mauvaise
et ne revient pas sur ses pas même quand il possède déjà les ingrédients de
la recette simple.

**Pattern 2 — Absence de raisonnement transitif (depth 2+)**
Exemple : `textcraft_140`, goal = `sandstone_stairs` (chemin : `4 sand →
craft sandstone → craft 4 sandstone_stairs using 6 sandstone`). Le modèle
a tenté `get 6 sandstone` → introuvable (normal : sandstone n'existe pas dans
l'environnement, il doit être crafté). Face à l'échec, il n'a **jamais déduit**
"sandstone est introuvable → je dois le crafter depuis sand". Il a cherché de
la `purple_dye` et `white_dye` sans rapport, tenté des recettes invalides
(`craft 4 cut sandstone using 4 sand`), et n'a jamais essayé
`craft 1 sandstone using 4 sand` malgré du sable dans son inventaire.
**Cause** : le modèle ne fait pas le lien entre "item introuvable par `get`"
et "item qui doit être crafté depuis ses constituants". Ce raisonnement
transitif est exactement ce qui est requis à depth 2, 3 et 4.

**Pattern 3 — Hallucination d'items inexistants (après ~15 tours d'échec)**
Quand le modèle est bloqué depuis de nombreux tours, il commence à inventer
des items (`mayonnaisepancake`, `scallionsoup`, etc.) qui ne figurent pas dans
les recettes fournies. Signe d'effondrement du raisonnement sous contrainte
d'horizon long.

**Succès depth 2 — contre-exemple positif**
`textcraft_173`, goal = `light_gray_stained_glass` (chemin : `get glass →
light_gray_dye introuvable → get azure_bluet → craft light_gray_dye →
craft stained_glass`). Le modèle a correctement identifié qu'il manquait un
précurseur, cherché la fleur qui produit le dye, et chaîné les deux crafts en
5 tours. **C'est exactement le pattern depth-2 qu'il faut généraliser.**

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

### Exp 8.2 — GRPO ScalingInter batch papier sur B200 (TRL + vLLM in-process)

**Objectif** : aligner le batch GRPO sur la recette papier TextCraft
(`train_batch_size=32`, `N=8` → 256 trajectoires/step) avec un budget
ScalingInter réduit (20 epochs, ~220 steps) sur B200 192 Go.

**Setup spécifique** :

| Élément | Valeur |
|---|---|
| Fine-tuning | Full FT (`--full-ft`) |
| `num_generations` (N) | 8 |
| `gradient_accumulation_steps` | 256 (32 prompts × N=8) |
| `max_rounds_schedule_epochs` | `6:0,12:4,18:8,24:12` |
| `num_epochs` | 20 (~220 steps) |
| Eval in-loop | tous les 11 steps (1×/epoch) |
| Checkpoint | `_best` uniquement (`save_steps=100000`) |
| Run name | `exp8.2_paper_batch_b200` |

**Résultats training** :

| Métrique | Valeur |
|---:|---:|
| Steps complétés | 220 / 220 |
| Wall time | ~18 h 24 |
| Pass@1 best (in-loop eval) | **20 / 100** @ step 187 |
| Pass@1 final (step 220) | 16 / 100 |
| Checkpoint conservé | `saves/trl_grpo/exp8.2_paper_batch_b200_best/` |

**Résultats eval in-loop (best @ step 187)** :

| Depth | Pass@1 |
|---|---:|
| 1 | 61 % |
| 2 | 0 % |
| 3 | 0 % |
| 4 | 0 % |

**Verdict** : léger gain vs baseline TRL (+2 pts best vs 18/100), loin du
papier 75/100. Depth 2 apparaît faiblement en fin de run (5 % @ step 220)
mais pas au checkpoint best. Comparaison exp8.1 (64 traj/step, best in-loop
25 % non sauvé faute de disque) : le batch papier n'apporte pas de saut net.

---

### Exp 16 — Raisonnement single-turn (plan oracle, sans entraînement)

**Objectif** : isoler la capacité de **planification one-shot** du modèle base,
indépendamment du contrôle multi-tour Thought/Action.

**Protocole en deux phases** :

| Phase | Script | Description |
|---|---|---|
| 1 | `collect_single_turn_plans.py` | recettes + goal → plan texte libre (100 items) |
| 2 | `replay_single_turn_plans.py` | plan → extraction JSON actions → replay TextCraft |

Phase 2 : 1 appel LLM/item pour convertir le plan en actions, puis exécution
déterministe dans l'env (sans LLM entre les steps). Métrique = **pass@1 oracle**.

#### Exp 16a — Qwen2.5-3B-Instruct (2026-06-24)

| Métrique | Valeur |
|---:|---:|
| **Pass@1 global** | **20 / 100 (20 %)** |
| Depth 1 | 13 / 31 (41.9 %) |
| Depth 2 | 7 / 41 (17.1 %) |
| Depth 3 | 0 / 25 (0 %) |
| Depth 4 | 0 / 3 (0 %) |
| Erreurs extraction JSON | 11 / 100 |
| Wall time (collect + replay) | ~6 min |

Artefacts : `runs/8_single_turn_exp16/core/exp16_single_turn_reasoning/{plans,replay_logs}/`.

#### Exp 16b — Qwen3.5-4B (2026-06-24)

| Métrique | Valeur |
|---:|---:|
| **Pass@1 global** | **54 / 100 (54 %)** |
| Depth 1 | 24 / 31 (77.4 %) |
| Depth 2 | 22 / 41 (53.7 %) |
| Depth 3 | 8 / 25 (32.0 %) |
| Depth 4 | 0 / 3 (0 %) |
| Erreurs extraction JSON | 2 / 100 |
| Wall time replay | ~381 s |

Backend HF `generate()` (`--backend hf --no-thinking`), vLLM non supporté pour
Qwen3.5. Artefacts : `runs/8_single_turn_exp16/core/exp16_qwen35_4b/{plans,replay_logs}/`.

#### Comparaison oracle vs multi-tour (même modèle Qwen3.5-4B)

| Protocole | Pass@1 | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|---:|
| **Exp 16b oracle single-turn** | **54 / 100** | 77 % | 54 % | 32 % | 0 % |
| **Exp 15 multi-tour interactif** | **77 / 100** | 97 % | 81 % | 56 % | 0 % |
| Δ oracle − multi-tour | **−23 pts** | −20 pts | −27 pts | −24 pts | = |

**Paradoxe apparent** : l'oracle devrait être une *borne supérieure* — le modèle
voit toutes les recettes, peut réfléchir sans contrainte de format, puis exécute
sans erreur de parsing inter-tours. Pourtant **54/100 < 77/100**.

**Interprétation** (pas un problème de mémoire de contexte) :

1. **Feedback env absent en oracle** : le multi-tour (Exp 15, `mean_rounds=13.9`,
   bien en dessous de la limite 30 tours) reçoit l'inventaire, les erreurs
   (`get` impossible → essayer un craft), et peut corriger en cours de route.
   L'oracle doit produire un plan *et* une séquence d'actions correcte *sans*
   aucune observation intermédiaire.

2. **Deux points de défaillance en oracle** : plan texte (phase 1) **puis**
   extraction JSON (phase 2). Le multi-tour n'a qu'un format Thought/Action par
   tour, avec validation immédiate à chaque step.

3. **Le multi-tour est plus facile que le one-shot parfait** sur TextCraft, tant
   que le contexte tient (≤30 tours, ~14 en moyenne). Ce n'est pas contradictoire
   avec la littérature agentique : l'interaction compense les erreurs de
   raisonnement transitif (cf. Pattern 2 Exp 6).

4. **Écart modèle-dépendant** : Qwen2.5 oracle ≈ baseline multi-tour (20 vs 18).
   Qwen3.5 oracle << multi-tour (54 vs 77) : le modèle plus fort *profite davantage*
   du feedback iteratif que de la planification upfront.

**Prochaine analyse** : classifier échecs plan vs extraction vs exécution dans
`replay_logs/` (notamment les 46 items où Qwen3.5 multitour réussit mais oracle
échoue).

#### Exp 16 — addendum audit & température (2026-07-08)

Audit complet du pipeline single-turn (détails : `docs/hebdo/10juillet/`). Trois
corrections/ajouts qui changent la lecture des scores 20 et 54 ci-dessus.

**(a) L'extracteur JSON est un solveur partiel — ablation « aveugle ».**
Le prompt d'extraction voit recettes + goal et peut *compléter/réparer* le plan.
En le privant de recettes+goal (traduction littérale, flag `--blind`) :

| Modèle | Extracteur informé | Extracteur aveugle | part « réparation » |
|---|---:|---:|---:|
| Qwen2.5-3B | 20 / 100 | **12 / 100** | 40 % du score |
| Qwen3.5-4B | 54 / 100 | **46 / 100** | 15 % du score |

→ « reasoning pur » réel encadré : 3B ∈ [12, 20], 4B ∈ [46, 54]. Le 4B dépend
bien moins de l'extracteur (plans auto-suffisants) → argument fort qu'il *planifie*
mieux, pas seulement qu'il score mieux. Les « erreurs d'extraction » (11 pour le
3B) ne sont **pas** de la troncature récupérable mais de la **dégénérescence en
boucle** (répétition ad infinitum à basse température) — échecs légitimes.

**(b) Balayage fin de température T=0.0→1.0 (pas 0.1) — image corrigée (2026-07-09).**
Balayage complet en reasoning-pur informé, 1 tirage/T, sur base 3B et best58
(11 T chacun) + Qwen3.5-4B partiel (T=0.0→0.6). Runs : `runs/8_single_turn_exp16/sweep_{base,b58,4b}/`.

| T | base 3B | best58 | 4B (partiel) |
|---:|---:|---:|---:|
| 0.0 | 11 | 21 | 56 |
| 0.1 | 15 | 15 | 54 |
| 0.2 | 15 | 18 | 54 |
| 0.3 | 11 | 24 | 52 |
| 0.4 | 11 | 18 | 52 |
| 0.5 | 14 | 24 | 52 |
| 0.6 | 11 | 18 | 57 |
| 0.7 | 10 | 21 | — |
| 0.8 | 13 | 18 | — |
| 0.9 | 14 | 20 | — |
| 1.0 | 10 | 17 | — |
| **moy** | **12.3** | **19.5** | **53.9** (T≤0.6) |
| σ / min-max | 1.9 / 10-15 | 2.7 / 15-24 | 1.9 / 52-57 |

**Ce balayage corrige DEUX conclusions précédentes tirées de points isolés :**

- **⚠️ Correction 1 — pas d'effet de température détectable (invalide l'ancien « −5/−6 »).**
  Chaque modèle reste dans une **bande de bruit plate** de 0 à 1 (base 10-15, best58
  15-24, 4B 52-57), sans tendance. Le greedy (T=0) n'est ni meilleur ni pire. La
  « pénalité −5/−6 de 0.7→1.0 » annoncée précédemment venait de comparer **deux
  tirages isolés bruités** (ex. base 0.7=20 était un tirage chanceux ; refait ici =10).
  Meta-leçon : **un pass@1/100 en tirage unique a ~±5 de bruit run-à-run** — trop
  pour résoudre un effet de 0,1 en température. (La déduplication « single-turn <
  multi-tour » reste vraie, mais pas via un argument de température.)

- **⚠️ Correction 2 — le RL A transféré à la planification (invalide l'ancien « 19≈20 »).**
  best58 (moy **19.5**) bat la base (moy **12.3**) à **10 configs sur 11**
  (1 égalité, 0 défaite), écart moyen **+7.2**, **test des signes p=0.002**.
  L'ancien « 19 ≈ base 20, aucun transfert » comparait un tirage haut de la base
  (20) à un tirage bas de best58 (19). Sur 11 tirages la vérité est robuste :
  **le GRPO a bien amélioré la planification one-shot (+7 pts)**.

**(c) Nuance : transfert réel mais modeste — le gain RL reste surtout réactif.**
best58 : **+7** en reasoning-pur single-turn (12→19) contre **+36** en multi-tour
(18→54). Le GRPO outcome-only améliore un peu la planification a priori, mais
l'essentiel du gain est dans la **politique réactive** (agir → observer l'erreur →
corriger). Pour franchir depth 3-4, injecter la planification explicitement
(CoT tour 1, SFT sur plans, distillation) reste pertinent.

> **Biais de sélection sur le best** : le « 58/100 » d'exp10.8 (step 368) est un
> pic isolé d'une courbe d'éval mono-tirage bruitée (steps voisins : 40, 49, 58,
> 49). Re-évaluation indépendante = **54/100**. Le vrai niveau du checkpoint est
> dans les bas-50 ; le 58 était du bruit capté vers le haut (*winner's curse*).
> Même cause que la Correction 1 : le bruit mono-tirage sur 100 items (~±5)
> fausse toute lecture d'un point unique — best sélectionné, comparaison de T,
> ou comparaison de deux modèles proches. Remède : multi-seed / pass@k.

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
| **Exp 6 verl 4× A100** | 50 | ~2 h 40 (somme `timing_s/step` sur 49 logs) + overhead | **38 / 100** | **+20** vs baseline 18 | checkpoint `global_step_50`, full FT, N=8, `rounds=30`, cf. §Exp 6 |
| **Exp 8.2 ScalingInter batch papier (B200)** | 220 | ~18 h 24 | **20 / 100** (best @187) | **+2** vs baseline 18 | full FT, N=8, 256 traj/step, cf. §Exp 8.2 |
| **Exp 16a single-turn oracle (Qwen2.5-3B)** | — | ~6 min | **20 / 100** | **+2** vs baseline 18 | plan one-shot + replay TextCraft, cf. §Exp 16 |
| **Exp 16b single-turn oracle (Qwen3.5-4B)** | — | ~6 min | **54 / 100** | **+36** vs baseline 18 | oracle **−23 pts** vs Exp 15 multitour 77/100 |
| **Exp 15 — Qwen3.5-4B baseline** | — | — | **77 / 100** | **+59** vs baseline 18 | sans entraînement, multitour interactif, non-thinking |

**Reference du papier AgentGym-RL** (Table 2, Yang et al. 2026) :
Qwen2.5-3B-Instruct + GRPO + full FT + N=8 + max_turns=30 constant → **Pass@1 = 75 / 100**.
Notre meilleure expé (v4-ScalingInter-sparse) à 14/100, soit un écart
de **−61 points** avec la recette du papier — voir §"Écart à la recette papier" ci-dessous.

**Exp 15 — Qwen3.5-4B baseline (sans entraînement)** : **77 / 100** — dépasse l'objectif papier.
Implication : un meilleur modèle de base suffit à franchir 75/100 sans RL.
Depth 4 reste un mur universel (0/3). Voir §Exp 15 pour le détail.

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

---

### Exp 15 — Baseline Qwen3.5-4B sur TextCraft (sans entraînement)

**Objectif** : mesurer si un modèle de base plus récent et plus grand relève le plancher,
avant toute décision sur le RL. Hypothèse : si 77/100 est atteignable sans training,
le signal GRPO à exploiter est très différent du cas Qwen2.5-3B (18/100 baseline).

**Setup** :

| Élément | Valeur |
|---|---|
| Modèle | Qwen/Qwen3.5-4B (hub HF) |
| Architecture | `Qwen3_5ForConditionalGeneration` (hybride Gated DeltaNet + attention) |
| Post-training | SFT + RL Qwen (thinking + agentic), thinking désactivé (`--no-thinking`) |
| Stack eval | HF `generate()` via `eval_fullft.py` (vLLM incompatible : archi trop récente) |
| GPU | B200 192 Go |
| `max_rounds` | 30 |
| `max_new_tokens` | 512 |
| Temperature | 1.0 |
| `HF_HOME` | `/tmp/hf_cache` (overlay 221 Go) |

**Résultats** :

| Depth | Items | Pass@1 | Taux |
|---|---|---|---|
| 1 | 31 | 30 | 96.8 % |
| 2 | 41 | 33 | 80.5 % |
| 3 | 25 | 14 | 56.0 % |
| 4 | 3 | 0 | 0.0 % |
| **Total** | **100** | **77** | **77.0 %** |

Wall time : 9560 s (~2h39). Mean rounds : 13.9. Erreur dominante : `format_error` (51 % recovery).

**Comparaison avec Exp 6 (Qwen2.5-3B après 50 steps RL verl)** :

| | Qwen2.5-3B + 50 steps GRPO | Qwen3.5-4B baseline |
|---|---:|---:|
| Depth 1 | 27/31 (87 %) | 30/31 (97 %) |
| Depth 2 | 10/41 (24 %) | 33/41 (80 %) |
| Depth 3 | 1/25 (4 %) | 14/25 (56 %) |
| Depth 4 | 0/3 | 0/3 |
| **Total** | **38/100** | **77/100** |

**Conclusion** : Qwen3.5-4B sans training dépasse non seulement Exp 6 mais aussi
l'objectif du papier (75/100). Depth 4 reste un mur universel (0/3) même pour ce modèle.
Le RL sur Qwen3.5-4B aurait une baseline de départ bien plus haute que sur Qwen2.5-3B,
mais la question se déplace : quelle marge de progression reste-t-il à exploiter par GRPO ?

**Note Exp 16b** : le même modèle en protocole oracle single-turn n'atteint que
**54/100** (−23 pts vs ce multi-tour). Voir §Exp 16 — le feedback env rend le
multi-tour *plus facile* que la planification one-shot parfaite, sans artefact
de limite de contexte (`mean_rounds=13.9`).

---

### Audit verl + smoke tests (préalable à la migration 8× A100)

**Contexte** : avant de provisionner une VM 8× A100 40 Go pour répliquer la
recette papier avec **verl** (= la stack `AgentGym-RL/verl` non patchée),
on a fait deux smoke tests pour vérifier (a) que l'algo de verl est
correct et (b) qu'on peut réellement faire tourner leur eval pipeline.

#### Smoke (a) — Tests offline sur `verl` (`scratch/smoke_verl_test.py`)

Trois unit tests CPU-only, ~12 s total :

| Test | Cible | Résultat |
|---|---|---|
| `compute_grpo_outcome_advantage` | math de la baseline relative GRPO sur 3 prompts (mixte / tout 0 / tout 1) | ✅ advantages normalisés à `±0.866` quand `std > 0`, à `0` sinon ; `returns == advantages` |
| `RolloutHandler.add_assistant_message` | construction du `loss_mask` token-level (Qwen ChatML) | ✅ longueurs cohérentes, prompt `mask=0`, 11 tokens `mask=1` couvrant *exactement* le contenu assistant |
| `RLHFDataset` prompt vs `scratch/03_eval_qwen.py` prompt | parité train/eval (pas de drift, pas de few-shot caché) | ✅ **prompts identiques byte-for-byte** (1382 chars chacun) — mêmes règles TextCraft, même ACK assistant canonique |

**Conclusion** : la machinerie verl est mathématiquement correcte et le
prompt qu'elle voit en training est exactement celui qu'on utilise en
eval. Pas de bricolage caché côté upstream.

#### Smoke (b) — Eval verl on-GPU sur 1× A100 40 Go (`examples/eval/textcraft_eval.local.sh`)

| Étape | Statut |
|---|---|
| TextCraft serveur up (port 36005) | ✅ |
| Dataset eval recopié (100 items) | ✅ |
| Worker Ray spawn + Qwen-3B chargé (CPU, checkpoint shards) | ✅ |
| **NCCL init (rank=1 sur 1)** | ❌ **SIGSEGV silencieux** |

Trace pertinente (`/tmp/smoke_verl_eval.log`) :

```
(ActorRolloutRefWorker pid=152894) NCCL version 2.20.5+cuda12.4
(raylet) A worker died or was killed... Worker exit type: SYSTEM_ERROR
ray.exceptions.ActorDiedError: ... pid: 152894 ...
```

`dmesg` ne montre **aucun OOM-killer**, la VRAM était à 0 MiB avant
lancement. Le crash arrive **immédiatement après** le print de la
version NCCL et **avant** tout chargement vLLM des poids sur GPU. Ce
n'est donc ni un problème mémoire, ni un problème de poids — c'est
l'init du process group NCCL à `world_size=1` qui plante.

**C'est exactement la même panne qu'en Phase B il y a 3 semaines** (cf.
WORKLOG, "Error 4 : Ray ActorDiedError pendant
`verl.agent_trainer.main_generation`"). On confirme donc empiriquement
ce que l'audit code-level avait prédit : le code path single-GPU de
verl n'est pas testé / pas fonctionnel, et patcher demanderait des
jours sans garantie.

#### Implications pour la suite

| Question | Réponse |
|---|---|
| L'algo verl est-il bon ? | **Oui** (smoke (a)) |
| Peut-on l'utiliser sur 1 GPU ? | **Non** (smoke (b)), même pour de l'eval |
| Que faire ? | Provisionner ≥ 2 GPUs et utiliser verl tel quel |
| Combien de GPUs pour la recette papier exacte (N=8, bs=32, full FT, max_resp=10240) ? | **8× A100 40 Go** (~320 Go agrégés ≈ paper setup avec FSDP) |
| Plan B si 8× A100 indisponible ? | Garder la stack TRL+LoRA actuelle, mais cap matériel = baseline |

Décision : on **provisionne 8× A100 40 Go** et on lance directement la
recette `textcraft_train.sh` avec les overrides 40 Go (`gpu_memory_utilization=0.65`,
`max_response_length=8192`, sinon paramètres papier inchangés).

---

### Oracle pass@k — marge exploitable par le RL (2026-06-16)

Protocole : N=20 trajectoires/item (T=1.0, ≤30 tours), pass@k non biaisé (Chen et al. 2021),
global + par depth. pass@1 = fiable ; pass@20 = atteignable « avec oracle » (borne sup.
optimiste). Script `src/eval/eval_oracle.py`. **Analyse détaillée : `runs/7_oracle/oracle_best32/ANALYSIS.md`.**

**pass@k par depth — baseline Qwen2.5-3B vs best GRPO (checkpoint 32/100) :**
| depth | items | base @1→@20 | best @1→@20 |
|---|---|---|---|
| 1 | 31 | 27 → 94 | 79 → 100 |
| 2 | 41 | 4 → 44 | 18 → **78** |
| 3 | 25 | 0 → **0** | 1 → **8** |
| 4 | 3 | 0 → **0** | 0 → **0** |
| global | 100 | 10 → 47 | 32 → **65** |

**Conclusions :**
- **Depth 2 = mur de fiabilité, exploitable par RL** : le best sait résoudre 78% des items
  depth-2 « parfois » mais 18% de façon fiable (marge **+60 pts**). Priorité pour pousser le score.
- **Depth 3-4 = mur de capacité** : best plafonne à 8% (depth 3 : 4 succès/500 tirages, 2 items)
  et **0%** (depth 4 : 0/60). Baseline = **0 partout** à depth 3-4 (0/500, 0/60).
  → pass@k=0 ⇒ aucune graine pour GRPO (Dr. GRPO) ⇒ le RL pur **ne peut pas** bootstrap ces
  profondeurs. Il faut injecter des succès : **curriculum / SFT / CoT**.
- **Nuance** : le RL a quand même *déplacé la frontière* (depth-2 : +14 items atteignables ;
  depth-3 : +2 items, jamais résolus par la baseline) → depth 3 est à la limite, pas un mur infini.

