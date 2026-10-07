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



---

### Exp 18 — few-shot prompting (2026-07-16, éval pure, aucun training)

**Question** : combien de points le prompt seul peut-il apporter au Qwen2.5-3B base ?
k exemples de tâches résolues injectés comme de VRAIS tours de dialogue (format
ReAct), issus des 70 recettes de l'univers hors train/test (18 d2, 45 d3, 7 d4 —
zéro contamination), chaque solution validée par replay env (70/70 reward=1).
Ordre entrelacé d2,d3,d2,d3,d4, préfixes emboîtés. Serveur vLLM 32k. Détails et
artefacts : `runs/0_baselines/exp18_fewshot/`.

| k (exemples) | 0 | 1 | 3 | **5** | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| Pass@1 /100 | 18 | 25 | 30 | **31** | 30 | 30 | 20 | 22* |

*k=50 : 99 items (1 crash). Par depth (k=5) : d1 25/31, d2 5/41, d3 1/25, d4 0/3.

**Lectures** :
1. **+13 pts sans aucun entraînement** (18 → 31, +72 % relatif) — comparable aux
   premiers runs GRPO full-FT (exp7.x : 15-22 après des centaines de steps).
2. Le gain vient de la **fiabilité depth 1-2** (d1 : 13→25/31) ; **depth ≥ 3 ne
   bouge pas** (0-1 succès) → confirme le mur de capacité du diagnostic oracle,
   que le prompting ne contourne pas.
3. Plateau k=3-20 (~30), **dégradation k≥30** (8-14k tokens d'exemples : la
   dilution du contexte long l'emporte sur l'information ajoutée).
4. **Ablation format** (témoins `format_bloc_k01/k03` : 5 et 8/100) : des exemples
   collés en bloc texte font halluciner les paires Action/Observation au modèle
   (il écrit lui-même les réponses de l'env et se désynchronise). Les
   démonstrations d'agent doivent être des tours de dialogue.

**Suites possibles** : RL avec prompt few-shot k=5 (départ 31 au lieu de 18,
plus de trajectoires à reward non nul pour GRPO) ; combinaison avec CoT tour 1.

#### Exp 18b — pass@10 few-shot (oracle N=10, 2026-07-17) : le mur d3 se fissure

| Config | pass@1 | pass@10 | d1@10 | d2@10 | d3@10 | d4@10 |
|---|---|---|---|---|---|---|
| zero-shot (réf. N=20) | 10.3 % | 37.8 % | 84 % | 29 % | **0 %** | 0 % |
| few-shot k=5 | 27.7 % | 60.0 % | 100 % | 66 % | 8 % | 0 % |
| few-shot k=20 | 27.0 % | **61.0 %** | 100 % | 63 % | **16 %** | 0 % |

**Le résultat qui change le diagnostic** : à depth 3, le zero-shot n'échantillonnait
JAMAIS de solution (pass@20 = 0, « mur de capacité », aucune graine GRPO). Avec 20
exemples en prompt, 16 % des items d3 sont résolus au moins une fois sur 10 tirages
(~4 items /25). Le mur d3 était donc en partie un artefact du prompt — **il existe
désormais du signal exploitable par le RL à depth 3**. k=20 couvre mieux d3 que k=5
(16 vs 8 %) à pass@1 égal : les exemples élargissent l'exploration, pas la fiabilité.
d4 (3 items) reste à 0. Artefacts : `exp18_fewshot/oracle_base_k{05,20}/`.

⚠ Note du 2026-07-17 : le best58 (exp10.8 mergé) n'est PLUS reconstructible — le
maillon initial de la lignée LoRA (merge 35 % d'exp10.3) a été supprimé de models/
sans backup, et les _best ne contiennent que les adapters. L'étude few-shot × best58
demandée est bloquée en attendant un éventuel backup externe. Leçon : sauvegarder
chaque merge de best sur disque persistant. Détail : docs/hebdo/21juillet/.

**Suite privilégiée** : RL initialisé avec prompt few-shot k≈5-20 — départ pass@1
~28-31, graines d3 disponibles, et l'écart pass@1→pass@10 (28→60) est exactement la
marge de fiabilisation que GRPO sait convertir.

---

### Exp 19–20 — few-shot RL + paliers LR/beta (2026-07, GRPO LoRA stack v2)

GRPO LoRA depuis Qwen2.5-3B nu, few-shot k=10 (mêmes exemples que exp18, format
dialogue multi-tour), N=8, 64 traj/step, éval périodique multi-tour /100.
Stack v2 (TRL 1.9 + vLLM colocate + flash-attn). Artefacts : `runs/10_fewshot_rl/`.

| Run | LR / schedule | Statut | Best Pass@1 (éval périodique) |
|---|---|---|---|
| exp19.2 | 5e-6 **constant**, beta 0.01 | arrêté step ~557 (collapse KL) | **45/100 @ step 300** |
| exp20 | 5e-6 + **÷3 / 3 epochs** (LR et beta), 21 ep | terminé 966 steps, pas de collapse | **58/100 @ step 800** |

**Lecture exp20 vs exp19.2** : les **paliers LR calendaires** (÷3 toutes les
3 epochs via `StagedLrBetaCallback`) **stabilisent** l'entraînement et permettent
d'**atteindre un meilleur pic test** qu'un LR constant à 5e-6 sur la durée du run.

**Limite observée (reward train, 2026-07-27)** : en regardant le reward moyen
sur les rollouts d'entraînement, chaque division de LR semble **interrompre une
dynamique d'apprentissage en cours** — le schedule est piloté par le **calendrier**
(epochs), pas par l'état d'apprentissage. Le Pass@1 test continue néanmoins de
progresser par à-coups (pic 58 au step 800, LR déjà à ~7e-9) : le schedule n'est
pas catastrophique, mais le signal train suggère un coût en dynamique à chaque palier.

**Piste ouverte — LR adaptatif au reward train** (non implémentée) :
- diviser LR (et éventuellement beta) quand la **dérivée d'une moyenne roulante**
  du reward train reste ≤ 0 pendant plusieurs steps ;
- fenêtre proposée : **~¼ d'epoch** (~12 steps, soit ~768 trajectoires/step × 12)
  pour lisser le bruit du tirage aléatoire d'items tout en restant réactif ;
- implémentation envisagée : callback type `RewardAdaptiveLrCallback` (même
  mécanisme de modification de `base_lrs` / `trainer.beta` que `StagedLrBetaCallback`).
  Détail : `docs/hebdo/31juillet/session_2026-07-27_paliers_lr_et_reward_train.md`.

**Suite** : exp21 reprend les paliers calendaires (LR 7e-6, mode single-turn) comme
témoin ; ablation reward-adaptive vs paliers calendaires à planifier.


## exp31.1 — ancre FIXE + β0.0001 (ablation : l'ancre mobile est-elle nécessaire ?) — `exp31.1_r8_fixedanchor_b00001` (fin 2026-08-24 16:47)

**Best test : step=282 pass_at_1=0.3400**

```
[test_eval] step 47 — Pass@1 = 13/100 (13%) en 35.4s
[test_eval] step 94 — Pass@1 = 16/100 (16%) en 28.9s
[test_eval] step 141 — Pass@1 = 22/100 (22%) en 24.7s
[test_eval] step 188 — Pass@1 = 24/100 (24%) en 26.9s
[test_eval] step 235 — Pass@1 = 32/100 (32%) en 23.1s
[test_eval] step 282 — Pass@1 = 34/100 (34%) en 13.9s
[test_eval] step 329 — Pass@1 = 25/100 (25%) en 9.6s
[test_eval] step 376 — Pass@1 = 25/100 (25%) en 17.0s
[test_eval] step 423 — Pass@1 = 26/100 (26%) en 129.2s
```

_Interprétation (24/08) : arrêté volontairement à l'ep ~9/200 — divergence KL
runaway (0.2 à l'ep 6 → 906 → 1572 → 2×10¹⁴ à l'ep 9, entropie 1.3→0.02,
completions 8k tokens). Verdict de l'ablation : β quasi nul + ancre fixe ne
tient pas ; combiné à exp30 (fixe+β0.01, collapse ep ~11) et exp31 (mobile+β0.001,
53 stable), l'ancre MOBILE est nécessaire, aucune valeur de β ne sauve l'ancre
fixe. Best 34 + optimizer et ckpt-376 sauvés sur le home. Détail :
`runs/13_moving_anchor/exp31.1_r8_fixedanchor_b00001/config.yaml`._

## exp32 — curriculum horizon (ScalingInter, N=16) — `exp32_horizon` (fin 2026-08-26 17:17)

**Best test : step=7238 pass_at_1=0.8200**

```
[test_eval] step 5640 — Pass@1 = 80/100 (80%) en 10.2s
[test_eval] step 5687 — Pass@1 = 75/100 (75%) en 9.8s
[test_eval] step 5734 — Pass@1 = 76/100 (76%) en 10.2s
[test_eval] step 5781 — Pass@1 = 78/100 (78%) en 9.9s
[test_eval] step 5828 — Pass@1 = 78/100 (78%) en 9.8s
[test_eval] step 5875 — Pass@1 = 74/100 (74%) en 9.7s
[test_eval] step 5922 — Pass@1 = 75/100 (75%) en 10.0s
[test_eval] step 5969 — Pass@1 = 71/100 (71%) en 11.1s
[test_eval] step 6016 — Pass@1 = 78/100 (78%) en 9.9s
[test_eval] step 6063 — Pass@1 = 76/100 (76%) en 9.7s
[test_eval] step 6110 — Pass@1 = 82/100 (82%) en 9.5s
[test_eval] step 6157 — Pass@1 = 78/100 (78%) en 9.7s
[test_eval] step 6204 — Pass@1 = 77/100 (77%) en 9.5s
[test_eval] step 6251 — Pass@1 = 77/100 (77%) en 10.4s
[test_eval] step 6298 — Pass@1 = 79/100 (79%) en 9.8s
[test_eval] step 6345 — Pass@1 = 79/100 (79%) en 11.7s
[test_eval] step 6392 — Pass@1 = 78/100 (78%) en 10.3s
[test_eval] step 6439 — Pass@1 = 81/100 (81%) en 9.4s
[test_eval] step 6486 — Pass@1 = 77/100 (77%) en 9.9s
[test_eval] step 6533 — Pass@1 = 75/100 (75%) en 9.8s
[test_eval] step 6580 — Pass@1 = 77/100 (77%) en 10.7s
[test_eval] step 6627 — Pass@1 = 81/100 (81%) en 9.3s
[test_eval] step 6674 — Pass@1 = 75/100 (75%) en 10.2s
[test_eval] step 6721 — Pass@1 = 78/100 (78%) en 10.3s
[test_eval] step 6768 — Pass@1 = 77/100 (77%) en 10.9s
[test_eval] step 6815 — Pass@1 = 79/100 (79%) en 10.0s
[test_eval] step 6862 — Pass@1 = 74/100 (74%) en 11.2s
[test_eval] step 6909 — Pass@1 = 76/100 (76%) en 10.4s
[test_eval] step 6956 — Pass@1 = 77/100 (77%) en 9.6s
[test_eval] step 7003 — Pass@1 = 78/100 (78%) en 9.8s
[test_eval] step 7050 — Pass@1 = 76/100 (76%) en 9.4s
[test_eval] step 7097 — Pass@1 = 76/100 (76%) en 9.8s
[test_eval] step 7144 — Pass@1 = 78/100 (78%) en 9.7s
[test_eval] step 7191 — Pass@1 = 79/100 (79%) en 9.8s
[test_eval] step 7238 — Pass@1 = 82/100 (82%) en 9.2s
[test_eval] step 7285 — Pass@1 = 81/100 (81%) en 10.0s
[test_eval] step 7332 — Pass@1 = 75/100 (75%) en 9.7s
[test_eval] step 7379 — Pass@1 = 78/100 (78%) en 9.6s
[test_eval] step 7426 — Pass@1 = 79/100 (79%) en 9.8s
[test_eval] step 7473 — Pass@1 = 80/100 (80%) en 9.7s
```

_Interprétation (26/08) : **RECORD DU PROJET — 82/100 (steps 6110 et 7238), objectif
papier (75) DÉPASSÉ.** Arrêt volontaire au plateau (ep ~80/200 — N=16 : 93 steps/epoch, dernières évals
75-82, 2 j 00 h GPU, ~23 s/step de moyenne grâce aux épisodes courts du début).
Recette exp25 + horizon progressif 10/20/30 (ep 0/15/30) + N=16 (bench exp25.1),
20 ré-ancrages sans collapse. +9 pts sur le précédent record (73, exp25.2).
⚠ Double delta vs exp25 (horizon ET N) : exp36_n16 (lancée dans la foulée)
isole l'effet de N. Analyse par depth à faire (phase 3). Best + optimizer :
`saves/trl_grpo/exp32_horizon_best` ; dernier ckpt : `..._ckpt7426` ; chaîne
d'ancres : `..._anchors` (20 cycles)._

## exp36 — contrôle N=16 sans curriculum — `exp36_n16` (fin 2026-08-27 09:00)

**Best test : step=1222 pass_at_1=0.5400**

```
[test_eval] step 47 — Pass@1 = 12/100 (12%) en 30.6s
[test_eval] step 94 — Pass@1 = 18/100 (18%) en 30.1s
[test_eval] step 141 — Pass@1 = 18/100 (18%) en 28.2s
[test_eval] step 188 — Pass@1 = 28/100 (28%) en 31.5s
[test_eval] step 235 — Pass@1 = 31/100 (31%) en 23.7s
[test_eval] step 282 — Pass@1 = 35/100 (35%) en 21.2s
[test_eval] step 329 — Pass@1 = 37/100 (37%) en 16.4s
[test_eval] step 376 — Pass@1 = 38/100 (38%) en 13.7s
[test_eval] step 423 — Pass@1 = 32/100 (32%) en 14.9s
[test_eval] step 470 — Pass@1 = 41/100 (41%) en 14.1s
[test_eval] step 517 — Pass@1 = 42/100 (42%) en 13.8s
[test_eval] step 564 — Pass@1 = 41/100 (41%) en 14.1s
[test_eval] step 611 — Pass@1 = 41/100 (41%) en 14.8s
[test_eval] step 658 — Pass@1 = 45/100 (45%) en 16.1s
[test_eval] step 705 — Pass@1 = 42/100 (42%) en 14.5s
[test_eval] step 752 — Pass@1 = 33/100 (33%) en 15.2s
[test_eval] step 799 — Pass@1 = 36/100 (36%) en 15.2s
[test_eval] step 846 — Pass@1 = 40/100 (40%) en 13.6s
[test_eval] step 893 — Pass@1 = 44/100 (44%) en 18.6s
[test_eval] step 940 — Pass@1 = 49/100 (49%) en 12.1s
[test_eval] step 987 — Pass@1 = 48/100 (48%) en 8.2s
[test_eval] step 1034 — Pass@1 = 42/100 (42%) en 9.4s
[test_eval] step 1081 — Pass@1 = 38/100 (38%) en 8.2s
[test_eval] step 1128 — Pass@1 = 38/100 (38%) en 8.5s
[test_eval] step 1175 — Pass@1 = 45/100 (45%) en 9.3s
[test_eval] step 1222 — Pass@1 = 54/100 (54%) en 15.2s
[test_eval] step 1269 — Pass@1 = 47/100 (47%) en 28.5s
[test_eval] step 1316 — Pass@1 = 30/100 (30%) en 48.5s
[test_eval] step 1363 — Pass@1 = 14/100 (14%) en 53.0s
[test_eval] step 1410 — Pass@1 = 13/100 (13%) en 122.9s
```

_Interprétation (27/08) : **COLLAPSE instructif — le 1er verdict du contrôle N=16.**
Montée SAINE et rapide 12→54 (step 1222, ep ~13 ; plus vite qu'exp25 N=8 à
trajectoires égales), puis effondrement d'entropie ep 8-12 (1.0→0.10, seuil
starvation 0.15), KL runaway (10⁸-10¹²) malgré les ré-ancrages (ep 4/8/12 : l'ancre
SUIT la politique en déterminisation au lieu de la freiner), éval 54→13, générations
détruites. Lecture : N=16 accélère le sharpening ~2× ; sans curriculum pour
ré-injecter de la difficulté (rôle que joue l'horizon 10/20/30 d'exp32, entropie
stable 0.6-0.7), la recette commune est instable → le curriculum d'exp32 agit en
RÉGULARISEUR d'exploration, pas seulement en accélérateur. N=16 seul ≠ le 82
d'exp32. Best 54 + optimizer sauvés ; ckpt-1410 (post-collapse, analyse seule).
Suite : exp36.1 (ancre /12 ep, job 48) en fin de file. Détail : session hebdo 27/08._

**Note méthodologique — pourquoi on compare à hyperparamètres égaux (27/08).**
Toute config partagée amplifiera toujours un régime par rapport à l'autre ; la
seule alternative rigoureuse serait un budget de tuning égal PAR BRAS (chercher
les meilleurs hyperparamètres de chaque régime séparément, puis comparer les
optima) — c'est un autre claim, bien plus cher, hors de portée avant le gel du
2/09, et la littérature curriculum vit très bien avec des comparaisons à recette
partagée. Le claim du rapport est donc formulé conditionnellement : **« à recette
commune (celle stabilisée en phase 1), le curriculum survit et atteint 82 là où
le sans-curriculum collapse à 54 »** — un résultat sur la robustesse et
l'interaction curriculum × recette, PAS une preuve que N=16 sans curriculum ne
peut jamais atteindre 82. Dit comme ça, exp36 n'affaiblit rien : il RENFORCE
exp32 (le curriculum rend la recette robuste dans une zone où elle est sinon
instable). Subtilité rendue visible par la correction d'échelle du 26/08 :
« 4 epochs » n'est de toute façon pas le même hyperparamètre en temps-updates
selon N (ré-ancrage tous les 184 steps à N=8, 372 à N=16) — l'égalité stricte
des hyperparamètres est déjà une convention. Corollaire pour exp36.1 : à
comparaison honnête, on travaille à **β fixe** et on règle le curseur d'un
mécanisme déjà en place (la période d'ancre, 4→12 ep) plutôt que d'ajouter une
force nouvelle (β ou entropy-coef plus forts = un delta de plus, moins
interprétable) ; trade-off connu : période trop longue = retour à la pathologie
de l'ancre fixe (plafond puis décrochage, exp24/exp30). Les ablations d'ancre
(4 vs 8 vs 12) restent notées comme travail ultérieur.

## exp33.2 — reprise depth-auto depuis best 72 (palier final) — `exp33.2_from72` (fin 2026-08-31 11:23)

**Best test : step=3290 pass_at_1=0.8000**

```
[test_eval] step 1880 — Pass@1 = 73/100 (73%) en 11.5s
[test_eval] step 1927 — Pass@1 = 70/100 (70%) en 11.1s
[test_eval] step 1974 — Pass@1 = 66/100 (66%) en 11.4s
[test_eval] step 2021 — Pass@1 = 68/100 (68%) en 11.2s
[test_eval] step 2068 — Pass@1 = 71/100 (71%) en 10.6s
[test_eval] step 2115 — Pass@1 = 73/100 (73%) en 12.5s
[test_eval] step 2162 — Pass@1 = 70/100 (70%) en 11.6s
[test_eval] step 2209 — Pass@1 = 73/100 (73%) en 9.8s
[test_eval] step 2256 — Pass@1 = 73/100 (73%) en 10.7s
[test_eval] step 2303 — Pass@1 = 72/100 (72%) en 10.7s
[test_eval] step 2350 — Pass@1 = 75/100 (75%) en 11.0s
[test_eval] step 2397 — Pass@1 = 76/100 (76%) en 9.6s
[test_eval] step 2444 — Pass@1 = 74/100 (74%) en 11.6s
[test_eval] step 2491 — Pass@1 = 72/100 (72%) en 11.5s
[test_eval] step 2538 — Pass@1 = 74/100 (74%) en 11.6s
[test_eval] step 2585 — Pass@1 = 73/100 (73%) en 11.4s
[test_eval] step 2632 — Pass@1 = 73/100 (73%) en 11.4s
[test_eval] step 2679 — Pass@1 = 74/100 (74%) en 11.5s
[test_eval] step 2726 — Pass@1 = 71/100 (71%) en 10.6s
[test_eval] step 2773 — Pass@1 = 77/100 (77%) en 9.8s
[test_eval] step 2820 — Pass@1 = 76/100 (76%) en 10.4s
[test_eval] step 2867 — Pass@1 = 77/100 (77%) en 9.5s
[test_eval] step 2914 — Pass@1 = 76/100 (76%) en 9.9s
[test_eval] step 2961 — Pass@1 = 70/100 (70%) en 10.9s
[test_eval] step 3008 — Pass@1 = 75/100 (75%) en 10.7s
[test_eval] step 3055 — Pass@1 = 72/100 (72%) en 13.0s
[test_eval] step 3102 — Pass@1 = 79/100 (79%) en 10.9s
[test_eval] step 3149 — Pass@1 = 75/100 (75%) en 11.8s
[test_eval] step 3196 — Pass@1 = 68/100 (68%) en 13.5s
[test_eval] step 3243 — Pass@1 = 72/100 (72%) en 11.3s
[test_eval] step 3290 — Pass@1 = 80/100 (80%) en 11.2s
[test_eval] step 3337 — Pass@1 = 72/100 (72%) en 12.1s
[test_eval] step 3384 — Pass@1 = 71/100 (71%) en 11.0s
[test_eval] step 3431 — Pass@1 = 77/100 (77%) en 12.3s
[test_eval] step 3478 — Pass@1 = 73/100 (73%) en 10.0s
[test_eval] step 3525 — Pass@1 = 72/100 (72%) en 12.0s
[test_eval] step 3572 — Pass@1 = 74/100 (74%) en 11.5s
[test_eval] step 3619 — Pass@1 = 75/100 (75%) en 10.9s
[test_eval] step 3666 — Pass@1 = 75/100 (75%) en 10.7s
[test_eval] step 3713 — Pass@1 = 75/100 (75%) en 11.5s
```

_Interprétation (03/09) : collapse sur la même échelle qu'exp36 (G16 sans
curriculum), un peu plus tôt (ep 7-8 contre 10-12). Entropie 0.94 (ep 2) → 0.66
(ep 6-7) → 0.19 (ep 7-8) → 0.01 ; KL 0.019 → 133 → 10¹²⁺ : entropie d'abord, KL
ensuite. Le sampler a fait ce que MAGELLAN prescrit : p(d4) = 0.0006 (prédiction
« désinvestit d4 » confirmée), mais p(d1) = 0.79 en fin de run parce que le LP
(|SR − SR retardé|) mesure du CHANGEMENT et qu'un collapse est un changement
énorme : les prédictions sur d1 ont oscillé, le sampler s'est concentré sur d1,
la diversité des tâches a chuté, le collapse s'est accéléré. Rétroaction
positive entre le signal de progrès et l'instabilité de l'apprenant : MAGELLAN
suppose une politique stable, il ne fournit pas l'entropie manquante. Aucun
gain à attendre d'un rejeu à cette recette. Tuée 08:47, la file a enchaîné sur
exp36.1._

## exp35.1 — reprise exp35 à l'ancre 16, 16 ep finales (budget 1024) — `exp35.1_from_anchor16` (fin 2026-09-02 16:04)

**Best test : step=470 pass_at_1=0.8000**

```
[test_eval] step 47 — Pass@1 = 74/100 (74%) en 6.9s
[test_eval] step 94 — Pass@1 = 71/100 (71%) en 6.9s
[test_eval] step 141 — Pass@1 = 73/100 (73%) en 6.6s
[test_eval] step 188 — Pass@1 = 75/100 (75%) en 6.6s
[test_eval] step 235 — Pass@1 = 79/100 (79%) en 6.7s
[test_eval] step 282 — Pass@1 = 75/100 (75%) en 6.7s
[test_eval] step 329 — Pass@1 = 75/100 (75%) en 8.3s
[test_eval] step 376 — Pass@1 = 74/100 (74%) en 8.4s
[test_eval] step 423 — Pass@1 = 75/100 (75%) en 8.8s
[test_eval] step 470 — Pass@1 = 80/100 (80%) en 7.9s
[test_eval] step 517 — Pass@1 = 77/100 (77%) en 7.6s
[test_eval] step 564 — Pass@1 = 76/100 (76%) en 8.0s
[test_eval] step 611 — Pass@1 = 69/100 (69%) en 7.7s
[test_eval] step 658 — Pass@1 = 77/100 (77%) en 8.4s
[test_eval] step 705 — Pass@1 = 74/100 (74%) en 7.1s
[test_eval] step 752 — Pass@1 = 77/100 (77%) en 7.0s
[test_eval] step 799 — Pass@1 = 77/100 (77%) en 7.0s
[test_eval] step 846 — Pass@1 = 74/100 (74%) en 7.4s
[test_eval] step 893 — Pass@1 = 74/100 (74%) en 7.3s
[test_eval] step 940 — Pass@1 = 76/100 (76%) en 7.6s
[test_eval] step 987 — Pass@1 = 79/100 (79%) en 7.1s
[test_eval] step 1034 — Pass@1 = 76/100 (76%) en 7.3s
[test_eval] step 1081 — Pass@1 = 71/100 (71%) en 7.5s
[test_eval] step 1128 — Pass@1 = 74/100 (74%) en 7.5s
[test_eval] step 1175 — Pass@1 = 72/100 (72%) en 7.8s
```

_Interprétation : à compléter à la relecture._

## exp34 — autocurriculum MAGELLAN (ALP appris) — `exp34_magellan` (fin 2026-09-03 08:47)

**Best test : step=611 pass_at_1=0.4500**

```
[test_eval] step 47 — Pass@1 = 15/100 (15%) en 29.3s
[test_eval] step 94 — Pass@1 = 22/100 (22%) en 28.0s
[test_eval] step 141 — Pass@1 = 27/100 (27%) en 25.9s
[test_eval] step 188 — Pass@1 = 19/100 (19%) en 23.3s
[test_eval] step 235 — Pass@1 = 32/100 (32%) en 22.4s
[test_eval] step 282 — Pass@1 = 41/100 (41%) en 16.0s
[test_eval] step 329 — Pass@1 = 38/100 (38%) en 13.1s
[test_eval] step 376 — Pass@1 = 36/100 (36%) en 12.1s
[test_eval] step 423 — Pass@1 = 37/100 (37%) en 12.4s
[test_eval] step 470 — Pass@1 = 35/100 (35%) en 13.1s
[test_eval] step 517 — Pass@1 = 43/100 (43%) en 12.1s
[test_eval] step 564 — Pass@1 = 40/100 (40%) en 13.4s
[test_eval] step 611 — Pass@1 = 45/100 (45%) en 13.0s
[test_eval] step 658 — Pass@1 = 44/100 (44%) en 13.3s
[test_eval] step 705 — Pass@1 = 32/100 (32%) en 10.9s
[test_eval] step 752 — Pass@1 = 18/100 (18%) en 15.6s
[test_eval] step 799 — Pass@1 = 19/100 (19%) en 23.3s
[test_eval] step 846 — Pass@1 = 21/100 (21%) en 17.6s
[test_eval] step 893 — Pass@1 = 23/100 (23%) en 44.0s
[test_eval] step 940 — Pass@1 = 0/100 (0%) en 75.9s
[test_eval] step 987 — Pass@1 = 2/100 (2%) en 99.7s
[test_eval] step 1034 — Pass@1 = 2/100 (2%) en 211.9s
```

_Interprétation : à compléter à la relecture._


## Analyse transverse KL — ce qui a mené aux choix (ancre mobile, β, rang) — 03/09/2026

Source : logs TRL de chaque run, clé `kl` (estimateur k3 par token, moyenné) et
`entropy`, agrégés par fenêtre de 2 epochs (médiane et max de la fenêtre).
Script inline, reproductible depuis `logs/<run>.log`.

| Run | ancre | β | rang | KL méd. ep 0-2 | ep 4-6 | ep 6-8 | ep 8-10 | entropie à ep 4-6 | issue |
|---|---|---|---|---|---|---|---|---|---|
| exp23 (full-FT) | fixe | 0.001 | — | 0.0011 | 0.0037 | 0.0056 | 0.0078 | 1.13 | +0.001/ep linéaire, 0.026 à ep 30, stable, 69 (lignée) |
| exp24 r16 | fixe | 0.001 | 16 | 0.0010 | 0.040 | 0.96 | 17.7 | 1.39 | collapse ep 8-10, pic 35 |
| exp24 r16 | fixe | 0.01 | 16 | 0.0010 | 0.0065 | 0.10 | 8.0 | 1.14 | collapse ep 8-10, pic 39 |
| exp24 r64 | fixe | 0.001 | 64 | 0.0010 | 0.28 | 1.85 | 3.7 | 1.26 | collapse ep 6-8, pic 25 |
| exp24 r64 | fixe | 0.01 | 64 | 0.0011 | 0.18 | 1.6e8 | — | 1.43 | collapse ep 6-8, plafond 21-29 |
| exp30 r8 | fixe | 0.01 | 8 | 0.0010 | 0.010 | 0.059 | 0.26 | 1.19 | 3.4 à ep 10-12, collapse, pic 39 |
| exp31.1 r8 | fixe | 0.0001 | 8 | 0.0010 | 0.047 | 478 | 3.7e12 | 1.42 | divergence ep 6-8, pic 34 |
| exp25 r8 | mobile /4 | 0.01 | 8 | 0.0010 | 0.0010 | 0.0014 | 0.0009 | 1.10 | 0.0010 ± 0.0002 pendant 111 ep ; 65 → 73 |
| exp31 r8 | mobile /4 | 0.001 | 8 | 0.0010 | 0.0010 | 0.0014 | 0.0009 | 1.18 | 0.0010 pendant 60 ep ; 53 |
| exp32 r8 (curriculum) | mobile /4 | 0.01 | 8 | 0.0011 (médiane des 80 ep) | | | | | 82 |

Lectures :

1. **Échelle naturelle.** Tous les runs démarrent à KL ≈ 0.001 : c'est la dérive
   d'un pas de LR 3e-6, quel que soit le régime. Une KL « saine » chez nous, c'est
   10⁻³ ; les runs stables y restent 80 à 110 epochs.
2. **Full-FT à ancre fixe : dérive linéaire.** +0.001 par epoch, 0.026 à l'epoch
   30, entropie constante ~1.2 : la politique s'éloigne lentement et reste
   « attachée ». C'est pourquoi la lignée exp23 n'a jamais eu besoin d'ancre mobile.
3. **LoRA à ancre fixe : dérive géométrique.** ×5 à ×10 par fenêtre de 2 epochs à
   partir de l'epoch 4 (r16 β0.001 : 0.0024 → 0.040 → 0.96 → 17.7). β retarde d'une
   fenêtre (r16 : 0.10 à ep 6-8 sous β0.01 contre 0.96 sous β0.001) ; le rang
   accélère (ep 4-6 : r8 0.010, r16 0.040, r64 0.28). L'entropie est encore HAUTE
   (1.2-1.4, au-dessus du départ) quand la KL franchit 0.05 : la panne à ancre fixe
   est un emballement de KL, pas un collapse d'entropie ; l'entropie s'effondre
   ensuite (générations détruites).
4. **Point de non-retour ≈ 0.05-0.1 de KL médiane.** Chaque run dont la médiane a
   franchi 0.05 sur une fenêtre était > 1 la fenêtre suivante et mort celle
   d'après. Règle de surveillance : KL médiane > 0.02 soutenue = alerte.
5. **Pourquoi β ne peut pas sauver l'ancre fixe (ordre de grandeur).** Le terme de
   politique vaut |L_pg| ≈ 2.9e-3 (médiane exp32, p90 6e-3). À KL = 10⁻³, le terme
   β·KL vaut 10⁻⁵ pour β=0.01 : 0,3 % du terme de politique, inerte par
   construction en régime sain. Il devient comparable au terme de politique quand
   KL ≈ 0.3 (β=0.01), ≈ 3 (β=0.001), ≈ 30 (β=0.0001) : pour tous les β testés, le
   frein n'engage qu'APRÈS le point de non-retour. Monter β à 0.1 engagerait à
   KL ≈ 0.03 mais β=0.01 plafonne déjà (r64 : 21-29). Comparaison au niveau des
   pertes, pas des gradients : argument d'ordre de grandeur, à présenter comme tel.
6. **Ancre mobile : la dérive disparaît par construction.** Référence = politique
   d'il y a 4 epochs → KL épinglée à 0.0010 ± 0.0002 pendant 111 epochs (exp25),
   60 (exp31), 80 (exp32), pour β = 0.001 comme pour β = 0.01. Les transitoires
   (5.4 à ep 50-52, 0.12 à ep 108-110 chez exp25 ; 137 à ep 2-4 chez exp32) sont
   absorbés dans la fenêtre. C'est la position de l'ancre qui fixe le régime, pas β.
7. **β=0.01 vs β=0.001 sous ancre mobile : indiscernables à epoch égale.** Test
   aux epochs 49-58 : exp25 (β0.01) 45-52, exp31 (β0.001) 46-53. Le 65-73 d'exp25
   vient de 110 + 60 epochs supplémentaires ; exp31 a été arrêtée à l'ep 60. Seule
   différence mesurée : β0.001 garde plus d'entropie (1.20 vs 1.05 à l'ep 60).
   β=0.01 en phase 2 est donc un choix de continuité de lignée, pas une supériorité
   mesurée. **Ne pas écrire « β=0.01 reste meilleur » dans le rapport.**
8. **Les collapses G16 (exp36, exp34) sont d'une autre nature** : entropie d'abord
   (1.0 → 0.59 → 0.10) à KL stable ~0.003, emballement KL ensuite. L'ancre mobile
   règle l'emballement de KL, pas l'extinction d'entropie ; le curriculum (exp32)
   est ce qui a maintenu l'entropie (0.83 → 0.22 en 80 ep, sans collapse).

## exp39 — G=16 à 8 tâches × 16 traj par pas (déconfondre G et tâches/pas), sans curriculum — `exp39_g16_8tasks` (fin 2026-09-04 14:44)

**Best test : step=94 pass_at_1=0.1800**

```
[test_eval] step 47 — Pass@1 = 13/100 (13%) en 27.6s
[test_eval] step 94 — Pass@1 = 18/100 (18%) en 25.3s
```

_Interprétation : à compléter à la relecture._

## exp39 — G=16 à 8 tâches × 16 traj/pas, sans curriculum (reprise ckpt-94 le 06/09) — `exp39_g16_8tasks` (fin 2026-09-09 02:15)

**Best test : step=3525 pass_at_1=0.7200**

```
[test_eval] step 1833 — Pass@1 = 49/100 (49%) en 17.3s
[test_eval] step 1880 — Pass@1 = 54/100 (54%) en 16.3s
[test_eval] step 1927 — Pass@1 = 57/100 (57%) en 14.9s
[test_eval] step 1974 — Pass@1 = 50/100 (50%) en 15.6s
[test_eval] step 2021 — Pass@1 = 57/100 (57%) en 15.0s
[test_eval] step 2068 — Pass@1 = 59/100 (59%) en 13.6s
[test_eval] step 2115 — Pass@1 = 56/100 (56%) en 15.8s
[test_eval] step 2162 — Pass@1 = 59/100 (59%) en 14.6s
[test_eval] step 2209 — Pass@1 = 58/100 (58%) en 14.7s
[test_eval] step 2256 — Pass@1 = 55/100 (55%) en 14.6s
[test_eval] step 2303 — Pass@1 = 58/100 (58%) en 14.7s
[test_eval] step 2350 — Pass@1 = 59/100 (59%) en 13.6s
[test_eval] step 2397 — Pass@1 = 62/100 (62%) en 13.2s
[test_eval] step 2444 — Pass@1 = 56/100 (56%) en 15.5s
[test_eval] step 2491 — Pass@1 = 64/100 (64%) en 13.8s
[test_eval] step 2538 — Pass@1 = 61/100 (61%) en 14.5s
[test_eval] step 2585 — Pass@1 = 64/100 (64%) en 12.8s
[test_eval] step 2632 — Pass@1 = 60/100 (60%) en 14.0s
[test_eval] step 2679 — Pass@1 = 60/100 (60%) en 13.9s
[test_eval] step 2726 — Pass@1 = 61/100 (61%) en 13.6s
[test_eval] step 2773 — Pass@1 = 60/100 (60%) en 14.3s
[test_eval] step 2820 — Pass@1 = 66/100 (66%) en 13.1s
[test_eval] step 2867 — Pass@1 = 62/100 (62%) en 12.6s
[test_eval] step 2914 — Pass@1 = 61/100 (61%) en 12.3s
[test_eval] step 2961 — Pass@1 = 62/100 (62%) en 13.6s
[test_eval] step 3008 — Pass@1 = 62/100 (62%) en 12.1s
[test_eval] step 3055 — Pass@1 = 64/100 (64%) en 12.1s
[test_eval] step 3102 — Pass@1 = 64/100 (64%) en 11.6s
[test_eval] step 3149 — Pass@1 = 69/100 (69%) en 10.9s
[test_eval] step 3196 — Pass@1 = 70/100 (70%) en 11.4s
[test_eval] step 3243 — Pass@1 = 66/100 (66%) en 12.3s
[test_eval] step 3290 — Pass@1 = 62/100 (62%) en 12.6s
[test_eval] step 3337 — Pass@1 = 65/100 (65%) en 11.8s
[test_eval] step 3384 — Pass@1 = 65/100 (65%) en 11.8s
[test_eval] step 3431 — Pass@1 = 66/100 (66%) en 11.4s
[test_eval] step 3478 — Pass@1 = 61/100 (61%) en 12.4s
[test_eval] step 3525 — Pass@1 = 72/100 (72%) en 11.7s
[test_eval] step 3572 — Pass@1 = 64/100 (64%) en 11.1s
[test_eval] step 3619 — Pass@1 = 68/100 (68%) en 12.2s
[test_eval] step 3666 — Pass@1 = 66/100 (66%) en 10.8s
```

_Interprétation : à compléter à la relecture._

## exp40 — G=16 × 8 tâches/pas, ancre /8 ép. (374 pas comme exp36), sans curriculum : contrôle propre du confondeur — `exp40_g16_8tasks_anchor8` (fin 2026-09-10 08:22)

**Best test : step=705 pass_at_1=0.4900**

```
[test_eval] step 47 — Pass@1 = 16/100 (16%) en 30.8s
[test_eval] step 94 — Pass@1 = 21/100 (21%) en 22.5s
[test_eval] step 141 — Pass@1 = 24/100 (24%) en 28.8s
[test_eval] step 188 — Pass@1 = 37/100 (37%) en 18.3s
[test_eval] step 235 — Pass@1 = 35/100 (35%) en 16.6s
[test_eval] step 282 — Pass@1 = 34/100 (34%) en 13.7s
[test_eval] step 329 — Pass@1 = 33/100 (33%) en 13.5s
[test_eval] step 376 — Pass@1 = 33/100 (33%) en 14.8s
[test_eval] step 423 — Pass@1 = 35/100 (35%) en 16.2s
[test_eval] step 470 — Pass@1 = 38/100 (38%) en 11.6s
[test_eval] step 517 — Pass@1 = 38/100 (38%) en 17.1s
[test_eval] step 564 — Pass@1 = 36/100 (36%) en 15.2s
[test_eval] step 611 — Pass@1 = 41/100 (41%) en 17.5s
[test_eval] step 658 — Pass@1 = 38/100 (38%) en 15.1s
[test_eval] step 705 — Pass@1 = 49/100 (49%) en 12.4s
[test_eval] step 752 — Pass@1 = 29/100 (29%) en 69.4s
[test_eval] step 799 — Pass@1 = 32/100 (32%) en 75.8s
[test_eval] step 846 — Pass@1 = 27/100 (27%) en 111.3s
[test_eval] step 893 — Pass@1 = 18/100 (18%) en 149.5s
```

_Interprétation : à compléter à la relecture._

## exp43 — Contrôle-G16 + borne k3 (reprise ckpt-5358 le 22/09, ép. 57,6 → 80) — `exp43_g16_klclamp10` (fin 2026-09-23 02:36)

**Best test : step=6768 pass_at_1=0.8200**

```
[test_eval] step 5593 — Pass@1 = 71/100 (71%) en 8.3s
[test_eval] step 5640 — Pass@1 = 74/100 (74%) en 7.4s
[test_eval] step 5687 — Pass@1 = 74/100 (74%) en 8.2s
[test_eval] step 5734 — Pass@1 = 73/100 (73%) en 8.0s
[test_eval] step 5781 — Pass@1 = 68/100 (68%) en 8.2s
[test_eval] step 5828 — Pass@1 = 75/100 (75%) en 8.1s
[test_eval] step 5875 — Pass@1 = 77/100 (77%) en 8.1s
[test_eval] step 5922 — Pass@1 = 73/100 (73%) en 8.3s
[test_eval] step 5969 — Pass@1 = 77/100 (77%) en 8.1s
[test_eval] step 6016 — Pass@1 = 71/100 (71%) en 8.3s
[test_eval] step 6063 — Pass@1 = 74/100 (74%) en 8.2s
[test_eval] step 6110 — Pass@1 = 74/100 (74%) en 8.6s
[test_eval] step 6157 — Pass@1 = 75/100 (75%) en 8.6s
[test_eval] step 6204 — Pass@1 = 70/100 (70%) en 8.8s
[test_eval] step 6251 — Pass@1 = 76/100 (76%) en 8.0s
[test_eval] step 6298 — Pass@1 = 75/100 (75%) en 8.3s
[test_eval] step 6345 — Pass@1 = 73/100 (73%) en 8.4s
[test_eval] step 6392 — Pass@1 = 76/100 (76%) en 8.2s
[test_eval] step 6439 — Pass@1 = 73/100 (73%) en 8.7s
[test_eval] step 6486 — Pass@1 = 75/100 (75%) en 8.8s
[test_eval] step 6533 — Pass@1 = 76/100 (76%) en 7.9s
[test_eval] step 6580 — Pass@1 = 76/100 (76%) en 8.2s
[test_eval] step 6627 — Pass@1 = 77/100 (77%) en 7.9s
[test_eval] step 6674 — Pass@1 = 78/100 (78%) en 8.3s
[test_eval] step 6721 — Pass@1 = 76/100 (76%) en 8.3s
[test_eval] step 6768 — Pass@1 = 82/100 (82%) en 7.9s
[test_eval] step 6815 — Pass@1 = 70/100 (70%) en 8.6s
[test_eval] step 6862 — Pass@1 = 75/100 (75%) en 8.4s
[test_eval] step 6909 — Pass@1 = 76/100 (76%) en 8.3s
[test_eval] step 6956 — Pass@1 = 74/100 (74%) en 8.6s
[test_eval] step 7003 — Pass@1 = 70/100 (70%) en 8.7s
[test_eval] step 7050 — Pass@1 = 76/100 (76%) en 8.4s
[test_eval] step 7097 — Pass@1 = 77/100 (77%) en 8.0s
[test_eval] step 7144 — Pass@1 = 78/100 (78%) en 8.4s
[test_eval] step 7191 — Pass@1 = 70/100 (70%) en 9.0s
[test_eval] step 7238 — Pass@1 = 78/100 (78%) en 8.1s
[test_eval] step 7285 — Pass@1 = 77/100 (77%) en 8.3s
[test_eval] step 7332 — Pass@1 = 76/100 (76%) en 8.8s
[test_eval] step 7379 — Pass@1 = 77/100 (77%) en 8.5s
[test_eval] step 7426 — Pass@1 = 78/100 (78%) en 8.2s
```

_Interprétation : à compléter à la relecture._

## exp45 — référence fixe + borne k3 (exp30 + un seul delta) : la borne remplace-t-elle l'ancre ? — `exp45_fixedanchor_klclamp10` (fin 2026-09-23 08:51)

**Best test : step=329 pass_at_1=0.3700**

```
[test_eval] step 47 — Pass@1 = 16/100 (16%) en 27.5s
[test_eval] step 94 — Pass@1 = 14/100 (14%) en 25.9s
[test_eval] step 141 — Pass@1 = 20/100 (20%) en 25.0s
[test_eval] step 188 — Pass@1 = 24/100 (24%) en 22.9s
[test_eval] step 235 — Pass@1 = 28/100 (28%) en 21.7s
[test_eval] step 282 — Pass@1 = 37/100 (37%) en 19.6s
[test_eval] step 329 — Pass@1 = 37/100 (37%) en 16.8s
[test_eval] step 376 — Pass@1 = 32/100 (32%) en 13.9s
[test_eval] step 423 — Pass@1 = 28/100 (28%) en 11.3s
[test_eval] step 470 — Pass@1 = 16/100 (16%) en 12.2s
[test_eval] step 517 — Pass@1 = 22/100 (22%) en 9.2s
[test_eval] step 564 — Pass@1 = 24/100 (24%) en 9.0s
[test_eval] step 611 — Pass@1 = 0/100 (0%) en 10.5s
[test_eval] step 658 — Pass@1 = 0/100 (0%) en 12.8s
```

_Interprétation : à compléter à la relecture._

## exp44 — curriculum Horizon 10/20/30 à G=8 (exp25 + un seul delta) : effet propre du curriculum — `exp44_horizon_g8` (fin 2026-09-24 14:03)

**Best test : step=3666 pass_at_1=0.6500**

```
[test_eval] step 1833 — Pass@1 = 51/100 (51%) en 17.3s
[test_eval] step 1880 — Pass@1 = 47/100 (47%) en 16.4s
[test_eval] step 1927 — Pass@1 = 49/100 (49%) en 18.3s
[test_eval] step 1974 — Pass@1 = 48/100 (48%) en 17.5s
[test_eval] step 2021 — Pass@1 = 51/100 (51%) en 16.9s
[test_eval] step 2068 — Pass@1 = 49/100 (49%) en 17.2s
[test_eval] step 2115 — Pass@1 = 49/100 (49%) en 17.4s
[test_eval] step 2162 — Pass@1 = 47/100 (47%) en 16.7s
[test_eval] step 2209 — Pass@1 = 50/100 (50%) en 17.3s
[test_eval] step 2256 — Pass@1 = 52/100 (52%) en 17.4s
[test_eval] step 2303 — Pass@1 = 50/100 (50%) en 16.5s
[test_eval] step 2350 — Pass@1 = 46/100 (46%) en 17.4s
[test_eval] step 2397 — Pass@1 = 54/100 (54%) en 17.2s
[test_eval] step 2444 — Pass@1 = 54/100 (54%) en 15.3s
[test_eval] step 2491 — Pass@1 = 50/100 (50%) en 16.8s
[test_eval] step 2538 — Pass@1 = 51/100 (51%) en 16.5s
[test_eval] step 2585 — Pass@1 = 48/100 (48%) en 16.3s
[test_eval] step 2632 — Pass@1 = 53/100 (53%) en 15.3s
[test_eval] step 2679 — Pass@1 = 52/100 (52%) en 18.9s
[test_eval] step 2726 — Pass@1 = 56/100 (56%) en 17.3s
[test_eval] step 2773 — Pass@1 = 51/100 (51%) en 16.9s
[test_eval] step 2820 — Pass@1 = 54/100 (54%) en 14.9s
[test_eval] step 2867 — Pass@1 = 53/100 (53%) en 14.5s
[test_eval] step 2914 — Pass@1 = 54/100 (54%) en 16.7s
[test_eval] step 2961 — Pass@1 = 53/100 (53%) en 15.6s
[test_eval] step 3008 — Pass@1 = 54/100 (54%) en 15.4s
[test_eval] step 3055 — Pass@1 = 57/100 (57%) en 14.2s
[test_eval] step 3102 — Pass@1 = 56/100 (56%) en 15.4s
[test_eval] step 3149 — Pass@1 = 56/100 (56%) en 15.2s
[test_eval] step 3196 — Pass@1 = 57/100 (57%) en 15.7s
[test_eval] step 3243 — Pass@1 = 56/100 (56%) en 17.0s
[test_eval] step 3290 — Pass@1 = 54/100 (54%) en 15.9s
[test_eval] step 3337 — Pass@1 = 55/100 (55%) en 15.1s
[test_eval] step 3384 — Pass@1 = 58/100 (58%) en 15.1s
[test_eval] step 3431 — Pass@1 = 59/100 (59%) en 14.1s
[test_eval] step 3478 — Pass@1 = 51/100 (51%) en 15.7s
[test_eval] step 3525 — Pass@1 = 60/100 (60%) en 13.9s
[test_eval] step 3572 — Pass@1 = 55/100 (55%) en 15.4s
[test_eval] step 3619 — Pass@1 = 56/100 (56%) en 14.8s
[test_eval] step 3666 — Pass@1 = 65/100 (65%) en 13.2s
```

_Interprétation : à compléter à la relecture._

## exp47 — expérience E : LoRA r8 à référence fixe, LR 1e-6 (exp30 + un seul delta) : la dérive KL géométrique vient-elle du LR ? — `exp47_fixedanchor_lr1e-6` (fin 2026-09-29 22:56)

**Best test : step=987 pass_at_1=0.4000**

```
[test_eval] step 47 — Pass@1 = 18/100 (18%) en 26.9s
[test_eval] step 94 — Pass@1 = 21/100 (21%) en 28.7s
[test_eval] step 141 — Pass@1 = 19/100 (19%) en 27.3s
[test_eval] step 188 — Pass@1 = 17/100 (17%) en 25.4s
[test_eval] step 235 — Pass@1 = 15/100 (15%) en 34.4s
[test_eval] step 282 — Pass@1 = 19/100 (19%) en 27.5s
[test_eval] step 329 — Pass@1 = 23/100 (23%) en 24.6s
[test_eval] step 376 — Pass@1 = 22/100 (22%) en 26.9s
[test_eval] step 423 — Pass@1 = 24/100 (24%) en 24.9s
[test_eval] step 470 — Pass@1 = 22/100 (22%) en 26.7s
[test_eval] step 517 — Pass@1 = 26/100 (26%) en 22.2s
[test_eval] step 564 — Pass@1 = 27/100 (27%) en 27.4s
[test_eval] step 611 — Pass@1 = 30/100 (30%) en 24.5s
[test_eval] step 658 — Pass@1 = 35/100 (35%) en 20.2s
[test_eval] step 705 — Pass@1 = 34/100 (34%) en 20.3s
[test_eval] step 752 — Pass@1 = 34/100 (34%) en 17.3s
[test_eval] step 799 — Pass@1 = 40/100 (40%) en 14.6s
[test_eval] step 846 — Pass@1 = 32/100 (32%) en 13.6s
[test_eval] step 893 — Pass@1 = 33/100 (33%) en 10.1s
[test_eval] step 940 — Pass@1 = 35/100 (35%) en 9.2s
[test_eval] step 987 — Pass@1 = 40/100 (40%) en 8.7s
[test_eval] step 1034 — Pass@1 = 34/100 (34%) en 9.4s
[test_eval] step 1081 — Pass@1 = 35/100 (35%) en 9.6s
[test_eval] step 1128 — Pass@1 = 29/100 (29%) en 11.0s
[test_eval] step 1175 — Pass@1 = 33/100 (33%) en 9.8s
[test_eval] step 1222 — Pass@1 = 35/100 (35%) en 8.2s
[test_eval] step 1269 — Pass@1 = 35/100 (35%) en 9.3s
[test_eval] step 1316 — Pass@1 = 31/100 (31%) en 7.9s
[test_eval] step 1363 — Pass@1 = 31/100 (31%) en 6.9s
```

_Interprétation : à compléter à la relecture._

## exp48 — exp46 + LR 1e-6 : un LR plus faible remplace-t-il le reset de ReLoRA ? — `exp48_movingref_lr1e-6` (fin 2026-09-30 17:05)

**Best test : step=987 pass_at_1=0.4000**

```
[test_eval] step 47 — Pass@1 = 17/100 (17%) en 27.2s
[test_eval] step 94 — Pass@1 = 12/100 (12%) en 27.3s
[test_eval] step 141 — Pass@1 = 14/100 (14%) en 28.3s
[test_eval] step 188 — Pass@1 = 18/100 (18%) en 28.4s
[test_eval] step 235 — Pass@1 = 17/100 (17%) en 27.6s
[test_eval] step 282 — Pass@1 = 21/100 (21%) en 26.7s
[test_eval] step 329 — Pass@1 = 22/100 (22%) en 26.5s
[test_eval] step 376 — Pass@1 = 21/100 (21%) en 27.1s
[test_eval] step 423 — Pass@1 = 17/100 (17%) en 33.9s
[test_eval] step 470 — Pass@1 = 30/100 (30%) en 25.7s
[test_eval] step 517 — Pass@1 = 25/100 (25%) en 30.2s
[test_eval] step 564 — Pass@1 = 19/100 (19%) en 25.0s
[test_eval] step 611 — Pass@1 = 33/100 (33%) en 22.2s
[test_eval] step 658 — Pass@1 = 32/100 (32%) en 20.6s
[test_eval] step 705 — Pass@1 = 32/100 (32%) en 18.2s
[test_eval] step 752 — Pass@1 = 37/100 (37%) en 16.1s
[test_eval] step 799 — Pass@1 = 31/100 (31%) en 15.2s
[test_eval] step 846 — Pass@1 = 33/100 (33%) en 13.7s
[test_eval] step 893 — Pass@1 = 37/100 (37%) en 10.9s
[test_eval] step 940 — Pass@1 = 33/100 (33%) en 10.6s
[test_eval] step 987 — Pass@1 = 40/100 (40%) en 8.6s
[test_eval] step 1034 — Pass@1 = 35/100 (35%) en 9.1s
[test_eval] step 1081 — Pass@1 = 36/100 (36%) en 8.4s
[test_eval] step 1128 — Pass@1 = 36/100 (36%) en 10.7s
[test_eval] step 1175 — Pass@1 = 38/100 (38%) en 10.6s
[test_eval] step 1222 — Pass@1 = 35/100 (35%) en 9.6s
[test_eval] step 1269 — Pass@1 = 37/100 (37%) en 8.6s
[test_eval] step 1316 — Pass@1 = 34/100 (34%) en 8.3s
[test_eval] step 1363 — Pass@1 = 33/100 (33%) en 8.9s
[test_eval] step 1410 — Pass@1 = 35/100 (35%) en 19.4s
[test_eval] step 1457 — Pass@1 = 34/100 (34%) en 12.3s
[test_eval] step 1504 — Pass@1 = 30/100 (30%) en 10.3s
[test_eval] step 1551 — Pass@1 = 32/100 (32%) en 8.3s
[test_eval] step 1598 — Pass@1 = 30/100 (30%) en 9.8s
[test_eval] step 1645 — Pass@1 = 32/100 (32%) en 9.6s
[test_eval] step 1692 — Pass@1 = 35/100 (35%) en 12.5s
[test_eval] step 1739 — Pass@1 = 32/100 (32%) en 23.5s
[test_eval] step 1786 — Pass@1 = 24/100 (24%) en 41.3s
[test_eval] step 1833 — Pass@1 = 30/100 (30%) en 42.9s
```

_Interprétation : à compléter à la relecture._

## exp49 — exp46 + purge Adam au ré-ancrage : Adam ou adaptateur, lequel stabilise ReLoRA ? — `exp49_movingref_resetadam` (fin 2026-10-01 05:03)

**Best test : step=1739 pass_at_1=0.4700**

```
[test_eval] step 47 — Pass@1 = 17/100 (17%) en 27.5s
[test_eval] step 94 — Pass@1 = 14/100 (14%) en 25.3s
[test_eval] step 141 — Pass@1 = 26/100 (26%) en 25.4s
[test_eval] step 188 — Pass@1 = 24/100 (24%) en 22.2s
[test_eval] step 235 — Pass@1 = 26/100 (26%) en 23.1s
[test_eval] step 282 — Pass@1 = 33/100 (33%) en 20.9s
[test_eval] step 329 — Pass@1 = 37/100 (37%) en 15.0s
[test_eval] step 376 — Pass@1 = 35/100 (35%) en 13.8s
[test_eval] step 423 — Pass@1 = 36/100 (36%) en 11.1s
[test_eval] step 470 — Pass@1 = 35/100 (35%) en 8.2s
[test_eval] step 517 — Pass@1 = 33/100 (33%) en 7.7s
[test_eval] step 564 — Pass@1 = 39/100 (39%) en 7.6s
[test_eval] step 611 — Pass@1 = 37/100 (37%) en 7.6s
[test_eval] step 658 — Pass@1 = 35/100 (35%) en 7.6s
[test_eval] step 705 — Pass@1 = 33/100 (33%) en 7.7s
[test_eval] step 752 — Pass@1 = 36/100 (36%) en 7.6s
[test_eval] step 799 — Pass@1 = 33/100 (33%) en 7.8s
[test_eval] step 846 — Pass@1 = 34/100 (34%) en 7.8s
[test_eval] step 893 — Pass@1 = 24/100 (24%) en 8.7s
[test_eval] step 940 — Pass@1 = 33/100 (33%) en 7.8s
[test_eval] step 987 — Pass@1 = 26/100 (26%) en 6.7s
[test_eval] step 1034 — Pass@1 = 33/100 (33%) en 6.4s
[test_eval] step 1081 — Pass@1 = 32/100 (32%) en 6.5s
[test_eval] step 1128 — Pass@1 = 35/100 (35%) en 6.2s
[test_eval] step 1175 — Pass@1 = 38/100 (38%) en 5.8s
[test_eval] step 1222 — Pass@1 = 40/100 (40%) en 5.5s
[test_eval] step 1269 — Pass@1 = 38/100 (38%) en 5.7s
[test_eval] step 1316 — Pass@1 = 41/100 (41%) en 5.7s
[test_eval] step 1363 — Pass@1 = 42/100 (42%) en 5.8s
[test_eval] step 1410 — Pass@1 = 40/100 (40%) en 6.0s
[test_eval] step 1457 — Pass@1 = 42/100 (42%) en 5.8s
[test_eval] step 1504 — Pass@1 = 45/100 (45%) en 5.9s
[test_eval] step 1551 — Pass@1 = 42/100 (42%) en 5.7s
[test_eval] step 1598 — Pass@1 = 45/100 (45%) en 5.6s
[test_eval] step 1645 — Pass@1 = 46/100 (46%) en 5.4s
[test_eval] step 1692 — Pass@1 = 43/100 (43%) en 5.5s
[test_eval] step 1739 — Pass@1 = 47/100 (47%) en 5.4s
[test_eval] step 1786 — Pass@1 = 44/100 (44%) en 5.4s
[test_eval] step 1833 — Pass@1 = 46/100 (46%) en 5.4s
```

_Interprétation : à compléter à la relecture._

## exp50 — G=16 + Horizon + borne k3 + ReLoRA, ancre /10 ép. : les deux protections permettent-elles d'espacer les ré-ancrages ? — `exp50_g16_horizon_klclamp_anchor10` (fin 2026-10-01 14:37)

**Best test : step=329 pass_at_1=0.3700**

```
[test_eval] step 47 — Pass@1 = 12/100 (12%) en 25.4s
[test_eval] step 94 — Pass@1 = 19/100 (19%) en 23.3s
[test_eval] step 141 — Pass@1 = 27/100 (27%) en 24.4s
[test_eval] step 188 — Pass@1 = 28/100 (28%) en 22.8s
[test_eval] step 235 — Pass@1 = 33/100 (33%) en 18.6s
[test_eval] step 282 — Pass@1 = 33/100 (33%) en 21.2s
[test_eval] step 329 — Pass@1 = 37/100 (37%) en 18.5s
[test_eval] step 376 — Pass@1 = 36/100 (36%) en 18.8s
[test_eval] step 423 — Pass@1 = 32/100 (32%) en 13.6s
[test_eval] step 470 — Pass@1 = 0/100 (0%) en 9.8s
[test_eval] step 517 — Pass@1 = 0/100 (0%) en 8.5s
[test_eval] step 564 — Pass@1 = 0/100 (0%) en 7.6s
[test_eval] step 611 — Pass@1 = 0/100 (0%) en 7.6s
[test_eval] step 658 — Pass@1 = 0/100 (0%) en 33.4s
[test_eval] step 705 — Pass@1 = 0/100 (0%) en 155.3s
[test_eval] step 752 — Pass@1 = 0/100 (0%) en 85.6s
[test_eval] step 799 — Pass@1 = 0/100 (0%) en 96.0s
[test_eval] step 846 — Pass@1 = 0/100 (0%) en 141.9s
[test_eval] step 893 — Pass@1 = 0/100 (0%) en 154.1s
[test_eval] step 940 — Pass@1 = 0/100 (0%) en 153.6s
[test_eval] step 987 — Pass@1 = 0/100 (0%) en 153.0s
[test_eval] step 1034 — Pass@1 = 0/100 (0%) en 149.6s
[test_eval] step 1081 — Pass@1 = 0/100 (0%) en 177.0s
```

_Interprétation : à compléter à la relecture._

## exp51_depthbal_uniform — F : train + réservoir few-shot (d4 1→8), échantillonnage uniform par profondeur, recette exp43 (G=16 + borne k3) — `exp51_depthbal_uniform` (fin 2026-10-02 13:18)

**Best test : step=705 pass_at_1=0.4700**

```
[test_eval] step 47 — Pass@1 = 22/100 (22%) en 27.3s
[test_eval] step 94 — Pass@1 = 20/100 (20%) en 28.0s
[test_eval] step 141 — Pass@1 = 21/100 (21%) en 26.0s
[test_eval] step 188 — Pass@1 = 24/100 (24%) en 22.1s
[test_eval] step 235 — Pass@1 = 27/100 (27%) en 22.4s
[test_eval] step 282 — Pass@1 = 34/100 (34%) en 21.1s
[test_eval] step 329 — Pass@1 = 36/100 (36%) en 16.5s
[test_eval] step 376 — Pass@1 = 43/100 (43%) en 14.4s
[test_eval] step 423 — Pass@1 = 37/100 (37%) en 13.0s
[test_eval] step 470 — Pass@1 = 35/100 (35%) en 13.3s
[test_eval] step 517 — Pass@1 = 39/100 (39%) en 13.7s
[test_eval] step 564 — Pass@1 = 40/100 (40%) en 13.7s
[test_eval] step 611 — Pass@1 = 38/100 (38%) en 13.1s
[test_eval] step 658 — Pass@1 = 41/100 (41%) en 13.3s
[test_eval] step 705 — Pass@1 = 47/100 (47%) en 13.3s
[test_eval] step 752 — Pass@1 = 42/100 (42%) en 13.3s
[test_eval] step 799 — Pass@1 = 40/100 (40%) en 12.3s
[test_eval] step 846 — Pass@1 = 43/100 (43%) en 8.4s
[test_eval] step 893 — Pass@1 = 11/100 (11%) en 8.0s
[test_eval] step 940 — Pass@1 = 19/100 (19%) en 14.4s
[test_eval] step 987 — Pass@1 = 17/100 (17%) en 21.9s
[test_eval] step 1034 — Pass@1 = 19/100 (19%) en 7.3s
[test_eval] step 1081 — Pass@1 = 21/100 (21%) en 28.1s
[test_eval] step 1128 — Pass@1 = 20/100 (20%) en 11.1s
[test_eval] step 1175 — Pass@1 = 22/100 (22%) en 91.0s
[test_eval] step 1222 — Pass@1 = 20/100 (20%) en 121.6s
[test_eval] step 1269 — Pass@1 = 17/100 (17%) en 164.1s
[test_eval] step 1316 — Pass@1 = 0/100 (0%) en 217.7s
```

_Interprétation : à compléter à la relecture._
