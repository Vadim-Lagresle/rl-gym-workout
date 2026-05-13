# Guide de lecture — AgentGym-RL × TextCraft × Qwen-3B

Ce guide est un plan de lecture du code spécifique à la chaîne **AgentGym-RL +
TextCraft + Qwen-3B + GRPO** dans ce repo. Objectif : pouvoir ouvrir n'importe
quel fichier impliqué et savoir ce qu'il fait, sans relire toute la stack verl.

On reste sur la branche `main` du repo `rl-gym-workout`. Les chemins partent
toujours de `~/rl-gym-workout/`. Tout ce qui touche au serveur d'env est dans
`AgentGym/`, tout ce qui touche au RL trainer est dans `AgentGym-RL/`.

---

## Vue d'ensemble en un schéma

```
                ┌────────────────────────────┐
                │  examples/train/...        │
                │  textcraft_train.4gpu.sh   │  ← script shell qu'on lance
                └────────────┬───────────────┘
                             │ overrides Hydra
                             ▼
              ┌──────────────────────────────┐
              │ AgentGym-RL/verl/            │
              │   agent_trainer/main_ppo.py  │  ← entrypoint Python
              │   agent_trainer/config/      │  ← ppo_trainer.yaml (defaults)
              │   agent_trainer/ppo/         │  ← boucle fit() + GRPO algo
              │   workers/agent_fsdp_workers │  ← FSDP actor / ref / rollout
              │   workers/rollout/agent_…    │  ← vLLM rollout multi-tour
              │   utils/agentgym/client.py   │  ← factory EnvClient
              └──────┬─────────────┬─────────┘
                     │ HTTP        │ same Python proc
                     ▼             ▼
        ┌────────────────────┐   ┌──────────────────┐
        │ TextCraft server   │   │ TextCraftEnvClient│
        │ FastAPI uvicorn    │←──│  (AgentGym/agen-  │
        │ :36005             │   │  tenv/.../env.py) │
        │ AgentGym/agentenv- │   └──────────────────┘
        │ textcraft/server.py│
        └────────────────────┘
```

---

## 1. Scripts shell (point d'entrée utilisateur)

Tous les scripts sont dans `examples/train/AgentGym-RL/` et
`examples/eval/`. Ce sont juste des wrappers Hydra qui appellent les
entrypoints Python avec des overrides.

| Fichier | Rôle |
|---|---|
| `examples/train/AgentGym-RL/textcraft_train.sh` | Recette papier upstream (Qwen-7B, ≥ 8 GPUs implicite, `rounds_ctrl=fixed`) |
| `examples/train/AgentGym-RL/textcraft_train.4gpu.sh` | **Notre version** : Qwen-3B, 4× A100 40 Go, sans ScalingInter, avec workarounds NCCL/load_format/VRAM |
| `examples/eval/textcraft_eval.sh` | Eval papier upstream (suppose un checkpoint training) |
| `examples/eval/textcraft_eval.local.sh` | Notre eval upstream adaptée single-GPU (sur HF brut, sans merger) |

Tout ce que les scripts shell font : `python3 -m verl.agent_trainer.main_ppo
<overrides>` ou `main_generation` côté eval. Hydra résout les overrides
contre `agent_trainer/config/ppo_trainer.yaml` ou `generation.yaml`.

---

## 2. Configs Hydra

Dans `AgentGym-RL/verl/agent_trainer/config/` :

- `ppo_trainer.yaml` — config par défaut pour `main_ppo`. **C'est la
  source de vérité pour tous les hyperparamètres.** Notable :
  - `data.train_batch_size: 1024` (à overrider, c'est le total queries/step)
  - `actor_rollout_ref.actor.strategy: fsdp` (le seul supporté)
  - `actor_rollout_ref.rollout.load_format: dummy_dtensor` (vLLM init avec
    poids aléatoires puis re-sync via FSDP DTensor — NE PAS forcer
    `safetensors`, n'existe pas dans l'enum `LoadFormat` du fork verl)
  - `actor_rollout_ref.rollout.tensor_model_parallel_size: 2` (default 2,
    on override à 1 pour 4 GPUs DP × 1 TP)
  - `algorithm.adv_estimator: gae` (default ; on override à `grpo`)
  - `algorithm.rounds_ctrl.type: fixed` (option : `scaling_inter_stepwise`)
- `generation.yaml` — config pour `main_generation` (eval-only path).

---

## 3. Entrypoints Python

### 3.a Training : `verl/agent_trainer/main_ppo.py` (96 lignes)

C'est petit, lisible d'un coup. Trois étapes :

1. **Décorateur `@hydra.main`** pour charger `ppo_trainer.yaml`.
2. `main_task` est un Ray remote qui :
   - dump la config résolue,
   - instancie le tokenizer,
   - construit le `role_worker_mapping` : `Actor=ActorRolloutRefWorker`,
     `RefPolicy=ActorRolloutRefWorker`, `Critic=CriticWorker` (les trois
     pointent vers le même `agent_fsdp_workers.py`),
   - construit le `RayPPOTrainer` et appelle `init_workers()` puis `fit()`.

C'est volontairement un mince glue layer. Toute la logique réelle est dans
`ray_trainer.py`.

### 3.b Eval : `verl/agent_trainer/main_generation.py` (154 lignes)

Path d'eval sans training. Il :
- lit `<task>_test.json` (les `item_id` à évaluer) + d'autres fichiers
  dans `data.path` pour faire un breakdown par sous-catégorie,
- construit un Ray WorkerGroup qui contient juste un `ActorRolloutRefWorker`
  en mode rollout (pas d'actor à train, pas de ref),
- lit `env_client.conversation_start` pour construire le prompt initial
  (cf. § 5.b),
- batche les item_ids et fait des `wg.generate_sequences(data)` qui
  déclenchent un rollout multi-tour TextCraft de bout en bout,
- agrège : `Pass@N = mean(max_over_samples(reward) > 0)`.

C'est le path qu'on lance avec `examples/eval/textcraft_eval.sh` après un
training réussi (sur un checkpoint mergé via `scripts/model_merger.py`).

---

## 4. Boucle GRPO côté trainer

### 4.a `agent_trainer/ppo/ray_trainer.py` (884 lignes) — la fit() loop

C'est le cœur. À survoler dans cet ordre :

| Lignes | Rôle |
|---:|---|
| 194-242 | **`RoundsScheduler`** : abstraction qui décide combien de tours max par épisode. `FixedRoundsScheduler` (constant) vs `StepRoundsScheduler` (ScalingInter, paliers). |
| 377-541 | **`RayPPOTrainer.__init__`** : résout les WorkerGroup specs, instancie le tokenizer, **construit `RLHFDataset`** (cf. § 4.b). |
| 543-555 | Branche `rounds_ctrl.type` sur fixed ou scaling_inter_stepwise → instancie le bon scheduler. |
| 557-611 | **`init_workers()`** : spawn des Ray actors et appels successifs `critic_wg.init_model()`, **`ref_policy_wg.init_model()`**, `actor_rollout_wg.init_model()`. C'est ici que la chaîne FSDP wrap + NCCL est exercée pour chaque rôle. |
| 613-733 | `_save_checkpoint`, `_balance_batch`, helpers de pad / unpad. |
| 735-884 | **`fit()`** : la vraie boucle. Pour chaque step :<br>1. `dataloader_iter.next()` → `gen_batch` (les `item_id` du step)<br>2. `actor_rollout_wg.generate_sequences(gen_batch)` → rollout multi-tour vLLM (déclenche les appels HTTP TextCraft)<br>3. `ref_policy_wg.compute_ref_log_prob(batch)` → KL reference<br>4. `core_algos.compute_grpo_outcome_advantage(...)` → advantages relatifs au groupe<br>5. `actor_rollout_wg.update_actor(batch)` → step PPO/GRPO sur l'actor FSDP<br>6. log metrics, save checkpoint si `save_freq` atteint |

### 4.b `agent_trainer/ppo/core_algos.py` (387 lignes) — la math GRPO

Deux fonctions critiques :

| Lignes | Fonction | Rôle |
|---:|---|---|
| 113-172 | `compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon)` | Pour chaque prompt, regroupe les `n` rollouts par `index`, calcule `(reward - group_mean) / (group_std + epsilon)`. Si `std == 0` → advantage = 0 (= step perdu). C'est exactement l'origine de notre §"Insight 5" sur N=2 et p≈0.18 → 70 % de steps perdus. |
| 276-330 | `compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange)` | Loss PPO classique : `-min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)` masquée par `eos_mask`. |

On a vérifié ces deux fonctions byte-for-byte dans `scratch/smoke_verl_test.py`
(3/3 PASS, cf. WORKLOG du 2026-05-12).

### 4.c Worker FSDP : `verl/workers/agent_fsdp_workers.py` (~800 lignes)

C'est le binding entre verl et PyTorch FSDP / vLLM. Sections utiles :

| Lignes | Fonction | Rôle |
|---:|---|---|
| 67-133 | `ActorRolloutRefWorker.__init__` | Init NCCL process group, `device_mesh`, lit le rôle (`actor`, `ref`, `actor_rollout`, etc.) et configure `_is_actor / _is_ref / _is_rollout`. |
| 135-281 | `_build_model_optimizer(role)` | Charge le modèle HF en `bf16` sur CPU, le wrap FSDP. **Pour `role='ref'` on a `CPUOffload(offload_params=True)`** (line 243) — c'est pour ça que la ref policy ne pèse pas le modèle entier sur GPU. AdamW optimizer construit seulement pour `role='actor'`. |
| 283-309 | `_build_rollout()` | Instancie `vLLMRollout` (notre fork verl) + `FSDPVLLMShardingManager` qui re-sync les poids FSDP actor vers vLLM avant chaque batch de génération. |
| 312-385 | `init_model()` | Orchestration : selon les `_is_*` flags, appelle `_build_model_optimizer` pour actor + ref, et `_build_rollout` pour rollout. C'est ce qui est appelé par `ray_trainer.init_workers()`. |

### 4.d vLLM rollout multi-tour : `verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` (369 lignes)

C'est **le cœur du rollout interactif**. À lire intégralement, c'est dense
mais court. Structure :

- `class vLLMRollout`:
  - `__init__` (lignes 50-120) : instancie `LLM(...)` du fork
    `verl.third_party.vllm`, lit la `agentgym_config`, construit le
    `env_client` via `init_env_client`.
  - `_validate_config` : sanity sur prompt_length × n / 8 < max_num_batched_tokens.
  - **`generate_sequences(data)`** (la méthode principale) :
    1. Pour chaque prompt × chaque `n` rollouts, crée un `RolloutHandler`
       (sous-classe qui maintient input_ids, attention_mask, position_ids,
       loss_mask, status, reward et l'historique de messages).
    2. Boucle `for round in range(max_rounds)`:
       - Filtre les rollouts encore actifs (`status == 'active'`).
       - Construit le prompt courant via le chat template Qwen,
         tokenize, génère via vLLM (`inference_engine.generate(...)`).
       - Pour chaque rollout actif, appelle `env_client.step(action)`
         via HTTP, met à jour le state, accumule reward + loss_mask
         pour les tokens assistant.
       - Si `done==True` → marque le rollout comme inactif.
    3. À la fin, retourne un `DataProto` avec les trajectoires complètes
       (input_ids concaténés tous tours, loss_mask qui isole les tokens
       générés par l'assistant, task_scores = reward final).

C'est l'extension principale de verl que les auteurs ont écrite — la
version originale ne supporte que des rollouts 1-shot. Tout le scaffold
"multi-tour avec env HTTP entre les tours" vit ici.

---

## 5. Communication avec le serveur TextCraft

Tu m'as dit que c'est pas ce qui t'intéresse le plus, donc je résume juste
le quoi (pas le comment du serveur) :

### 5.a Le serveur — `AgentGym/agentenv-textcraft/agentenv_textcraft/server.py` (65 lignes)

FastAPI uvicorn qui expose 8 endpoints REST :

| Endpoint | Méthode | Payload entrée | Retour |
|---|---|---|---|
| `/create` | POST | `{}` (ou commands/goal customs) | `{"id": <int>, "observation": <str description recettes + goal>, "done": false, "reward": 0}` |
| `/reset` | POST | `{"id": <int>, "data_idx": <int>}` | `{"observation": <str>, "done": false, "reward": 0}` |
| `/step` | POST | `{"id": <int>, "action": <str>}` | `{"observation": <str feedback>, "done": <bool>, "reward": <0 ou 1>}` |
| `/observation` | GET | `id=<int>` | `<str>` |
| `/commands` | GET | `id=<int>` | liste des recettes valides |
| `/goal` | GET | `id=<int>` | str du goal |
| `/detail` | GET | `id=<int>` | infos debug |
| `/close` | POST | `{"id": <int>}` | rien d'utile |

Les seuls qui comptent dans la boucle RL : **`/create`** au tout début (par
trainer), **`/reset`** au début de chaque épisode, **`/step`** à chaque tour.

### 5.b Le client Python — `AgentGym/agentenv/agentenv/envs/textcraft.py` (130 lignes)

`TextCraftEnvClient(BaseEnvClient)` :

- **`conversation_start`** (class attribute, lignes 14-28) : le system prompt
  initial envoyé à l'agent (les règles "craft / get / inventory" + format
  `Thought: / Action:`). C'est la première chose qui rentre dans le LLM —
  on l'a vérifié byte-for-byte vs notre prompt d'eval (cf. WORKLOG
  smoke (a), 1382 chars identiques).
- `__init__` : appelle `/create` au serveur, stocke `env_id`.
- `reset(idx)` : appelle `/reset` avec `data_idx=idx` pour charger une
  tâche spécifique du dataset.
- **`step(action)`** : c'est ici que l'action du LLM est :
  - parsée par regex pour extraire ce qui suit `Action:` (line 86) — si >
    une `Action:` trouvée, retourne l'erreur "Only one 'Action' is allowed
    per response" sans même appeler le serveur,
  - nettoyée des caractères non-alphanum (line 94),
  - envoyée à `/step` du serveur,
  - le retour `{observation, reward, done}` est wrappé dans un `StepOutput`.
- `observe()` retourne juste la dernière `observation`.

### 5.c Côté verl — `verl/utils/agentgym/client.py` (55 lignes)

Factory `init_env_client(args)` qui mappe `args.task_name` → classe
`*EnvClient` correspondante (`TextCraftEnvClient`, `WebshopEnvClient`,
etc.). Retry x10 si le serveur n'est pas encore up. C'est appelé une fois
au démarrage par `vllm_rollout.py::__init__`.

---

## 6. Données train/eval

| Fichier | Contenu |
|---|---|
| `AgentEval/train/textcraft_train.json` | 374 items, format `[{"item_id": "textcraft_<idx>"}, ...]` |
| `AgentEval/eval/textcraft_test.json` | 100 items, même format |
| `AgentEval/textcraft/textcraft_test.json` | (copie de l'eval, requis par `main_generation.py`) |
| `AgentEval/textcraft/textcraft_default.json` | (idem, fait office de "catégorie" pour le breakdown) |

Le JSON ne contient **que des `item_id`**, pas le prompt ni le goal. Le
goal est résolu côté serveur en faisant `data_idx=idx_dans_la_liste` :
le `TextCraft_Wrapper` côté serveur tient une liste interne ordonnée des
374 tâches train (resp. 100 eval) et utilise l'index pour piocher le goal,
les recettes accessibles, et les ingrédients dans le crafting tree
Minecraft.

---

## 7. Carte mentale pour modifier quelque chose

| Tu veux toucher… | Va dans… |
|---|---|
| Le system prompt | `AgentGym/agentenv/agentenv/envs/textcraft.py`, `conversation_start` |
| La reward function | côté serveur : `AgentGym/agentenv-textcraft/agentenv_textcraft/env_wrapper.py::step`. Côté trainer pas grand-chose à toucher, on consomme juste `reward` brut. |
| Le calcul des advantages | `AgentGym-RL/verl/agent_trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` |
| La loss PPO | `core_algos.py::compute_policy_loss` |
| Le scheduler `max_rounds` (curriculum ScalingInter) | `agent_trainer/ppo/ray_trainer.py::RoundsScheduler` |
| La boucle multi-tour côté trainer | `verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py::generate_sequences` |
| Le format du prompt entre les tours (chat template) | `vllm_rollout.py::RolloutHandler.add_assistant_message` / `add_user_message` |
| Les hparams | les overrides Hydra dans `examples/train/AgentGym-RL/textcraft_train.4gpu.sh` |
| Les défauts non overridés | `agent_trainer/config/ppo_trainer.yaml` |

---

## 8. Différences entre notre setup et la recette papier

(rappel pour ne pas oublier en relisant le code)

- Modèle : Qwen2.5-**3B** (papier idem pour AgentGym-RL-3B = 75/100 ;
  papier ScalingInter SOTA = **7B**).
- GPUs : 4× A100 **40 GB** (vs cluster multi-noeud papier, probablement
  8× A100 80 GB).
- `rollout.n=8` ✅ (= papier).
- `train_batch_size=8` ⚠️ (papier=32) — limite VRAM 40 GB.
- `max_response_length=4096` ⚠️ (papier=10240) — limite VRAM.
- `max_tokens=512` ✅ (= papier).
- `ppo_mini_batch_size=8` ✅ (= papier).
- `ppo_epochs=2` ✅ (= papier).
- `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` ✅ (= papier).
- `lr=1e-6` ✅ (= papier).
- `gradient_checkpointing=True` ⚠️ (papier ne l'utilise probablement pas,
  c'est notre concession à 40 GB).
- `actor.fsdp_config.optimizer_offload=True` ⚠️ (papier=False, AdamW
  states restent sur GPU dans le setup papier qui a 80 GB ; nous on
  offload pour récupérer ~10 GB de VRAM).
- `rounds_ctrl=fixed`, `rounds=30` ✅ (= papier pour AgentGym-RL-3B —
  pas de ScalingInter ici).

Les écarts marqués ⚠️ sont les seules concessions à notre VM 4× 40 GB. Les
✅ matchent papier-exact.
