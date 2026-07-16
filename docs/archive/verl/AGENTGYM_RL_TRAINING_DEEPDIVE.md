# AgentGym-RL — plongée dans le training PPO / GRPO multi-tour

Document de travail pour le dépôt `rl-gym-workout` (fork AgentGym-RL + AgentGym).
Objectif : comprendre la boucle de training, les hyperparamètres Hydra, les hooks de customisation, et estimer le risque **single-GPU** vs rebasculer sur **TRL+GRPO**.

**À noter :** les configs YAML vivent sous `AgentGym-RL/verl/agent_trainer/config/` (pas `verl/trainer/config/`). Les scripts d'exemple sont sous `examples/train/AgentGym-RL/` à la racine du dépôt.

---

## 1. Vue d'ensemble

### 1.1 Schéma d'architecture (texte)

```
[Driver: RayPPOTrainer.fit]
 │
 ├─ DataLoader → RLHFDataset (item_id, raw_prompt, tokens)
 │
 ├─ RPC → RayWorkerGroup(ActorRolloutRefWorker)
 │     │
 │     ├─ FSDP : actor (+ ref policy colocalisé)
 │     └─ generate_sequences() → vLLMRollout.generate_sequences()
 │           │
 │           ├─ verl.third_party.vllm.LLM (inférence, poids sync depuis FSDP)
 │           └─ Pour chaque slot du batch × n :
 │                 init_env_client() → TextCraftEnvClient (HTTP vers serveur AgentGym)
 │                 Boucle rounds: vLLM.generate → decode → env.step(action)
 │
 ├─ (driver) compute_advantage (GAE ou GRPO, etc.)
 ├─ RPC update_critic (si GAE) / update_actor
 └─ logging (wandb/console), checkpointing

Environnement TextCraft : TextCraftEnvClient ↔ POST /create, /reset, /step, /close
```

### 1.2 Qui orchestre quoi

| Composant | Rôle |
|-----------|------|
| `verl.agent_trainer.main_ppo` | `hydra.main` + `ray.init` + `RayPPOTrainer` |
| `RayPPOTrainer` (`ray_trainer.py`) | Boucle `fit()`, avantages, métriques, `RoundsScheduler` |
| `ActorRolloutRefWorker` (`agent_fsdp_workers.py`) | FSDP, rollout vLLM, `compute_log_prob`, `update_actor` |
| `vLLMRollout` (`vllm_rollout.py`) | Rollout multi-tour, agrégation reward épisodique |
| `init_env_client` (`client.py`) | Fabrique `TextCraftEnvClient` (package `agentenv`) |
| `RLHFDataset` (`rl_dataset.py`) | JSON d'`item_id`, prompts via `conversation_start` |

### 1.3 Écarts par rapport à un PPO « chat » classique

- **Multi-tour :** une seule « réponse » tensorisée = **toute la trajectoire** (plusieurs tours user/assistant) jusqu'à `max_rounds` ou `done` (`vllm_rollout.py`, boucle `while rounds < max_rounds`).
- **Récompense sparse par épisode :** un scalaire par rollout placé sur le **dernier token utile** de la portion réponse (`reward_tensor[i, valid_response_length[i]-1] = scores[i]`), pas de reward dense par token intermédiaire.
- **RoundsScheduler (ScalingInter-RL) :** borne supérieure **`max_rounds` passée au rollout** (`gen_batch.meta_info['max_rounds']`) peut **croître** selon l'étape d'entraînement (`StepRoundsScheduler`), pour allonger progressivement l'horizon d'interaction.

---

## 2. Boucle de training détaillée

### 2.1 `RayPPOTrainer.fit()` — structure

Le driver charge éventuellement un checkpoint, puis pour chaque batch : génération, alignement du batch répété, recompute log-probs, ref, critic (si GAE), rewards/advantages, mises à jour.

Extrait central (structure) :

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:735-884
def fit(self):
    """
    The training loop of PPO.
    """
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            gen_batch = batch.pop(...)
            gen_batch.meta_info['global_steps'] = self.global_steps
            gen_batch.meta_info['max_rounds'] = self.rounds_scheduler.get_rounds()
            with _timer('gen', timing_raw):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)
            self._balance_batch(batch, metrics=metrics)
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            if self.use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
            reward_tensor = batch.batch['scores']
            batch.batch['token_level_scores'] = reward_tensor
            batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)
            actor_output = self.actor_rollout_wg.update_actor(batch)
            logger.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
            self.rounds_scheduler.step()
```

**Évaluation périodique :** le champ `trainer.test_freq` est présent dans `ppo_trainer.yaml`, mais **cette boucle `fit()` n'implémente pas** de validation périodique (contrairement à certains docs upstream verl). Seuls comptent les métriques batch (dont sommes de `task_scores`, longueurs, etc.) et `save_freq`.

### 2.2 Génération des rollouts multi-tour

1. Le driver envoie `gen_batch` à tous les ranks via `ActorRolloutRefWorker.generate_sequences` (`agent_fsdp_workers.py` ~435), qui appelle `self.rollout.generate_sequences(prompts)` (vLLM + AgentGym).
2. Dans `vLLMRollout.generate_sequences` :
   - Un **client env par ligne** : `env_clients = [init_env_client(...) for _ in range(batch_size)]` avec `batch_size = prompts × config.n`.
   - **Reset + observe** : `reset(item_id)`, `observe()`, puis message utilisateur initial.
   - **Boucle** `while rounds < max_rounds and not all_done_flag` : collecte des prompts non terminés, **`self.inference_engine.generate`** sur les `prompt_token_ids`, puis **`agent_step`** en parallèle (thread pool) : décode la réponse, `env_clients[idx].step(content)`, met à jour le message utilisateur avec l'observation.

Appel environnement au reset et à chaque tour :

```python
# AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:213-241
env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
for idx, rollout_handler in enumerate(rollout_handler_ls):
    try:
        env_clients[idx].reset(rollout_handler.item_id)
        task = env_clients[idx].observe()
        rollout_handler.add_user_message(self.tokenizer, task)
        ...
def agent_step(i, idx):
    content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
    rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
    step_output = env_clients[idx].step(content)
    state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
        step_output.state, step_output.reward, step_output.done,
    )
    rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
```

### 2.3 Agrégation des rewards

- **Par épisode / trajectory :** `rollout_handler.score` est mis à jour à chaque `step` avec `step_output.reward` ; la **valeur finale** après tous les tours est dans `scores[i]`.
- **Placement token-level :** un seul non-zéro par séquence (dernier token de la réponse valide) — reward « outcome » pour toute la trajectoire compressée dans `response_*` tensors :

```python
# AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:323-326
reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)  # (bs, response_length)
valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
for i in range(len(scores)):
    reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
```

- Les métriques donnent aussi `task_scores` identiques à ce tensor et `task_rounds` (nombre de tours par agent).

Il n'y a **pas** de reward cumulé explicite par token de policy loss hors ce dernier slot ; les avantages GAE/GRPO voient le reste comme zéro (masque `response_mask`).

### 2.4 RoundsScheduler (= ScalingInter-RL)

Deux implémentations (`ray_trainer.py`) :

**Fixe :**

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:208-219
class FixedRoundsScheduler(RoundsScheduler):
    def __init__(self, rounds: int):
        self.max_rounds = rounds
    def get_rounds(self):
        return self.max_rounds
```

**Stepwise (curriculum du nombre de tours) :** `max_rounds` prend des valeurs dans `rounds_ls`, avec changement de palier tous les `steps_scaling_inter` steps (incohérences possibles avec `set_global_steps` au resume — code research).

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:222-245
class StepRoundsScheduler(RoundsScheduler):
    def __init__(self, steps_scaling_inter: int, rounds_ls: List[int]):
        self.rounds_ls = rounds_ls
        self.steps_scaling_inter = steps_scaling_inter
        self.max_rounds = rounds_ls[0]
        self.current_stage = 0
        self.global_steps = 1
    def step(self):
        if self.current_stage + 1 < len(self.rounds_ls) and self.global_steps % self.steps_scaling_inter == 0:
            self.current_stage += 1
            self.max_rounds = self.rounds_ls[self.current_stage]
        self.global_steps += 1
    def get_rounds(self):
        return self.max_rounds
```

Instanciation depuis Hydra :

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:543-549
if self.config.algorithm.rounds_ctrl.type == 'fixed':
    self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
elif self.config.algorithm.rounds_ctrl.type == 'scaling_inter_stepwise':
    self.rounds_scheduler = StepRoundsScheduler(
        steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
        rounds_ls=self.config.algorithm.rounds_ctrl.rounds)
```

Script exemple **ScalingInter-RL** : `examples/train/ScalingInter-RL/textcraft_train.sh` passe
`algorithm.rounds_ctrl.type=scaling_inter_stepwise`, `steps_scaling_inter=100`, `rounds=[10,20,30]`.

---

## 3. Hyperparamètres clés

### 3.1 Défauts `ppo_trainer.yaml` (extraits)

| Clé | Défaut (fichier) | Commentaire |
|-----|------------------|-------------|
| `data.train_batch_size` | 1024 | Taille **prompt** par step ; le batch réel rollout est × `rollout.n` |
| `actor_rollout_ref.rollout.n` | 1 | **>1** pour GRPO (plusieurs trajectoires / prompt) |
| `algorithm.adv_estimator` | `gae` | Passer `grpo` pour sans critic |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | Pénalité KL dans **reward** si pas `use_kl_loss` |
| `actor_rollout_ref.actor.use_kl_loss` | False | **True** typique avec GRPO (KL dans la **loss acteur**) |
| `actor_rollout_ref.actor.ppo_epochs` | 1 | |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 2 | **Doit être ≤ world_size** ; mettre **1** sur 1 GPU |
| `trainer.n_gpus_per_node` | 8 | |
| `algorithm.rounds_ctrl.rounds` | 15 | Max tours (mode fixed) |

Contrainte validée au démarrage :

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:440-443
real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
assert real_train_batch_size % n_gpus == 0, \
    f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
```

**Single GPU :** `train_batch_size × rollout.n` doit être divisible par **1** (toujours vrai), mais il faut surtout ajuster **mémoire** (vLLM + FSDP + longueurs).

### 3.2 Overrides dans `examples/train/AgentGym-RL/textcraft_train.sh`

Le script enchaîne Hydra, par ex. :

- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.n=${rollout_sample_num}` (8)
- `data.train_batch_size=32`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.rounds_ctrl.type=fixed` et `rounds=30`

**Manque dans ce script :** pas d'override explicite de `trainer.n_gpus_per_node` — il hérite donc de **8** du YAML par défaut, ce qui est incohérent avec `tensor_model_parallel_size=1` sur une machine 1×GPU ; pour du single-GPU il faut **`trainer.n_gpus_per_node=1`** (et `trainer.nnodes=1`).

### 3.3 Fichiers YAML du dépôt AgentGym-RL

- `verl/agent_trainer/config/ppo_trainer.yaml` — config principale Hydra (`config_name='ppo_trainer'`).
- `verl/agent_trainer/config/generation.yaml` — `main_generation.py` (eval).
- `verl/agent_trainer/config/sft_trainer.yaml`, `evaluation.yaml` — hors boucle PPO agentique principale.

---

## 4. Hooks de customisation

### 4.1 Reward shaping (process reward par tour)

- **Signal principal :** récompense **environnement** dans `step_output.reward` ; pour du shaping, le point naturel est **`TextCraftEnvClient.step`** (ou le serveur derrière `/step`) :
  `AgentGym/agentenv/agentenv/envs/textcraft.py` **lignes 85–106** (parsing d'action puis POST).
- **Côté rollout :** après `step`, on pourrait fusionner `reward` avec un bonus avant `rollout_handler_ls[idx].score = step_output.reward` dans `agent_step` (`vllm_rollout.py` ~233–238) — nécessite un fork.
- Le bloc `custom_reward_function` du YAML **n'est pas câblé** dans `RayPPOTrainer.fit()` du chemin AgentGym observé ; ne pas compter dessus sans grep/branch supplémentaire.

### 4.2 Curriculum sur le dataset

- **Sélection d'items :** JSON chargé par `RLHFDataset` (`datasets.load_dataset("json", ...)`), ordre **RandomSampler** si `data.shuffle=True` (`ray_trainer.py` ~517–524). Pas de champ « difficulté » exploité dans `__getitem__`.
- **Curriculum par longueur d'interaction :** `algorithm.rounds_ctrl.type=scaling_inter_stepwise` + liste `rounds`.
- Pour un curriculum par difficulté : **filtrer / réordonner le JSON**, ou surcharger `RLHFDataset.__getitem__` / `_read_files_and_tokenize`.

### 4.3 Action parsing custom (TextCraft)

Parser strict dans `TextCraftEnvClient.step` :

```python
# AgentGym/agentenv/agentenv/envs/textcraft.py:85-96
def step(self, action: str) -> StepOutput:
    action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
    if len(action_matches) > 1:
        return StepOutput(
            state="Error: Only one 'Action' is allowed per response. ...",
            reward=0,
            done=False,
        )
    action = action_matches[-1] if action_matches else ""
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
```

Pour plus de tolérance : sous-classer le client ou patcher ce module (attention : changement du comportement de reward / done).

### 4.4 Logging des trajectoires

Si `global_steps` est présent et `rollout_log_dir` configuré, écriture JSON par rank :

```python
# AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:328-342
if global_steps:
    try:
        os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
        with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
            json_msg = []
            for idx, msgs in enumerate(messages):
                records = {
                    "item_id": rollout_handler_ls[idx].item_id,
                    "conversations": [msg.to_dict() for msg in msgs],
                    "reward": scores[idx]
                }
                json_msg.append(records)
            json.dump(json_msg, f, ensure_ascii=True, indent=4)
```

Point d'accroche debug : ajouter des champs dans `records`, ou logger dans `agent_step` avant/après `env.step`.

---

## 5. Configuration indicative single-GPU (1× A100 40 Go, Qwen-3B-Instruct, GRPO)

### 5.1 Overrides Hydra (exemple)

Adapter `examples/train/AgentGym-RL/textcraft_train.sh` :

```bash
trainer.n_gpus_per_node=1
trainer.nnodes=1
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.model.path=/home/v.lagresle/rl-gym-workout/models/Qwen2.5-3B-Instruct
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
data.train_batch_size=8
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

Réduire si OOM : `data.max_response_length`, `actor_rollout_ref.rollout.max_model_len`, `rollout.max_tokens`, `gpu_memory_utilization`, `rollout.n`, `train_batch_size`.

### 5.2 GRPO vs PPO+GAE dans ce fork

- **Config :** `algorithm.adv_estimator=grpo` désactive le critic et utilise `compute_grpo_outcome_advantage` :

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:421-424
elif self.config.algorithm.adv_estimator == 'grpo':
    self.use_critic = False
```

```python
# AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py:151-156
elif adv_estimator == 'grpo':
    advantages, returns = core_algos.compute_grpo_outcome_advantage(...)
```

- **Implémentation :** présente dans `core_algos.py` (outcome reward, normalisation par groupe via `uid`). Les `uid` sont dupliqués avec `batch.repeat(..., interleave=True)` pour aligner les `n` rollouts du même prompt.
- **KL :** avec GRPO le YAML commente `use_kl_loss: True for GRPO` ; la loss acteur ajoute une pénalité KL vs `ref_log_prob` si `use_kl_loss` (`dp_actor.py` ~210–258). Le chemin **ref policy reste instancié** dans `main_ppo` (toujours `Role.RefPolicy` dans le mapping) : coût mémoire même sans critic.

**Alerte code :** dans `compute_grpo_outcome_advantage`, ligne ~149, `torch.std(torch.tensor([id2score[idx]]))` semble suspect (liste extra) — à valider empiriquement ; possible bug fork / cas limites.

### 5.3 Ordres de grandeur (très conservateurs)

- **Mémoire :** Qwen-3B bf16 ~6 Go de poids ; **hybride FSDP + vLLM** + activations longues (`max_response_length` jusqu'à 10k+ tokens) peut monter **vite** au-delà de 20–30 Go ; prévoir **gradient checkpointing** déjà dans YAML, baisser `max_model_len` / longueur effective.
- **Temps / step :** dominant = **génération multi-tour** + **latence HTTP** env ; pour batch 8–32 et dizaines de tours, **minutes par step** n'est pas rare — dépend du serveur TextCraft et de `max_tokens` par tour.

---

## 6. Risques connus et arbitrage TRL vs verl

### 6.1 Parité eval (`main_generation.py`) vs training — imports vLLM

Les deux utilisent **`verl.third_party.vllm.LLM`** pour le moteur hybrid, pas seulement `pip vllm` :

```python
# AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:40-42
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
```

Le package `verl.third_party.vllm` **sélectionne** un shim selon la version installée (`__init__.py` lignes 30–59) — versions non listées → `ValueError`. Si l'eval casse sur votre stack vLLM, **le rollout d'entraînement est exposé au même risque** (même import + même `LLM`).

### 6.2 FSDP / multi-GPU

- `main_ppo` n'autorise que `actor.strategy == 'fsdp'`.
- `ActorRolloutRefWorker` : chemins explicites pour world_size > 1 (ex. après `compute_log_prob`, reshard FSDP).
- **Single-GPU :** `tensor_model_parallel_size=1`, `n_gpus_per_node=1` — le code prévoit world_size 1, mais reste **dépendant du fork vLLM + Ray + FSDP**.

### 6.3 Décision (aide à trancher)

| Critère | Rester sur AgentGym-RL / verl | Migrer vers TRL+GRPO |
|--------|------------------------------|----------------------|
| Multi-tour + HTTP AgentGym déjà intégré | Fort + | À recâbler (boucle env + tensorisation) |
| Maturité / debug stack | Fork research, asserts KL/critic bizarres, docs test_freq non alignées | Écosystème HF plus standard |
| Single GPU | Faisable mais **serré** (hybrid engine) | Souvent plus simple pour 1 GPU |
| Dépendance vLLM custom | **Critique** (versions pinnées) | Contrôle plus direct si generation HF/vLLM séparé |

**Recommandation prudente :** si le blocage actuel est **vLLM third_party** ou **stabilité** : TRL+GRPO vaut le POC ; si l'objectif est **repro papers AgentGym-RL** avec peu de réécriture : rester sur verl en **pinnant** la stack testée upstream et en fixant `trainer.n_gpus_per_node`.

---

## 7. Points obscurs / honnêteté

- Synchronisation exacte **trainer.global_steps** vs **StepRoundsScheduler.global_steps** après reprise de checkpoint peut être **décalée** (deux compteurs).
- `compute_grpo_outcome_advantage` : calcul d'écart-type par groupe mérite relecture / tests.
- `RolloutHandler` : typo `assistat_prefix_msg`, format **Qwen codé en dur** — autres familles de modèles = friction.

---

## 8. Conclusions (5 lignes)

1. **Training single-GPU :** plausible sur 40 Go avec Qwen-3B **si** on force `trainer.n_gpus_per_node=1`, `tensor_model_parallel_size=1`, et qu'on réduit séquences / batch / `rollout.n` ; le **moteur hybride FSDP+vLLM** reste plus lourd et fragile qu'un stack « trainer léger ».
2. **Risque principal :** même chaîne **`verl.third_party.vllm`** que l'eval — si l'eval casse sur la version vLLM, **l'entraînement est probablement affecté** aussi.
3. **GRPO :** **natif** (`adv_estimator=grpo`, `use_kl_loss`, pas de critic spawné) ; la ref policy **reste** dans le graphe Ray/FSDP.
4. **Recommandation :** garder verl si on veut **coller au fork** et accepter de pinner l'environnement ; envisager **TRL+GRPO** si la priorité est **maintenabilité 1 GPU** et moins de dépendance au shim vLLM du fork.
5. **Suspect pour notre use-case :** (a) imports / versioning **`verl.third_party.vllm`**, (b) **`main_ppo` + ref policy toujours active** + assert critic YAML, (c) **reward uniquement sur dernier token** + **GRPO std** dans `core_algos.py` ligne ~149.

---

## 9. Portions de code les plus suspectes (single-GPU / maintien)

1. `AgentGym-RL/verl/third_party/vllm/__init__.py:40-60` — sélecteur de version **strict** ; mismatch → crash commun eval/train.
2. `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py:40-42` + construction **`LLM(...)`** — dépendance au fork hybride FSDP↔vLLM.
3. `AgentGym-RL/verl/agent_trainer/main_ppo.py:53-60` — **`assert actor.strategy == critic.strategy`** alors que GRPO **ne lance pas** le worker critic : dette config / piège pour configs minimales.

---

## 10. Références (chemins absolus)

- `/home/v.lagresle/rl-gym-workout/AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py`
- `/home/v.lagresle/rl-gym-workout/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`
- `/home/v.lagresle/rl-gym-workout/AgentGym-RL/verl/utils/agentgym/client.py`
- `/home/v.lagresle/rl-gym-workout/AgentGym-RL/verl/agent_trainer/main_generation.py`
- `/home/v.lagresle/rl-gym-workout/examples/train/AgentGym-RL/textcraft_train.sh`
- `/home/v.lagresle/rl-gym-workout/AgentGym/agentenv/agentenv/envs/textcraft.py`
