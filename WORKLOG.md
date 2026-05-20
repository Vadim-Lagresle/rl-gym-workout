# Worklog — rl-gym-workout

Journal de bord pour reprendre le projet sans perdre de contexte.

> **Récap de la session 2026-05-05** (smoke tests, premier eval, gotchas
> détaillés) : voir [`docs/SESSION_2026-05-05.md`](docs/SESSION_2026-05-05.md).

## Setup (fait)

- VM GCP `vadimagent`, **A100 40 Go**, driver 580 / CUDA 13 (compatible binaries cu124).
- Submodule `AgentGym/` initialisé (`git submodule update --init --recursive`).
- Miniconda installé dans `~/miniconda3`.
- Deux envs conda **persistent sur le disque** :
  - `**agentgym-rl`** (Python 3.10) — la stack d'entraînement
  (torch 2.4 cu124, flash-attn 2.7.3, vllm dev, ray 2.55.1, transformers 4.51.3, verl, agentenv).
  - `**agentenv-textcraft`** (Python 3.10) — le serveur de jeu HTTP
  (agentenv + agentenv-textcraft + uvicorn/fastapi).
- Serveur TextCraft validé : `textcraft --host 127.0.0.1 --port 36005` démarre OK.
- `from verl import DataProto; from agentenv.envs import TextCraftEnvClient` passe sans erreur dans `agentgym-rl`.

Snapshots : `setup/requirements-agentgym-rl.txt`, `setup/requirements-agentenv-textcraft.txt`.
Script de re-création complet : `setup/setup.sh`.

## Architecture (rappel express)

Deux processus indépendants, parlent en HTTP :

```
┌───────────────────────────┐  HTTP  ┌──────────────────────────┐
│ Trainer  (agentgym-rl)    │  ───▶  │ Env server (agentenv-...)│
│   verl + vLLM + Ray       │  ◀───  │ FastAPI + uvicorn         │
│   GPU                     │        │ port 36005 par défaut    │
└───────────────────────────┘        └──────────────────────────┘
```

Endpoints REST exposés par chaque env-server :
`POST /create`, `POST /reset`, `POST /step`, `GET /observation`, `POST /close`.

Côté code :

- `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` → cœur du rollout multi-tour
- `AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py` → boucle PPO + RoundsScheduler (ScalingInter-RL)
- `AgentGym-RL/verl/utils/agentgym/client.py` → factory EnvClient
- `AgentGym/agentenv/agentenv/envs/textcraft.py` → le client HTTP TextCraft (5 méthodes : reset/observe/step/close + conversation_start)
- `AgentGym/agentenv-textcraft/agentenv_textcraft/server.py` → le FastAPI

## TODO — prochaine session

1. **Smoke test HTTP** :
  - Panneau A : `conda activate agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005`
  - Panneau B : `curl -X POST http://127.0.0.1:36005/create -d '{}' -H 'content-type: application/json'`
2. **Mini-agent Python** sans LLM — pour comprendre `EnvClient` :
  ```python
   from agentenv.envs import TextCraftEnvClient
   c = TextCraftEnvClient(env_server_base="http://127.0.0.1:36005",
                          data_len=1, timeout=60)
   c.reset(0)
   print(c.observe())
   out = c.step("Thought: I'll inventory.\n\nAction: inventory")
   print(out.state, out.reward, out.done)
  ```
3. **Télécharger** Qwen2.5-3B-Instruct (HF) :
  `huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct`
4. **Adapter** `examples/eval/textcraft_eval.sh` :
  - `model.path=models/Qwen2.5-3B-Instruct`
  - retirer le `model_merger.py` (pas de checkpoint à fusionner)
  - `agentgym.env_addr=http://127.0.0.1:36005`
5. **Lancer l'éval** : `bash examples/eval/textcraft_eval.sh` — premier rollout multi-tour, voir les rewards.
6. **Petit training** : `bash examples/train/AgentGym-RL/textcraft_train.sh`, mais
  - modèle = Qwen2.5-3B-Instruct (pas 7B, A100 40 Go),
  - `train_batch_size=8`, `rollout.n=2`, `rounds=10`.
7. **Coder mes idées** : modifier `vllm_rollout.py::generate_sequences` ou `ray_trainer.py::fit` selon le besoin.

## Pièges déjà rencontrés (à ne pas refaire)

- `flash-attn` wheel : NE PAS la renommer `flash_attn.whl` ; pip exige le nom complet.
- `transformers` : pinner à **4.51.3** ; les versions 5.x cassent verl. Le warning trl est inoffensif.
- `agentenv` exige Python ≥ 3.10 (le README de TextCraft qui dit 3.9 est obsolète).
- `Ctrl-b puis %` est une **séquence tmux**, pas une commande shell. Et `Ctrl-b ←/→` change le panneau actif.
- Avec un A100 **40 Go** : impossible de tenir Qwen-7B + ref + critic. Rester en 3B et **GRPO** (pas de critic) ; ou attendre de passer en 80 Go avant de scaler.
- **TextCraft : chemin relatif `recipes/`**. Le serveur charge ses recettes Minecraft via le chemin relatif `agentenv_textcraft/recipes/` (codé en dur dans `env_wrapper.py::TextCraft_Wrapper.__init__`). Si tu lances `textcraft` depuis n'importe où sauf `AgentGym/agentenv-textcraft/`, ça crash avec `FileNotFoundError: agentenv_textcraft/recipes/`. **Toujours faire `cd AgentGym/agentenv-textcraft/` avant `textcraft --host ... --port ...`**. (Workaround long terme : passer `minecraft_dir=<chemin absolu>` au constructeur, ou patcher l'upstream.)
- **Cursor sandbox réseau pour les serveurs locaux**. Par défaut Cursor exécute les commandes dans un sandbox avec un namespace réseau séparé : un serveur lancé là tourne sur `127.0.0.1:36005` *à l'intérieur du sandbox*, mais le port n'est PAS exposé à l'hôte. Symptôme : `ss -tlnp | grep 36005` ne retourne rien, et `curl 127.0.0.1:36005` répond `Connection refused` alors que le process est bien vivant. **Solution : lancer tout serveur d'env (TextCraft, etc.) hors sandbox** (`required_permissions: ["all"]` dans Cursor, ou via un terminal Cursor classique non sandboxé). Sinon le trainer ne pourra pas parler au serveur d'env.

## Smoke test HTTP (validé le 2026-05-05)

- Serveur lancé : `cd AgentGym/agentenv-textcraft && conda activate agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005`
- Vérif port côté hôte : `ss -tlnp | grep 36005` → doit montrer `LISTEN ... 127.0.0.1:36005 ... users:(("textcraft",pid=...))`
- Test endpoint :
  ```
  curl -s -X POST http://127.0.0.1:36005/create -H 'content-type: application/json' -d '{}' | python3 -m json.tool
  ```
- Réponse attendue : `{"id": 0, "observation": "Crafting commands:\\ncraft 1 golden carrot ...\\nGoal: craft gold ingot.", "done": false, "reward": 0}`

## Mini-cycle EnvClient sans LLM (validé le 2026-05-05)

- Script : `scratch/01_minicycle.py` (à lancer dans env conda `agentgym-rl`).
- Cycle : `TextCraftEnvClient(...)` → `reset(0)` → 3 `step` → reward=1, done=True.
- Tâche #0 du dataset = "craft gold ingot" ; résolue en `inventory → get 9 gold nugget → craft 1 gold ingot using 9 gold nugget`.
- **Lenteur cold-start : ~8min30 pour 3 calls HTTP**. Le serveur répond en ms, c'est l'import `from agentenv.envs import TextCraftEnvClient` (qui charge verl + torch + ray etc.) qui prend tout ce temps. Non bloquant mais à creuser si on veut un dev loop rapide.

## Modèle Qwen2.5-3B-Instruct téléchargé (2026-05-05)

- Commande : `huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct` (depuis env `agentgym-rl`).
- Chemin : `models/Qwen2.5-3B-Instruct/` (dossier `models/` ajouté au `.gitignore`).
- Taille : 5.8 Go (2 shards safetensors + tokenizer + configs).
- Download time : ~58 s sur la VM GCP.
- **Gotcha sandbox Cursor + DNS** : avec `required_permissions: ["full_network"]`, `huggingface.co` ne se résout PAS dans le sandbox (`Temporary failure in name resolution`). Il faut `required_permissions: ["all"]` pour que le download fonctionne (sandbox complètement désactivé).

## Dataset AgentGym-RL-Data-ID téléchargé (2026-05-05)

- Commande : `huggingface-cli download AgentGym/AgentGym-RL-Data-ID --repo-type dataset --local-dir AgentEval`
- Taille : 7.9 Mo (juste des JSONs : 100 items par tâche, format `{"item_id": "<task>_<idx>"}`).
- Structure : `AgentEval/eval/<task>_test.json` et `AgentEval/train/<task>_train.json`.

## Eval verl `main_generation.py` ÉCHOUE en single-GPU (2026-05-05)

`examples/eval/textcraft_eval.local.sh` (version patchée upstream) crash en SIGSEGV
silencieux dans `wg.init_model()`, juste après `NCCL version 2.20.5+cuda12.4`.
- Symptôme : `ray.exceptions.ActorDiedError: ... Worker exit type: SYSTEM_ERROR`,
  pas de stack trace côté Python.
- Pas un OOM (RAM/GPU libres), pas un driver mismatch (vLLM standard fonctionne).
- Cause identifiée : `verl.third_party.vllm` est un fork forké pour le training PPO
  multi-GPU avec FSDP. Avec `load_format=dummy_dtensor` (default), il init vLLM avec
  des poids random et attend qu'on patche les vrais poids depuis un DTensor FSDP.
  Pour un modèle HF brut + 1 GPU, l'init NCCL crash.
- Forcer `load_format=safetensors` ne suffit pas : le crash persiste.
- Test isolant : `scratch/02_vllm_standalone.py` charge Qwen-3B avec vLLM standard
  et génère normalement → confirme que le bug est dans le fork verl, pas dans vLLM.

**Workaround** : on évalue avec un script standalone (vLLM standard + agentenv).

## Eval Qwen-3B-Instruct sur TextCraft (2026-05-05) — script custom

- Script : `scratch/03_eval_qwen.py` (vLLM standard + `TextCraftEnvClient`, ~150 lignes).
- Smoke test 7 items : Pass@1 = **4/7 = 57%** (variance haute, on lancera les 100 ensuite).
- Logs détaillés des trajectoires : `scratch/eval_logs/textcraft_<id>.json`.
- **Observation pédagogique :**
  - Quand Qwen réussit : 2-6 tours, raisonnement clair "get → craft".
  - Quand Qwen échoue : timeout à 30 tours, **principalement à cause de violations
    du format** (plusieurs `Action:` dans une réponse → l'env rejette, Qwen ne
    corrige pas, boucle infinie). Aussi des erreurs de planification (croit qu'il
    faut craft un sub-item alors que `get` marche directement).
  - C'est exactement le terrain où le RL multi-tour aide : apprendre à respecter
    le format et à mieux planifier.

## Mode resume + run partiel sur VM stable (2026-05-05 fin de journée)

- Migration VM Spot → VM standard (A100 non-spot) faite via snapshot du disque
  persistant. La nouvelle VM hérite de tout (conda envs, modèle, dataset, logs).
- Ajout d'un **mode resume** dans `scratch/03_eval_qwen.py` : au démarrage, le
  script liste les items qui ont déjà un JSON valide dans `scratch/eval_logs/`
  et les skip. Pratique en cas d'arrêt VM ou de préemption.
  - Variable d'env `FORCE_REDO=1` pour ignorer les logs précédents et tout rerun.
  - Le tokenizer + vLLM ne sont chargés QUE s'il y a au moins 1 item à faire
    (gain de ~30 s en cas de full-resume).
- Run partiel lancé : 70 items prévus, 42 effectués avant arrêt volontaire pour
  fin de journée. Total loggués dans `scratch/eval_logs/` : **73 items**
  (30 originaux + 42 du nouveau run + 1 fichier orphelin `textcraft_420.json`
  d'un test antérieur, à ignorer pour les stats finales).
- **Pass@1 partiel sur 73 logs** : 18 succès → 24.7 %. Note : ce chiffre est
  biaisé par la difficulté décroissante (les 30 premiers item_id du dataset
  semblent plus simples que les item_id 140-180 explorés ensuite — Qwen-3B a
  enchaîné 22 timeouts d'affilée sur la fin du run).
- Reste **27 items à faire** demain sur les 100 de l'éval officielle.
- **Reprise demain matin** : VPN ON sur le Mac, démarrer la VM, reconnecter
  Cursor SSH, puis :
  1. terminal A : `cd ~/rl-gym-workout/AgentGym/agentenv-textcraft &&
     conda activate agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005`
  2. terminal B : `cd ~/rl-gym-workout && conda activate agentgym-rl &&
     python scratch/03_eval_qwen.py`
  Le mode resume va automatiquement skipper les 73 déjà faits et terminer les
  27 derniers (~25 min).

## Eval Qwen-3B FINALE sur TextCraft 100 items (2026-05-06)

Reprise sur la nouvelle VM stable (créée depuis le snapshot du 2026-05-05),
mode resume → 23 items restants exécutés ce matin. **100/100 items évalués**.

### Résultats finaux

| Métrique | Valeur |
|---|---|
| **Pass@1** | **18 / 100 = 18.0 %** |
| Avg@1 | 0.18 |
| Mean rounds (succès) | 7.2 |
| Mean rounds (échec) | 30.0 (= MAX_ROUNDS) |
| Mean duration / item | 63.4 s |
| Total cumulated rollout time | ~106 min |

### Observations qualitatives clés

- **Tous les échecs sont des timeouts à 30 tours**, pas des `done=true` avec
  reward=0. Qwen-3B-Instruct ne sait jamais "abandonner" proprement, il boucle
  jusqu'à la limite. Implication directe : **un signal de progrès intermédiaire
  (process reward) ou une pénalité de longueur pourrait beaucoup aider** pour
  un futur RL.
- **Les succès sont très rapides (mean 7.2 tours)** : quand le modèle "voit" la
  recette, il converge en quelques actions. Le problème principal est donc la
  *robustesse de planification*, pas la complexité de la solution finale.
- **Ratio succès/échec varie fortement avec l'index de l'item** : les 30
  premiers items du dataset sont à ~43 % Pass@1, les items 140-180 à ~10 %.
  La difficulté n'est pas uniforme, ce qui plaide pour un curriculum
  intelligent (apprendre sur le facile d'abord, complexifier ensuite).

### Gotchas découverts ce matin

- **Mac en veille = SSH disconnect**. Si on lance un long job (eval, training)
  et qu'on laisse le Mac en veille, la session SSH Cursor tombe. Le job
  continue côté VM (les processes sont indépendants), MAIS Cursor ne voit
  plus rien et le `cwd` revient sur le Mac local. Trois solutions possibles :
  1. Garder le Mac réveillé : `caffeinate -d` dans un terminal local pendant
     toute la session.
  2. Désactiver la mise en veille en Settings > Battery > "Prevent automatic
     sleeping when display is off" (ou équivalent).
  3. Ajouter dans `~/.ssh/config` côté Mac : `ServerAliveInterval 60` et
     `ServerAliveCountMax 30` pour que SSH ping le serveur toutes les 60 s.
  Solution la plus pragmatique : option 3 + `caffeinate -d` quand on lance
  un eval/training long.
- **`nvidia-smi` plante dans le sandbox Cursor** mais marche en `["all"]`.
  Le sandbox Cursor par défaut bloque l'accès aux device nodes `/dev/nvidia*`,
  même quand l'utilisateur est root. Pour tout ce qui touche le GPU
  (`nvidia-smi`, vLLM, training), toujours utiliser `required_permissions:
  ["all"]`. Pas un vrai problème (on le faisait déjà), juste à savoir pour
  ne pas paniquer si on voit `NVIDIA-SMI has failed to communicate with the
  driver` lors d'un check rapide.
- **Process orphan après `kill` du shell wrapper**. Hier soir, le subagent a
  kill le shell ID du run d'eval mais le process Python child (PID 4830) a
  survécu en orphan et a continué à logger ~3 min. Lesson : toujours faire
  un `ps aux | grep python.*eval` après un kill pour vérifier qu'il n'y a
  pas de zombies, et killer explicitement les PID python.

### Décisions stratégiques notées

- **AgentGym (le serveur d'env + le client) garde sa valeur** : on ne
  réimplémentera pas TextCraft. Mais **AgentGym-RL (verl + scripts) n'apporte
  pas grand-chose** dans notre contexte single-GPU + research. Pour le
  training, on partira sur **mini-PPO from-scratch** (par-dessus notre
  `scratch/03_eval_qwen.py`) ou **TRL** (HuggingFace). Détails à figer dans
  le prochain doc de session.

## Phase B — smoke test training AgentGym-RL (2026-05-06)

Objectif : vérifier si le training upstream `verl.agent_trainer.main_ppo`
marche en single-GPU (A100 40 Go) pour Qwen2.5-3B-Instruct, avant d'investir
du temps dans un vrai run.

### Config test

- Script local créé : `examples/train/AgentGym-RL/textcraft_train.local.sh`
  (copie de l'upstream, sans toucher `textcraft_train.sh` original).
- Overrides clés:
  - `trainer.n_gpus_per_node=1`, `trainer.nnodes=1`
  - `trainer.total_training_steps=1` (smoke test pur)
  - `algorithm.adv_estimator=grpo`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
  - `actor_rollout_ref.rollout.load_format=safetensors`
  - `actor_rollout_ref.model.path=/home/v.lagresle/rl-gym-workout/models/Qwen2.5-3B-Instruct`
  - `data.train_batch_size=4`, `rollout.n=2`, `rounds=20`
- Serveur TextCraft relancé sur `127.0.0.1:36005` hors sandbox.

### Résultat

- **ÉCHEC en ~36 secondes, avant le step 1.**
- Le run atteint:
  - `dataset len: 374`
  - `Size of train dataloader: 93`
  - `Total training steps: 1`
  - puis crash pendant `trainer.init_workers()` à l'init de la ref policy.
- Trace finale:
  - `ray.exceptions.ActorDiedError`
  - `Worker exit type: SYSTEM_ERROR`
  - juste après `NCCL version 2.20.5+cuda12.4`
  - stack: `trainer.init_workers() -> self.ref_policy_wg.init_model()`
    (`ray_trainer.py:607`)
- Ce pattern est **identique** au crash de l'eval upstream (`main_generation.py`)
  observé la veille : probable SIGSEGV dans le fork `verl.third_party.vllm`
  sur notre setup 1 GPU + modèle HF brut.

### Décision

- Stopper la piste \"forcer verl à marcher\" pour ne pas perdre de temps.
- **Plan B validé : TRL+GRPO** sur notre boucle multi-tour custom
  (`scratch/03_eval_qwen.py`) avec un squelette de training minimal.

## Phase C — Plan B TRL+GRPO (2026-05-06 fin d'après-midi)

Objectif : valider qu'on peut entraîner un policy gradient en single-GPU
sur TextCraft sans dépendre du fork `verl.third_party.vllm`.

### Setup technique

- Création d'un env conda dédié : `trl-grpo` (isolé de `agentgym-rl` pour
  ne pas casser les versions déjà stables du projet principal).
- Stack validée dans `trl-grpo` :
  - `torch==2.6.0+cu124`
  - `transformers==4.57.1`
  - `trl==1.3.0`
  - `peft`, `accelerate`, `datasets`
- `agentenv` installé en editable dans cet env pour réutiliser
  `TextCraftEnvClient`.

### Implémentation

- Nouveau script : `scratch/07_trl_grpo_textcraft_smoke.py`
  - charge le split train TextCraft
  - construit des prompts chat alignés avec le baseline eval
  - entraîne avec `GRPOTrainer` + LoRA (pour tenir la VRAM)
  - reward custom TextCraft
- Version finale du reward : **multi-action simulée**
  - extrait toutes les lignes `Action: ...` d'une completion
  - les exécute séquentiellement dans l'env (jusqu'à `MAX_SIM_ROUNDS=20`)
  - reward = sparse succès + shaping (valid/invalid steps, répétitions, overflow)
  - ce n'est pas encore un vrai rollout interactif turn-by-turn, mais bien
    plus proche du multi-tour qu'un reward 1-step.

### Résultats des smoke tests

- Tentative 1 : KO (`AttributeError` sur `reward_func.__name__`)
- Tentative 2 : KO (OOM optimizer)
- Tentative 3 : KO (`generation_batch_size` non divisible par `num_generations`)
- **Tentative 4 : OK**
  - run: `--max-items 16 --max-steps 1 --num-generations 2`
  - `exit_code=0`, métriques GRPO produites
- **Test plus robuste : OK**
  - run: `--max-items 32 --max-steps 3 --num-generations 2`
  - `exit_code=0`
  - `train_runtime ~17s`, 3 steps exécutés, losses/rewards/entropy loggés

### Conclusion opérationnelle

- **Plan B TRL+GRPO est viable** en single-GPU sur cette VM.
- La prochaine étape est d'implémenter un vrai rollout interactif (génération
  action par action avec feedback env entre les tours) via `rollout_func` de
  TRL, puis de lancer un training plus long (50-100 steps) avec eval périodique
  sur nos 100 items test.

### Itération suivante (2026-05-06, soirée) — test intermédiaire 20 steps

- `scratch/07_trl_grpo_textcraft_smoke.py` a été amélioré avec un reward
  **multi-action simulé** :
  - extraction de toutes les lignes `Action: ...` de la completion,
  - exécution séquentielle dans TextCraft (`MAX_SIM_ROUNDS=20`),
  - shaping léger: bonus/pénalités selon feedback env (`Could not`, `Error`,
    répétitions, dépassement de rounds).
- Run lancé :
  - env: `trl-grpo`
  - commande: `python scratch/07_trl_grpo_textcraft_smoke.py --max-items 128 --max-steps 20 --num-generations 2 --run-name trl_grpo_step20_multiaction`
  - résultat: **succès (`exit_code=0`)**
  - `train_runtime`: ~88s pour 20 steps.
- Observations :
  - Le pipeline reste stable (pas de crash Ray/NCCL, pas d'OOM).
  - La reward moyenne reste proche de 0 sur ce mini run (normal pour une
    première reward shaping encore brute + dataset difficile).
  - Un run a échoué uniquement par race condition (trainer lancé quelques
    secondes avant le serveur), puis rerun OK une fois le serveur prêt.

## Phase C bis — rollout interactif via `rollout_func` TRL (2026-05-06, soirée)

Objectif : passer du reward \"open-loop\" (parsing post-hoc) à un vrai mini-loop
interactif pendant la génération (assistant -> env.step -> observation user -> ...).

### Implémentation

- `scratch/07_trl_grpo_textcraft_smoke.py` patché :
  - ajout de `textcraft_rollout_func(prompts, trainer)` (API TRL expérimentale),
  - pour chaque sample, on crée un `TextCraftEnvClient`, reset par item, puis
    boucle de tours:
    1) rendu prompt chat,
    2) génération d'un assistant turn via `trainer._generate_single_turn(...)`,
    3) extraction de la première `Action:`,
    4) appel `env.step(...)`,
    5) ajout de l'observation comme message user.
  - les `completion_ids` et `logprobs` sont concaténés sur tous les tours et
    renvoyés au trainer pour la loss GRPO.
- La reward function lit désormais `episode_reward` et `invalid_steps` passés en
  extra fields par `rollout_func`.
- `environment_factory` TRL a été abandonné (requiert `transformers>=5.2.0`,
  incompatible avec notre stack stable 4.57.1). Gestion d'env faite directement
  dans `rollout_func`.

### Validation

- Run test:
  - `--max-items 16 --max-steps 2 --num-generations 2`
  - run name: `trl_grpo_rolloutfunc_v2_step2`
  - résultat: **succès (`exit_code=0`)**
- Performance:
  - stable, pas d'OOM ni crash,
  - mais **beaucoup plus lent** (step_time ~111s au step 1) car chaque step
    contient une vraie interaction multi-tour avec l'env.
- Signal reward:
  - `rewards/textcraft_reward/mean` positif (~0.36) sur ce test court,
  - donc le reward interactif renvoie bien un gradient exploitable.

### Run intermédiaire 10 steps (2026-05-06, soirée)

- Objectif : confirmer la stabilité de la version interactive sur un horizon
  un peu plus long avant un éventuel run 50–100 steps.
- Commande :
  - env: `trl-grpo`
  - `python scratch/07_trl_grpo_textcraft_smoke.py --max-items 64 --max-steps 10 --num-generations 2 --run-name trl_grpo_rolloutfunc_v2_step10`
- Setup :
  - serveur TextCraft relancé proprement (`textcraft --host 127.0.0.1 --port 36005`)
  - vérifié `Application startup complete` avant de lancer le trainer.
- Statut : run en cours au moment de cette mise à jour (progress bar `0/10`,
  modèle chargé, GRPOTrainer instancié sans warning bloquant).
- Hypothèse de durée : ~15-20 min (step_time observé ~111s × 10 steps + setup),
  à valider avec le `train_runtime` final.
- Prochaine action après ce run : faire une eval de contrôle avec
  `scratch/03_eval_qwen.py` sur quelques items pour voir si le Pass@1 bouge
  vs la baseline 18%.

## Notes méthodologiques à traiter (2026-05-11)

Trois points identifiés en revue de méthode pendant que le run 50 steps tourne.
À traiter une fois ce run terminé pour ne pas brouiller la comparaison
baseline / step10 / step50.

### 1. Bug de shaping dans `textcraft_reward` (à fixer après step50)

`scratch/07_trl_grpo_textcraft_smoke.py::textcraft_reward` calcule `n_actions`
via `count_actions` sur la completion **entière concaténée** (tous les tours
collés). Du coup le bonus `+0.02` n'est attribué que si la completion totale a
exactement 1 ligne `Action:`, et le malus `-0.05` est appliqué dès qu'il y a
>1 action (c'est-à-dire dès qu'il y a un épisode multi-tour). L'intention
initiale était "1 action par message d'assistant", pas "1 action sur tout
l'épisode" — c'est un bug d'implémentation.

Conséquence observée sur le run 10 steps : le modèle a appris à **tenter en 1
tour ou abandonner**, parce que c'est ce que le shaping récompense. Trajectoire
de reward : −0.19 (épisodes longs perdants) → +0.95 (1-tour gagnants) → +0.345
(rechute multi-tour) → +0.95 (1-tour). Sur le test set, step10 attrape 8 items
que le baseline ratait (items triviaux résolubles en 1 action) mais en perd 8
autres (items complexes qu'il aurait dû résoudre en multi-tour). Cas d'école
de reward hacking, parfaitement cohérent avec le shaping.

Fix prévu : porter le décompte `n_actions` au niveau du tour, en remontant
l'info depuis `rollout_func` (qui voit chaque tour individuellement) plutôt
qu'en parsant la completion concaténée. Appliquer le shaping sur la moyenne
des n_actions par tour. Lancer ensuite un step50_v2 avec ce fix pour mesurer
l'impact net du shaping correct vs le shaping cassé.

### 2. Monter `num_generations` (N) — nécessite un GPU plus gros

Actuellement N=2 dans notre setup, choix imposé par la VRAM A100 40 Go (Qwen-3B
+ LoRA + référence + buffers GRPO + rollout multi-tour saturent déjà). Le
papier AgentGym-RL utilise **N=8** (cf. `examples/train/AgentGym-RL/textcraft_train.sh`
ligne 24, `rollout_sample_num=8`), le papier DeepSeek-GRPO va jusqu'à N=64
sur certaines tâches.

N=2 est le strict minimum pour que GRPO ait du sens (sinon avantage = 0). Avec
N=2, le signal effectif est bruité : quand les 2 completions donnent le même
reward (cas fréquent), `frac_reward_zero_std=1` et le step n'apprend rien.

À faire dès qu'on a accès à un GPU avec ≥80 Go de VRAM (A100 80 Go, H100,
ou idéalement **B200** pour avoir aussi plus de débit) : remonter N à 4 ou 8,
réviser `max_completion_length`, augmenter `per_device_train_batch_size`. C'est
la première amélioration structurelle à faire — plus impactante que de toucher
au reward shaping ou au curriculum.

**Mesure réelle pendant le run 50 steps (2026-05-11)** : à N=2 on consomme
37,4 Go / 40 Go (91 % de pleine charge), il reste ~3 Go libres. Donc N=3
aurait été faisable d'entrée (j'ai été trop conservateur sur le run 10 steps),
N=4 risqué, N=8 infaisable sans toucher autre chose. Si on doit rester sur
l'A100 40 Go pour le prochain run, deux leviers pour grimper N : (a) baisser
`max_completion_length` de 128 à 96 (~25 % d'économie d'activations par
completion, ouvre N=4), (b) baisser `MAX_SIM_ROUNDS` de 20 à 12 (économie de
KV cache pendant la génération, peut permettre N=6 combiné au point a). Mais
ces compromis dégradent la qualité du training. Idéalement on saute à N=8 sur
un GPU plus gros.

### 3. Levier "syntax normalizer" en amont de TextCraft

TextCraft a un parser **strictement rule-based regex** (3 verbes seulement :
`craft X using Y`, `get N item`, `inventory`, cf. `agentenv_textcraft/environment.py`
lignes 12-16). Aucune tolérance, aucune normalisation lexicale, aucun
LLM-as-judge. Donc le modèle doit apprendre **deux choses indépendantes** en
même temps : (a) la stratégie de craft, (b) la grammaire syntaxique stricte
attendue par le serveur. 25 % des échecs baseline sont uniquement des erreurs
de format (cf. `scratch/04_analyze_eval.py`), pas des erreurs de stratégie.

Idée à explorer (curriculum) : insérer un **petit LLM SFT-é uniquement à la
normalisation syntaxique** (par ex. un Qwen-0.5B ou 1.5B fine-tuné sur quelques
milliers de paires "phrase naturelle → action TextCraft valide") entre l'agent
principal et le serveur. L'agent principal travaille en langage plus naturel
("get three oak logs", "build a crafting table"), le normalizer traduit en
`get 3 oak_log` / `craft 1 crafting_table using 4 oak_planks`, et seul ça est
envoyé au serveur. Bénéfices attendus :

- découple la difficulté syntaxique de la difficulté stratégique pendant le
  RL training ;
- permet d'utiliser le baseline directement comme stratège sans qu'il soit
  pénalisé sur la syntaxe ;
- fournit une comparaison contrôlée "stratège seul (GRPO sur Qwen-3B normal)"
  vs "stratège + normalizer".

Coût : un petit SFT à part (donc une étape supplémentaire dans le pipeline)
et une dépendance d'inférence supplémentaire pendant le rollout. À discuter
avec les encadrants — c'est typiquement le genre d'idée qui peut donner un
résultat de papier à elle seule si on la prend au sérieux.

**Critique honnête après revue (2026-05-11)** : sur TextCraft précisément, l'idée
est probablement disproportionnée. La grammaire ne fait que 3 verbes regex, le
baseline produit déjà ~75 % d'actions bien formées, et quand je creuse les 25 %
d'échecs "de format", ce sont en réalité (a) des erreurs de **protocole**
(plusieurs `Action:` par message), (b) des erreurs de **nomenclature** (`planks`
au lieu de `oak_planks` — connaissance des items, pas grammaire), (c) des erreurs
**stratégiques masquées** (`craft house using 3 wood` — syntaxiquement valide
mais "house" n'existe pas dans le crafting tree). Un normalizer ne traite vraiment
que (a) et une partie de (b). Le pattern reste pertinent comme angle de recherche
sur un env plus complexe (SQL, APIs réelles), pas ici. Garder l'idée en piste
exploratoire long terme.

### 4. Trois alternatives plus directes au "syntax normalizer" pour TextCraft

À traiter dans cet ordre de priorité avant de revenir au normalizer.

**4.a — Fix du reward shaping (point 1 ci-dessus)**. Le bug actuel pénalise
activement le bon comportement multi-tour. C'est le levier le plus gros à coût
zéro. À faire dès que le run en cours est fini.

**4.b — Post-processing rule-based de l'action côté client**. Avant d'envoyer
le texte de l'agent au serveur TextCraft, intercaler une fonction Python d'une
trentaine de lignes : garder uniquement la première ligne `Action:`, lowercase,
normaliser espaces et underscores, mapper quelques alias connus (`wood` →
`oak_log`, `pickaxe` → `wooden_pickaxe` par défaut, etc.). Pas de LLM, pas de
SFT, pas de dépendance d'inférence. Devrait absorber 60-70 % des erreurs de
format identifiées par `scratch/04_analyze_eval.py`. À implémenter dans
`scratch/07_trl_grpo_textcraft_smoke.py::textcraft_rollout_func` (entre
`first_action_or_empty` et `env.step`) et dans `scratch/03_eval_qwen.py` /
`scratch/08_eval_qwen_lora.py` pour avoir le même normalisateur à l'eval.

**4.c — LoRA "format" séparé du LoRA "stratégie"**. Si 4.b ne suffit pas,
entraîner un petit LoRA dédié en SFT pur sur des paires (mauvais format → bon
format) extraites de nos propres logs d'échec baseline. À l'inférence, stacker
ce LoRA-format avec le LoRA-stratégie (peft supporte le stacking nativement).
~20 Mo d'adaptateurs en plus, une seule passe d'inférence, reste dans
l'écosystème qu'on utilise déjà. Beaucoup plus léger qu'un deuxième LLM
complet, et le decoupling "format vs stratégie" est le même que celui du
normalizer.

## Phase D — 50-step training run + eval (2026-05-11)

Premier run "long" de TRL+GRPO sur TextCraft : passage de 10 → 50 steps avec
un dataset un peu plus diversifié (256 items au lieu de 64), même
configuration que step10 par ailleurs (N=2, LoRA r=16, max_completion 128,
shaping inchangé).

### Commandes exactes

Phase 1 (terminer step10 sur les 100 items) :

```
# panneau A
conda activate agentenv-textcraft
cd /home/v.lagresle/rl-gym-workout/AgentGym/agentenv-textcraft
textcraft --host 127.0.0.1 --port 36005

# panneau B
conda activate agentgym-rl
cd /home/v.lagresle/rl-gym-workout
LORA_PATH=saves/trl_grpo/trl_grpo_rolloutfunc_v2_step10/checkpoint-10 \
  EVAL_TAG=step10 python scratch/08_eval_qwen_lora.py
```

Phase 2 (training step50) :

```
conda activate trl-grpo
python scratch/07_trl_grpo_textcraft_smoke.py \
  --max-items 256 --max-steps 50 --num-generations 2 \
  --run-name trl_grpo_rolloutfunc_v2_step50
```

Phase 3 (eval step50) :

```
conda activate agentgym-rl
LORA_PATH=saves/trl_grpo/trl_grpo_rolloutfunc_v2_step50/checkpoint-50 \
  EVAL_TAG=step50 python scratch/08_eval_qwen_lora.py
```

### Training metrics

- `train_runtime` : **4032.59 s ≈ 67.2 min** (ETA estimé 45–50 min, réel
  +35 % — `step_time` rollout entre 120 s et 220 s, plus variable que prévu).
- `train_loss` final : 0.0430.
- 50/50 steps complétés, exit_code=0, checkpoints `40`, `45`, `50` sauvegardés.

### Trajectoire reward (`rewards/textcraft_reward/mean`)

Logs émis sur les steps de rollout (impair). 25 valeurs sur 50 steps :

| step | reward  | step | reward  | step | reward  | step | reward  | step | reward  |
|-----:|--------:|-----:|--------:|-----:|--------:|-----:|--------:|-----:|--------:|
| 1    | -0.190  | 11   | -0.170  | 21   | +0.365  | 31   | -0.205  | 41   | -0.195  |
| 3    | -0.245  | 13   | -0.180  | 23   | **+0.950** | 33 | +0.350 | 43 | +0.350  |
| 5    | -0.180  | 15   | -0.195  | 25   | -0.165  | 35   | -0.205  | 45   | -0.200  |
| 7    | -0.205  | 17   | +0.385  | 27   | -0.215  | 37   | -0.155  | 47   | -0.185  |
| 9    | -0.230  | 19   | -0.245  | 29   | -0.175  | 39   | -0.205  | 49   | -0.195  |

- Premiers 5 (steps 1–9) : moyenne ≈ **−0.21**.
- Milieu 5 (steps 21–29) : moyenne ≈ **+0.23** (tirée par les spikes 17/21/23).
- Derniers 5 (steps 41–49) : moyenne ≈ **−0.19**.

Lecture : **trajectoire oscillante, pas de tendance ascendante**. Les rares
batches "gagnants" (4/25 logs avec reward > 0) coïncident avec des completions
courtes (< 100–700 tokens) où le modèle réussit en 1–2 tours, exactement le
mode collapse "1-tour ou rien" identifié dans la note méthodo §1
(reward-shaping bug). Le run 50 steps n'a donc rien fait d'autre que renforcer
ce reward hacking déjà observé à step10, sans permettre de progrès net.

### Anomalie notée pendant le training

- **Step 35–36 : `grad_norm` = 1765.89** (vs typique 0.5–4.5) sur la passe
  gradient (loss=0.4823 quand même finie, pas de NaN, pas d'OOM, training
  continue). Pic isolé (grad_norm reste sain ensuite : 1.2 / 4.6 / 1.3 / 6.3
  sur les steps suivants). Probablement déclenché par le batch très favorable
  qui précède (step 33, reward +0.350 avec completions de 109 à 1405 tokens
  → variance énorme sur les advantages). Pas critique sur ce run, mais à
  surveiller : si on monte les hparams (N=8, lr×2…), ce genre de spike peut
  diverger franchement. Un `max_grad_norm` explicite (clipping) serait
  prudent pour le run d'après.
- Pas d'autre warning bloquant. Pas de timeout HTTP serveur. Pas de problème
  GPU (mémoire stable à ~37 Go/40).

### 3-way comparison Pass@1 sur les 100 items du test set

n=100 (intersection complète, on a les 100 logs pour les trois variantes).

| variante     | Pass@1   | delta vs base | delta vs step10 |
|--------------|---------:|--------------:|----------------:|
| baseline     | 18/100   | —             | —               |
| LoRA step10  | 18/100   | **+0**        | —               |
| LoRA step50  | 14/100   | **−4**        | **−4**          |

Triple confusion (baseline pass, step10 pass, step50 pass) :

| (B, S10, S50) | count |
|:-------------:|------:|
| (T, T, T)     | 4     |
| (T, T, F)     | 6     |
| (T, F, T)     | 4     |
| (T, F, F)     | 4     |
| (F, T, T)     | 3     |
| (F, T, F)     | 5     |
| (F, F, T)     | 3     |
| (F, F, F)     | 71    |

- Items résolus par les **trois** : 4 (les "vraiment faciles", 1-tour évidents).
- Items résolus uniquement par baseline : 4 (perdus par les deux LoRA).
- Items résolus uniquement par step50 : 3 (gain net mais petit).
- Items résolus par step50 mais pas par step10 : 7 (=4+3) ; items résolus
  par step10 mais pas par step50 : 11 (=6+5). step50 a donc un **shift net
  négatif de −4** par rapport à step10.

### Conclusion

**step50 dégrade Pass@1 vs baseline (−4 pts) et vs step10 (−4 pts) ; le
training a renforcé le reward hacking "1-tour ou rien" sans converger vers
une stratégie multi-tour gagnante. Prochaine étape obligatoire : fixer le
shaping (note méthodo §1) avant de relancer un run plus long.**

## Run GRPO v3 — résultats (2026-05-12)

Deuxième run "long" (50 steps), même setup que v2 sauf **fix du reward
shaping** (note méthodo §1) : `n_actions` est désormais comptée **par tour
d'assistant** dans `rollout_func` et le shaping +0.02 / −0.05 est appliqué
sur la moyenne par tour, plus sur la concaténation de toute la completion.
Reste identique : N=2, LoRA r=16, max_completion=128, MAX_SIM_ROUNDS=20,
256 items, `max_grad_norm=1.0` (clip explicite ajouté en réaction au spike
1765 de v2).

Eval LoRA step50 v3 lancée hier soir avait été interrompue à 9/100 ;
reprise ce matin en mode resume (cache détecté, 91 items neufs en
~39 min sur 1×A100 40 Go). Serveur TextCraft relancé sur 127.0.0.1:36005
au préalable.

### Tableau Pass@1 4-way (n=100, intersection complète)

| variante     | Pass@1   | mean rounds | delta vs base | delta vs v2 |
|--------------|---------:|------------:|--------------:|------------:|
| baseline     | 18/100   | 25.9        | —             | —           |
| LoRA step10  | 18/100   | 25.7        | **+0**        | —           |
| LoRA step50 v2 | 14/100 | 26.6        | **−4**        | —           |
| LoRA step50 v3 | **8/100** | 27.8     | **−10**       | **−6**      |

### Confusion 4-way (B, S10, v2, v3)

Buckets non vides triés par "succès cumulés" :

| (B, S10, v2, v3) | count |
|:----------------:|------:|
| (T, T, T, T)     | 2     |
| (T, F, T, T)     | 2     |
| (T, T, F, T)     | 2     |
| (T, T, T, F)     | 2     |
| (F, F, T, T)     | 1     |
| (F, T, T, F)     | 3     |
| (T, F, T, F)     | 2     |
| (T, T, F, F)     | 4     |
| (F, F, F, T)     | 1     |
| (F, F, T, F)     | 2     |
| (F, T, F, F)     | 5     |
| (T, F, F, F)     | 4     |
| (F, F, F, F)     | 70    |

Lectures clés :
- Résolus par v3 uniquement (gagnés par v3 et perdus par v2) : **3** items
  (`textcraft_1`, `textcraft_5`, `textcraft_8`).
- Résolus par v2 uniquement (perdus par v3) : **9** items
  (`textcraft_0`, `textcraft_3`, `textcraft_7`, `textcraft_11`,
  `textcraft_12`, `textcraft_15`, `textcraft_21`, `textcraft_29`,
  `textcraft_435`).
- Résolus par v2 **et** v3 : 5.
- Résolus par v3 mais pas par baseline : 1 (`textcraft_2`).
- Net shift v3 vs v2 : **−6**. v3 perd plus d'items qu'il n'en gagne.
- 70 items "noyau dur" perdus par les 4 variantes (probablement
  "vraiment difficiles" ou mauvaise nomenclature serveur).

### Comparaison training v3 vs v2

Source : `saves/trl_grpo/trl_grpo_rolloutfunc_v3_step50/checkpoint-50/trainer_state.json`
(le terminal log du training v3 a tourné dans une session précédente et
n'est plus dans `terminals/` ; les métriques fines `mean_n_actions_per_turn`
ne sont donc pas extraites côté trainer state — limitation honnête).

#### Reward trajectory (`rewards/textcraft_reward/mean`, 25 rollouts)

| step | reward  | step | reward  | step | reward  | step | reward  | step | reward  |
|-----:|--------:|-----:|--------:|-----:|--------:|-----:|--------:|-----:|--------:|
| 1    | −0.120  | 11   | −0.125  | 21   | **+1.020** | 31 | −0.150  | 41   | −0.105  |
| 3    | −0.185  | 13   | −0.215  | 23   | +0.420  | 33   | +0.480  | 43   | −0.115  |
| 5    | −0.110  | 15   | +0.395  | 25   | −0.125  | 35   | −0.105  | 45   | −0.125  |
| 7    | −0.145  | 17   | −0.105  | 27   | −0.105  | 37   | −0.150  | 47   | −0.110  |
| 9    | −0.160  | 19   | −0.120  | 29   | −0.165  | 39   | −0.135  | 49   | +0.415  |

- Premiers 5 (1–9) : moyenne **−0.144** (vs v2 : −0.21, donc démarrage
  légèrement moins défavorable).
- Milieu 5 (21–29) : moyenne **+0.209** (vs v2 : +0.23, à peu près identique,
  tiré par le spike step 21).
- Derniers 5 (41–49) : moyenne **−0.008** (vs v2 : −0.19, donc fin nettement
  moins négative).

5/25 batches avec reward > 0 (vs 4/25 en v2). Trajectoire toujours
**oscillante, pas d'apprentissage net**, mais moins négative que v2 en fin
de run.

#### Pattern "completion courte → reward élevé" : a-t-il disparu ?

**Non.** Smoking gun : **step 21**, `completions/mean_length = 98.5`
(min 98, max 99), reward = **+1.02**, `frac_reward_zero_std = 1.0` (les 2
rollouts ont produit exactement la même réponse ultra-courte et ont
résolu le problème). C'est la signature comportementale du mode "1-tour
ou rien" identifiée sur v2. Idem step 23 (min 152), step 33 (min 102),
step 49 (min 599) : à chaque fois qu'un batch contient au moins un
rollout court qui gagne, le reward du batch monte. Les 20 batches avec
completions ~1000–1600 tokens restent tous négatifs (reward ≈ −0.11 à
−0.22).

Le fix de shaping n'a donc **pas supprimé l'incitation à gagner en 1 tour** :
il a juste retiré le malus pour les épisodes multi-tour, mais le bonus du
gain reste tellement plus gros que le coût d'un tour long que la politique
préfère toujours tenter court.

#### `completions/mean_length` v3

- Moyenne sur les 25 rollouts : **≈ 1148 tokens** (similaire à v2).
- Min : **98** (step 21, suspect), max : **2560** (step 3, clip).
- Range typique sans outliers : 977–1649 tokens. Distribution un peu plus
  resserrée que v2 (variance des moyennes plus faible), mais avec les
  mêmes effondrements occasionnels vers les completions courtes.

#### `grad_norm` v3

- Range : 0.0 (steps 21–22, frac_reward_zero_std=1, advantage nul) à
  **19.19** (step 28, max).
- Médiane ≈ 1.5, P95 ≈ 8.
- **Aucun spike pathologique** comparable au 1765 observé sur v2/step 35–36 :
  le `max_grad_norm=1.0` ajouté en réaction joue son rôle. Le pic step 28
  reste contenu et n'a pas dégénéré.

Verdict training : **stabilité numérique propre, optimisation visible mais
non productive.** Le clip protège, mais le signal de reward conduit vers
le même attracteur que v2 — résultat empirique : Pass@1 baisse encore.

### Verdict honnête

**Le fix de shaping n'a pas amélioré le Pass@1 ; il l'a au contraire
dégradé** (8/100 vs 14/100 sur v2, soit −6 absolus, −10 vs baseline).
Le reward hacking "1-tour ou rien" n'a pas disparu — il est même plus
visible dans le training v3 (step 21 : completion de 98 tokens avec
reward +1.02 et écart-type nul, ce qui n'apparaissait pas aussi
proprement sur v2). Le malus multi-tour retiré ne suffit pas : tant que
le bonus de gain en 1 tour reste structurellement plus grand que le coût
d'un long épisode, la politique préfère parier court. Les mean rounds
montent légèrement (27.8 vs 26.6) mais sans gain : v3 essaie un peu plus
de multi-tour, échoue plus souvent, et ne réussit pas plus.

**Recommandation pour la suite** : ne pas continuer GRPO sur ce shaping.
Deux options à explorer avant de relancer un run :

1. **Refonte du reward signal** : (a) plancher du shaping bien plus négatif
   sur les rollouts courts non aboutis pour casser le pari "1-tour ou
   rien", (b) bonus par tour utile (action qui change l'inventaire) plutôt
   que bonus de format. Coût zéro, peut être testé en 1 step10.
2. **Levier 4.b du WORKLOG** : post-processing rule-based de l'action côté
   client. Absorbe les erreurs de format pures (60–70 % attendu) et ne
   touche pas au RL. Si le baseline + post-processing dépasse déjà 18/100,
   on a une meilleure base de départ pour le RL.

**À éviter à coût égal** : monter N à 4 (déjà identifié comme prio §2,
mais le problème actuel n'est pas la variance de l'avantage — c'est le
signal de reward lui-même qui est mal aligné). Inutile d'investir du GPU
sur un signal cassé.

## Note méthodo §2 — Implémenter du CoT long-form / self-verification sur TextCraft (rédaction 2026-05-12)

### Contexte

AgentGym-RL utilise du ReAct "shallow" : `<think>` court (50-200 tokens),
pas de self-verification, pas de backtracking. C'est volontaire (cf.
discussion lecture papier — argument marketing : "no need for explicit
long-reasoning"). Reasoning models style R1 / QwQ / o1 / Qwen3-thinking
font l'inverse : `<think>` long (1000-30000+ tokens), self-checking
émergent, backtracking, exploration de branches.

**Question pratique** : sur Qwen2.5-3B-Instruct + TextCraft, est-ce qu'on
peut récupérer une partie de ces gains, et à quel coût ?

**Constat préalable** : TextCraft est un environnement de **planification**,
pas de raisonnement profond. Pas de fact-checking à faire, pas d'ambiguïté
inférentielle. La difficulté est l'**arbre de dépendances de craft**
(profondeur 1 à 4) à dérouler dans le bon ordre, en gérant l'inventaire
et les substitutions d'ingrédients génériques. Donc :
- **Long CoT planning au début d'épisode** : potentiellement très utile
  (pré-construire l'arbre de craft inversé en 200-500 tokens).
- **Self-verification après chaque action** : limitée — pas grand-chose à
  vérifier autre que l'inventory check qui existe déjà comme action.
- **Backtracking** : utile quand le modèle réalise qu'il manque un
  ingrédient ou qu'une recette ne s'applique pas.

Conclusion : on n'a pas besoin de toute la machinerie R1. Un long-form
**planning structuré au tour 1**, puis ReAct shallow ensuite, devrait
capturer 80 % des gains attendus.

### 4 niveaux d'effort possibles

#### Niveau 1 — Prompt engineering (effort : 1 h, risque : nul)

Principe : modifier le system prompt TextCraft pour forcer un plan
structuré au début, avec format dédié.

Implémentation concrète :

```
You are given crafting recipes to craft items in Minecraft.

Before your first action, you MUST output a complete crafting plan in this
exact format:

<plan>
Goal: <target item>
Recipe tree (in execution order):
  1. craft <intermediate_1> using <inputs_1>
  2. craft <intermediate_2> using <inputs_2>
  ...
  N. craft <goal> using <inputs_N>
Raw materials needed: <list from environment>
</plan>

Then execute the plan one step at a time. After each successful step,
output:
<check>step <i> done, inventory now contains <items></check>

If a step fails, output:
<replan>reason: ..., new plan: ...</replan>
and emit a new <plan>...</plan>.

Use the format:
Thought: <short reasoning>
Action: <one of: get X | inventory | craft X using Y>
```

Touches :
- `scratch/03_eval_qwen.py` → ajouter `SYSTEM_PROMPT_V2` constant et un
  flag `USE_LONG_COT` lu depuis env var.
- Rien à toucher côté serveur TextCraft (le shape des actions reste le
  même : `get` / `inventory` / `craft`).

Coût compute : zéro entraînement, juste rallonger un peu les rollouts en
inférence (~+300 tokens par épisode en moyenne).

Gain attendu : **+5 à +15 points Pass@1** sur baseline 18/100. Le modèle
Qwen-3B-Instruct devrait imiter le format spontanément vu qu'il est
instruct-tuned. Les depth 1 et 2 ne devraient pas régresser (tâches
triviales), les gains viendraient principalement de depth 3-4 où la
pré-planification absorbe les erreurs de séquencage.

Signal à observer pour valider :
- `mean_rounds` baisse (les épisodes échouent moins par dérive multi-tour).
- Distribution `completion_length` bi-modale (tour 1 long ~500 tokens,
  tours suivants courts ~80 tokens).
- Pass@1 sur depth 3-4 spécifiquement (extraire la profondeur du dataset
  côté serveur — à vérifier si exposée dans l'observation).

Risque : Qwen-3B-Instruct peut ne pas suivre le format strictement
(plans incomplets, replanning manqué). À mitiger par 1-2 few-shot examples
inline dans le system prompt.

**Décision** : à tester en priorité avant tout RL supplémentaire. Coût
zéro, baseline mieux placée si gain.

#### Niveau 2 — Reward shaping pendant le RL (effort : 1-2 j, risque : moyen)

Principe : reprendre v3 du run GRPO et ajouter des composantes au reward
qui encouragent le format long-CoT défini en Niveau 1.

Implémentation concrète :

```python
# Dans rollout_func.compute_reward(completion, observation, ...)

def textcraft_reward_v4(completion, env_reward, ...):
    base = env_reward  # 0 ou 1 du serveur

    # Bonus format plan complet au tour 1
    has_plan = re.search(r"<plan>.*</plan>", completion[:2000], re.DOTALL)
    plan_bonus = 0.1 if has_plan else 0.0

    # Bonus : le plan mentionne tous les sub-items de l'arbre
    if has_plan:
        plan_text = has_plan.group()
        n_subgoals_mentioned = count_subgoals_in_plan(plan_text)
        plan_quality_bonus = min(0.05 * n_subgoals_mentioned, 0.15)
    else:
        plan_quality_bonus = 0.0

    # Pénalité format error (action mal formée)
    n_format_errors = count_format_errors(completion)
    format_penalty = -0.05 * n_format_errors

    # Pénalité "1-tour ou rien" (déjà v3) : retirée, on encourage plutôt
    # le multi-tour structuré via les bonus ci-dessus

    return base + plan_bonus + plan_quality_bonus + format_penalty
```

Touches :
- `AgentGym-RL/verl/utils/agentgym/rollout_func.py` ou équivalent
  (selon le wrapper actuel) → nouvelle fonction `compute_reward_v4`.
- Hyperparam `reward_version=v4` à ajouter au config YAML.

Coût compute : un run GRPO de 50 steps comme v2/v3, donc même ordre
de grandeur (~3 h sur A100 40 Go avec LoRA).

Gain attendu : si Niveau 1 marche déjà bien, Niveau 2 devrait
**consolider et étendre** : le modèle apprend à produire le format
**fiablement** (pas juste quand le prompt l'exige bien), et peut le
généraliser à des tâches difficiles où il aurait dévié sinon.

Risque principal : **reward hacking sur la longueur du plan**. Le modèle
peut spammer des steps inutiles dans `<plan>` pour grappiller le bonus.
À mitiger par :
- Cap dur sur la longueur du `<plan>` (genre `min(0.05 * n, 0.15)`).
- Bonus conditionné à la **présence ultérieure** de chaque sub-goal dans
  les actions effectives (sinon le plan est juste du décor).

Signal à observer :
- Évolution de `frac_with_plan` au cours des steps (devrait monter de
  ~20-40 % à 80-95 %).
- Pas de divergence : `completions/mean_length` doit rester < 1500
  tokens par épisode (sinon reward hacking sur longueur).
- Pass@1 step50 > Pass@1 step10 (= preuve que le RL apprend réellement,
  contrairement à v2/v3 où on a régression).

**Décision** : si Niveau 1 valide >+5 pts, lancer Niveau 2. Sinon, le
signal est trop faible, retravailler le system prompt d'abord.

#### Niveau 3 — Cold-start SFT puis RL (effort : 1 semaine, risque : faible)

Principe : générer un dataset de trajectoires TextCraft avec long-CoT
produites par un gros modèle, SFT Qwen-3B dessus, puis RL.

Implémentation concrète :

1. **Génération du dataset SFT** :
   - 200-500 tâches TextCraft (depth 1-4 stratifié).
   - Solveur : Claude-3.7-Sonnet avec thinking activé, ou GPT-5 thinking,
     ou DeepSeek-R1 via API. Coût estimé : ~$50-100.
   - Format imposé : `<plan>...</plan>` puis `Thought: / Action:` à chaque
     tour, avec `<check>` à la fin de chaque step réussie.
   - Filtrage : ne garder que les trajectoires qui résolvent l'épisode
     (env_reward = 1). Probablement ~70-80 % des trajectoires retenues
     vu la simplicité de TextCraft pour ces modèles.

2. **SFT Qwen-3B-Instruct** :
   - 1-2 epochs sur ces 200-500 trajectoires.
   - LR 1e-5, batch 4, gradient accumulation 4.
   - Loss masking : ne calculer la loss que sur les tokens de l'assistant
     (`<plan>`, `Thought:`, `Action:`, `<check>`). Les `Observation:`
     du serveur sont masquées.
   - Sortie : `models/Qwen2.5-3B-textcraft-cot-sft/`.
   - Coût compute : ~1-2 h sur A100 40 Go.

3. **RL par-dessus** :
   - Reprendre GRPO v3, mais en partant du checkpoint SFT au lieu du
     vanilla instruct.
   - Reward identique au v3 (ou v4 du Niveau 2).
   - 50-100 steps.

Touches :
- Nouveau script `scratch/04_generate_cot_trajectories.py` qui pilote
  l'API Claude/GPT/R1 sur les tâches TextCraft via le serveur local.
- Nouveau script `scratch/05_sft_textcraft.py` (TRL `SFTTrainer` avec
  masking).
- Reprise de la pipeline RL existante avec `model.path = saves/sft/...`.

Gain attendu : **+15 à +30 pts Pass@1** sur baseline si tout converge.
C'est l'approche la plus "puissante" mais aussi celle où on perd la
lisibilité scientifique : on ne sait plus si les gains viennent du
prompt, du SFT, ou du RL.

Risque : **distribution shift entre SFT et RL** (le modèle apprend des
patterns que le serveur réel ne récompense pas exactement de la même
façon que le solveur expert). Atténuable en générant les trajectoires
SFT **avec le vrai serveur TextCraft local** (pas via simulation).

**Décision** : à faire uniquement si Niveaux 1 et 2 plafonnent. Ordre
de grandeur de gain probablement marginal vs Niveau 2 sur TextCraft
spécifiquement (rendements décroissants), mais investissement
intéressant si on veut publier ou généraliser à d'autres envs.

#### Niveau 4 — Vraie self-verification émergente via RL pur (effort : plusieurs semaines, risque : élevé)

Principe : laisser émerger les comportements R1-like (self-check,
backtracking) via RL avec rewards bien calibrés, à la R1-Zero.

Évaluation : **probablement overkill pour TextCraft seul**. Les
comportements émergents R1 ont été observés sur des problèmes math /
code où la chaîne de raisonnement est longue et vérifiable par étapes
intermédiaires. TextCraft a des actions atomiques courtes, peu d'espace
pour de la self-verification non-triviale.

Décision : **ne pas faire** sur TextCraft. Si on voulait explorer R1-Zero
style, il faudrait étendre AgentGym-RL à un environnement plus
raisonnement-bound (Deep Search, ScienceWorld, ou un domaine custom).

### Plan d'attaque recommandé

Ordre d'exécution séquentielle, chaque étape conditionne la suivante :

1. **Niveau 1 d'abord** (1 h de travail, pas de GPU). Si Pass@1 baseline
   18/100 → 23+/100, on a déjà battu tous nos runs RL précédents
   gratuitement. Documente honnêtement le gain dans le worklog.
2. **Si Niveau 1 marche** : Niveau 2 (un run RL de 50 steps avec le
   reward v4). Objectif : transformer le gain prompt-based en gain
   model-based qui survit même sans prompt élaboré.
3. **Si Niveau 2 marche mais plafonne** : Niveau 3 (cold-start SFT).
   Coût plus élevé mais marge plus grande.
4. **Niveau 4 jamais** sur TextCraft seul.

### Méta-question : on a quand même la base instruct comme point de départ

Une remarque honnête à garder en tête : si on fait Niveau 3 en partant
d'un modèle plus capable (genre R1-Distill-Qwen-7B au lieu de
Qwen-3B-Instruct), on ne saura **plus** isoler la contribution du RL
vs celle du prior reasoning. Pour avoir une science propre, il faudrait
les 4 cases du 2×2 :

|                         | Sans RL                  | Avec RL                  |
|-------------------------|--------------------------|--------------------------|
| Qwen-3B-Instruct        | (1) baseline 18/100      | (2) v2/v3 ≤ 14/100       |
| R1-Distill-Qwen-7B      | (3) à mesurer            | (4) à mesurer            |

La cellule (3) seule est intéressante : elle dit ce qu'un modèle déjà
"reasoning" apporte sur TextCraft **sans aucun training spécifique**.
Si (3) > (2), ça met directement en question l'intérêt du RL agentique
sur des petits modèles non-reasoning, et oriente vers Niveau 3 avec un
backbone reasoning. Mesure peu coûteuse à faire (juste un eval avec
`scratch/03_eval_qwen.py` adapté pour charger R1-Distill).

**À planifier** comme expérience de side : avant de relancer du RL,
faire l'éval (3) seule pour avoir le bon repère.


---

## 2026-05-12 — Audit verl + smoke tests avant migration 8× A100

But : avant de killer la VM et provisionner une 8× A100 40 Go pour
répliquer la recette papier avec verl tel quel, vérifier que (a) la
math/algo de verl est correcte, et (b) qu'on n'a pas raté un quick win
qui permettrait de le faire tourner sur 1 GPU.

### Smoke (a) — `scratch/smoke_verl_test.py` (offline, CPU, ~12 s)

Trois unit tests sur les briques critiques de verl. **3/3 PASS.**

1. `compute_grpo_outcome_advantage` (GRPO advantage math)
   - Prompt avec rewards `[1, 0, 1, 0]` → advantages normalisés à
     `±0.866` ; `returns == advantages`.
   - Prompt avec rewards tous égaux → advantages à 0 (pas de signal,
     comportement attendu).

2. `RolloutHandler.add_assistant_message` (token-level loss_mask sur
   Qwen ChatML)
   - `input_ids`, `attention_mask`, `position_ids`, `loss_mask` même
     longueur.
   - Prompt initial : `loss_mask=0` partout.
   - 11 tokens passent à `loss_mask=1` et correspondent exactement au
     contenu assistant injecté.

3. `RLHFDataset` prompt vs `scratch/03_eval_qwen.py` prompt (parité
   train/eval)
   - Les deux prompts font **1382 chars** chacun.
   - **Identiques byte-for-byte** : mêmes règles TextCraft
     (`conversation_start[0]`), même ACK assistant, aucun few-shot ni
     primer caché côté verl.

→ Pas de bricolage opaque côté upstream. Si on entraîne avec verl on
verra exactement le même prompt qu'à l'eval — pas de drift.

### Smoke (b) — `examples/eval/textcraft_eval.local.sh` (on-GPU, 1× A100)

But : faire passer 100 items d'eval verl bout-en-bout (Qwen-3B brut)
sur un seul GPU. **FAIL — crash NCCL.**

Setup au moment du lancement : GPU 0 MiB / 0 %, TextCraft serveur up
(pid 71473), `load_format=safetensors`, `gpu_memory_utilization=0.85`,
`tensor_model_parallel_size=1`.

Séquence du crash (`/tmp/smoke_verl_eval.log`) :
- Worker Ray spawn OK.
- Qwen-3B chargé sur CPU (`Loading checkpoint shards: 2/2` OK).
- `NCCL version 2.20.5+cuda12.4` imprimé.
- **Worker meurt immédiatement** (`SYSTEM_ERROR`, exit code 2,
  pas de Python traceback) — c'est un SIGSEGV silencieux.

Diagnostic :
- `dmesg --since "5 min ago"` : pas d'OOM-killer.
- GPU à 0 MiB avant et après → pas un OOM côté CUDA.
- Crash avant le chargement vLLM des poids → pas un problème de
  weights/safetensors.
- → C'est l'init du process group NCCL à `world_size=1` qui plante.
  Comportement non testé côté upstream verl (le fork suppose
  ≥ 2 ranks).

C'est **exactement la même panne** que celle déjà documentée
(`Error 4` — Ray ActorDiedError pendant
`verl.agent_trainer.main_generation`) au début du projet. On confirme
donc empiriquement la conclusion de l'audit code : **le code path
single-GPU de verl ne marche pas**, patcher prendrait des jours sans
garantie.

### Décision

- Smoke (a) ✅ : on a la garantie que la math de verl est bonne et que
  son prompt = notre prompt d'eval. Pas de drift caché.
- Smoke (b) ❌ : on ne peut pas tourner verl tel quel sur 1 GPU, même
  pour de l'eval. Donc inutile d'essayer pour du training.
- → On provisionne **8× A100 40 Go** depuis le snapshot disque actuel
  et on lance `examples/train/AgentGym-RL/textcraft_train.sh` avec
  juste deux overrides pour le 40 Go vs 80 Go :
  `actor_rollout_ref.rollout.gpu_memory_utilization=0.65`
  et `data.max_response_length=8192` (down de 10240). Le reste de la
  recette papier reste intact (N=8, bs=32, full FT FSDP,
  kl_coef=0.001, rounds=30, lr=1e-6).

Détails dans [`docs/RESULTS.md`](docs/RESULTS.md) §"Audit verl + smoke
tests (préalable à la migration 8× A100)".

### Checklist de reprise sur la VM 8× A100 40 Go

À faire à la prochaine session, dans cet ordre :

1. **Sanity checks de la nouvelle VM**
   ```bash
   nvidia-smi             # doit montrer 8 GPUs, 40 Go chacun
   df -h ~                # disque (100 Go) toujours plein de ce qu'on avait
   curl -I https://gitlab.crto.in:8443   # tunnel GitLab OK (HTTP/2 302)
   git -C ~/rl-gym-workout status        # branche main, working tree clean
   ```

2. **Réveiller le serveur TextCraft** (en arrière-plan, persistant)
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate agentenv-textcraft
   cd ~/rl-gym-workout/AgentGym/agentenv-textcraft
   nohup setsid textcraft --host 127.0.0.1 --port 36005 \
       > /tmp/textcraft.log 2>&1 < /dev/null &
   sleep 3
   curl -sS -X POST http://127.0.0.1:36005/create -H 'content-type: application/json' -d '{}'
   ```

3. **Replay rapide du smoke (a) pour valider l'env conda** (~12 s, CPU)
   ```bash
   conda activate agentgym-rl
   cd ~/rl-gym-workout
   python scratch/smoke_verl_test.py   # doit afficher 3 PASS
   ```

4. **Replay du smoke (b) sur 8 GPUs** = première vraie validation matérielle
   - Modifier `examples/eval/textcraft_eval.local.sh` :
     `trainer.n_gpus_per_node=8`, `rollout.tensor_model_parallel_size=8`,
     `rollout.gpu_memory_utilization=0.65`.
   - Run sur 100 items eval. ETA ~5 min (8 GPUs, vLLM TP=8).
   - Si Pass@1 ≈ 14/100 → on retrouve notre baseline. ✅ → étape 5.
   - Sinon → debug NCCL/vLLM TP avant de lancer le training.

5. **Lancer le training papier-exact** via `examples/train/AgentGym-RL/textcraft_train.sh`
   avec, en plus des overrides du script (cf. l'analyse §"Écart à la recette papier" dans RESULTS.md) :
   - `trainer.n_gpus_per_node=8`, `trainer.nnodes=1`
   - `actor_rollout_ref.rollout.tensor_model_parallel_size=8`
   - `actor_rollout_ref.rollout.gpu_memory_utilization=0.65` (40 Go vs 80 Go)
   - `data.max_response_length=8192` (down de 10240, gain mémoire ~20 %)
   - Tout le reste du `textcraft_train.sh` reste intact (N=8, bs=32,
     full FT FSDP, kl_coef=0.001, rounds=30, lr=1e-6, ~120 steps).

6. **Monitoring training**
   - Tail `train_*/grpo_actor.log` pour `train/loss`, `train/reward_mean`,
     `train/grad_norm`.
   - Sur 8× A100 40 Go avec recette papier complète, ETA training : 6-12 h
     (à confirmer après les 5 premiers steps).

7. **Eval finale** sur 100 items test, comparaison Pass@1 vs paper 75/100.
   Documenter dans `docs/RESULTS.md` §"Exp 6 — Réplication papier 8× A100".

Si tu reviens et que tu vois ce message en relisant le worklog : tu peux
me demander directement *"on reprend la checklist 8× A100, étape 1"*, je
saurai où on en est sans devoir tout redécouvrir.

## Session 2026-05-13 — Exp 6 (réplication papier verl, Qwen-3B, 4× A100 40 GB)

- VM passée de 1× A100 à **4× A100 40 GB** (compute_capability 8.0, driver 580 / CUDA 13).
- Serveur TextCraft relancé `nohup setsid textcraft --host 127.0.0.1 --port 36005`.

### Root cause du "crash NCCL world_size=1" qu'on traînait depuis 3 semaines

C'était **pas** un bug de `verl` single-GPU. C'était un bug de l'image GCP A100 :
`/etc/profile.d/env.sh` source `/usr/local/gib/scripts/set_nccl_env.sh`, qui force
`NCCL_NET=gIB` + ~10 tuner flags InfiniBand. Le plugin gIB tente de charger
`libibverbs.so` (non installé sur l'image), puis NCCL crash silencieusement
juste après la ligne `NCCL version 2.20.5+cuda12.4` avec `SYSTEM_ERROR exit code 2`.
On le reproduisait à world_size=1 ET à world_size=4 — d'où la confusion.

Validé empiriquement via `scratch/smoke_fsdp_4gpu.py` (test FSDP standalone Qwen-3B
4 GPUs) : on reproduit le crash avec l'env GCP par défaut, et il disparaît avec :

```bash
env -i ... LD_LIBRARY_PATH= NCCL_NET=Socket NCCL_IB_DISABLE=1 python3 ...
```

### Fix permanent dans `examples/train/AgentGym-RL/textcraft_train.4gpu.sh`

Bloc "NCCL workaround for GCP A100 VMs" qui :
- `unset` tous les `NCCL_*` injectés par gIB,
- supprime `/usr/local/gib/lib64` du `LD_LIBRARY_PATH`,
- force `NCCL_NET=Socket` et `NCCL_IB_DISABLE=1`.

Le NVLink intra-node fonctionne quand même par-dessus le contrôle TCP. NCCL
tombe en mode P2P/SHM natif et le bandwidth NVLink est utilisé pour les
all-reduce/broadcast comme avant.

### Autres workarounds nécessaires pour faire tourner verl tel quel

1. **`load_format=safetensors` n'existe pas** dans l'enum `LoadFormat` du fork
   `verl.third_party.vllm`. Notre `.local.sh` le forçait pour contourner le
   crash NCCL en single-GPU. Sur 4 GPUs on retire l'override → le défaut
   `dummy_dtensor` fonctionne (vLLM init avec poids aléatoires, puis sync
   FSDP DTensor par le `FSDPVLLMShardingManager`).
2. **VRAM 40 GB demande gradient_checkpointing=True + optimizer_offload=True
   + gpu_memory_utilization=0.4 + max_response_length=4096**. Sans ces flags,
   OOM à la première backward (37.66/40 GB déjà occupés, manque 4.6 GB).

### Résultat smoke test (1 step) — PASS

```
step:1 - critic/task_score/mean: 0.016 - critic/task_score/max: 1.000
       - actor/grad_norm: 0.421 - actor/pg_loss: 0.044 - actor/kl_loss: 0.001
       - response_length/mean: 2065 - timing_s/step: 284s (gen 248 + ref 10 + upd 22)
       - GPU memory used: ~23 GB / 40 GB per GPU
```

Pipeline complet upstream **directement réutilisable** sur notre matériel
après les 3 fixes ci-dessus. Aucune modification du code Python verl.

### Bilan run (2026-05-13)

- Training **50 steps** terminé (run `agentgym_rl_qwen3b_4gpu_fixed_20260513_0941_a2a5b6d/`, logs sous `saves/` — gitignoré). Eval post-training + merge : voir section **« Session 2026-05-13 (fin) »** ci-dessous.

### Nouveaux fichiers

- `examples/train/AgentGym-RL/textcraft_train.4gpu.sh` — script training principal.
- `examples/eval/textcraft_eval.4gpu_ckpt.sh` — merge FSDP + eval Pass@1.
- `scratch/auto_eval_after_train.sh` — watcher PID → eval auto.
- `scratch/smoke_fsdp_4gpu.py` — repro standalone du bug gIB.
- `docs/CODE_GUIDE_AGENTGYM_RL.md` — guide de lecture complet de la stack.

## Session 2026-05-13 (fin) — Eval post-training Exp 6, fix merge HF, courbes, clarification « Pass@1 »

### Résultats enregistrés (dans le dépôt Git)

| Artefact | Chemin |
|---|---|
| Logs épisode (100 JSON + transcripts) | `scratch/eval_logs_4gpu_global_step_50/textcraft_*.json` |
| Tableau scores (CSV) | `scratch/eval_logs_4gpu_global_step_50/scores_table.csv` |
| Résumé numérique (JSON) | `scratch/eval_logs_4gpu_global_step_50/scores_summary.json` |
| Log texte vLLM / items | `scratch/eval_logs_4gpu_global_step_50/eval.log` |
| Courbes training (PNG) | `scratch/training_curves.png`, `scratch/training_losses.png` |
| Script agrégation reproductible | `scratch/summarize_textcraft_eval_dir.py` |

**Eval** (checkpoint verl `global_step_50`, merge FSDP → HF, puis `scratch/03_eval_qwen.py` + vLLM standard, 100 items `textcraft_test.json`) :

- **38 succès / 100** → **taux de réussite 38 %** sur le test set.
- **Métrique affichée « Pass@1 » dans le script** : c’est le **même nombre** (38/100) parce qu’on ne fait **qu’un seul rollout indépendant par `item_id`**. Ce n’est **pas** une limite à « un tour » entre le LLM et TextCraft (voir paragraphe suivant).
- `mean_rounds` (agrégat JSON) ≈ **21.2** tours / épisode en moyenne sur les 100 items — les succès utilisent souvent peu de tours, les échecs tapent souvent le plafond **30** (`MAX_ROUNDS` dans `03_eval_qwen.py`, aligné sur les **30 rounds** du training papier / `textcraft_train.4gpu.sh`).

### Pourquoi le mot « Pass@1 » prête à confusion (et ce que le papier autorise vraiment)

- Le papier / la config verl fixe un **horizon intra-épisode** : jusqu’à **30 allers-retours** modèle ↔ environnement **par problème** (une trajectoire).
- Le nom **Pass@1** vient de la littérature *code generation* : **k essais indépendants sur le même problème** avant de compter un succès (ex. pass@64 = au moins un succès parmi 64 programmes tirés). Ici **k = 1** au sens **« un seul tirage complet par item d’éval »**, pas « une seule action ».
- Donc : **30 tours max ≠ Pass@1**. Les deux axes sont orthogonaux : **tours** = profondeur d’une trajectoire ; **pass@k** = combien de trajectoires complètes on autorise par item. Notre script pourrait afficher « **success rate (1 trial/item)** » pour éviter l’ambiguïté ; le terme Pass@1 a été repris pour coller aux tableaux du papier (où le test final est aussi typiquement **une** politique évaluée sur N problèmes).

### Modifs techniques effectuées (cette session / chat)

1. **`examples/eval/textcraft_eval.4gpu_ckpt.sh`** — merge FSDP → HuggingFace **conditionnel corrigé** : avant, on ne lançait `model_merger.py` que si `actor/huggingface/config.json` était absent. Un merge interrompu laissait tokenizer + config **sans** `*.safetensors` → vLLM : `Cannot find any model weights`. Désormais on merge si aucun poids HF n’est détecté (`*.safetensors` ou `pytorch_model.bin`), ou si `FORCE_MERGE=1`.
2. **`scratch/03_eval_qwen.py`** — `MODEL_PATH` et `EVAL_LOG_DIR` lus depuis l’environnement (pour l’eval checkpoint) ; docstring clarifiant Pass@1 vs `MAX_ROUNDS`.
3. **`scratch/plot_training_curves.py`** (+ PNG) — courbes parsées depuis `run.log` (fichier sous `saves/…`, **gitignoré** ; les PNG dans `scratch/` documentent le run localement).
4. **`scratch/summarize_textcraft_eval_dir.py`** — export CSV/JSON des scores à partir des logs d’épisode.

### Training Exp 6 (rappel chiffré, `run.log` local)

- **49 lignes** `step:N` parsées pour les courbes (steps 1–49 ; checkpoint `global_step_50` sauvegardé à la fin).
- Somme des `timing_s/step` sur ces lignes ≈ **9617 s** (~2 h 40) de steps chronométrés ; wall-clock total training plus élevé (init Ray, checkpoints). Voir `saves/agentgym_rl_4gpu/agentgym_rl_qwen3b_4gpu_fixed_20260513_0941_a2a5b6d/run.log` sur la VM.

---

## Session 2026-05-18 — Réorganisation complète du dépôt (directives encadrant)

Suite à une réunion avec l'encadrant chercheur, le dépôt a été restructuré pour séparer
clairement le code qu'on a écrit, les dépendances externes, les données et les résultats par run.

### Modifications effectuées

| Opération | Avant | Après |
|---|---|---|
| Scripts principaux | `scratch/03_eval_qwen.py`, `07_trl_grpo_textcraft_smoke.py`, … | `src/eval_baseline.py`, `src/train_grpo.py`, … |
| Scripts exploration | `scratch/01_minicycle.py`, `02_…`, `06_…`, `10_…`, `smoke_*` | `runs/prototypes/` |
| Logs d'éval par run | `scratch/eval_logs*/` (6 dossiers) | `runs/exp*/eval_logs*/` |
| Datasets | `AgentEval/` | `data/` |
| Dépendances externes | `AgentGym/`, `AgentGym-RL/`, `examples/` à la racine | `external/AgentGym/`, `external/AgentGym-RL/`, `external/agentgym_rl_paper/` |
| PDFs des papiers | `2406.04151v1.pdf`, `2509.08755v1.pdf` à la racine | `docs/references/agentgym_rl_paper.pdf`, `docs/references/agenteval_dataset_paper.pdf` |
| Images README fork | `assets/` (11 fichiers) | supprimé |
| Configs Hydra verl | `outputs/2026-05-*/` | supprimé (redondant avec scripts dans `external/`) |
| `scratch/` | dossier fourre-tout | supprimé après vidage |

### Nouveaux fichiers créés

- `external/USAGE.md` — liste exacte des fichiers utilisés dans les dépendances externes (server TextCraft, verl)
- `runs/exp*/config.yaml` — config + hyperparamètres + résultats + verdict pour chaque expérience

### Chemins mis à jour dans les scripts

Tous les chemins fonctionnels cassés par la réorganisation ont été corrigés dans `src/` :

| Script | Chemin corrigé |
|---|---|
| `src/eval_baseline.py` | `AgentEval/eval/` → `data/eval/` ; `scratch/eval_logs` → `runs/exp1_baseline/eval_logs` |
| `src/eval_lora.py` | `AgentEval/eval/` → `data/eval/` ; `scratch/eval_logs_*` → `runs/eval_logs_*` |
| `src/train_grpo.py` | `AgentEval/train/` → `data/train/` |
| `src/compare_runs.py` | `scratch/eval_logs*` → `runs/exp*/eval_logs*` |
| `src/analyze_eval.py` | `__file__.parent/eval_logs` → `runs/exp1_baseline/eval_logs` |
| `src/plot_curves.py` | `scratch/training_curves.png` → `runs/training_curves.png` |
| `src/auto_eval.sh` | `examples/eval/` → `external/agentgym_rl_paper/eval/` |

### .gitignore — rien modifié

Le `.gitignore` existant couvre déjà tout correctement :
- `saves/` — checkpoints modèles (trop lourds, locaux uniquement)
- `models/` — poids Qwen de base
- `__pycache__/`, `*.pyc` — bytecode Python
- `*.pdf` — PDFs des papiers (déplacés dans `docs/references/` mais toujours gitignorés)
- `wandb/`, `checkpoints/`, `executer_logs/` — artefacts training

**Aucune ligne ajoutée au `.gitignore` lors de cette session.**

---

## Session 2026-05-19 — Analyse du dataset TextCraft et propositions d'expériences

### Analyse de la structure du dataset TextCraft

#### Mapping index → depth (correction d'une erreur précédente)

Le fichier `data/eval/textcraft_test.json` contient des `item_id` de la forme `textcraft_N`.
L'index `N` est la position dans la liste `item_recipes_min_depth(1)` triée par depth
(cf. `external/AgentGym/agentenv-textcraft/agentenv_textcraft/environment.py` ligne 166-168).
Les items depth 0 (matières premières) sont **exclus par design** du jeu.

Distribution réelle du jeu TextCraft (544 items craftables, depth ≥ 1) :

| Depth | Total jeu | Train (374) | Test (100) | Exclus |
|---|---|---|---|---|
| 1 | 124 | 93 | 31 | 0 |
| 2 | 292 | 233 | 41 | 18 |
| 3 | 117 | 47 | 25 | 45 |
| 4 | 11 | 1 | 3 | 7 |
| **Total** | **544** | **374** | **100** | **70** |

**Notre test set = le test set du papier** (31/41/25/3 = 100 ✓, vérifié par reverse-engineering
des scores Table 3 : GPT-4o 100%/87.8%/64%/0%/83% → N1=31, N2=41, N3=25, N4=3).

La différence baseline 18/100 (nous) vs 14/100 (papier) est de la variance stochastique
à temperature=1, pas un écart de dataset.

#### Les 11 items depth 4 et leur statut

| Statut | idx | Item | Structure de la recette |
|---|---|---|---|
| TEST | 533 | polished_granite_slab | granite → poli → dalle |
| TEST | 534 | polished_andesite_stairs | andesite → poli → escaliers |
| TEST | 535 | lodestone | chiseled_stone → netherite → lodestone |
| **EXCLU** | 536 | purple_banner | laine_colorée(dye chain) + bâton |
| TRAIN | 537 | lectern | bookshelf(paper+leather) + wooden_slabs |
| **EXCLU** | 538 | polished_granite_stairs | granite → poli → escaliers |
| **EXCLU** | 539 | polished_andesite_slab | andesite → poli → dalle |
| **EXCLU** | 540 | cyan_banner | laine_colorée(dye chain) + bâton |
| **EXCLU** | 541 | gray_banner | laine_colorée(dye chain) + bâton |
| **EXCLU** | 542 | lime_banner | laine_colorée(dye chain) + bâton |
| **EXCLU** | 543 | hopper_minecart | hopper(chest+iron) + minecart(iron) |

Les 7 exclus sont des coupures de la fenêtre du split (séquentiel par index).
4 d'entre eux sont des banners structurellement quasi-identiques.

#### Contrainte de comparabilité avec le papier

La seule contrainte pour être comparable au papier est de **ne pas entraîner sur les 100 items
du test set**. Le train set est entièrement libre. Les 444 items non-test (dont les 7 exclus
depth 4) peuvent tous être utilisés pour l'entraînement.

TextCraft implémente 860 fichiers de recettes Minecraft (crafting table uniquement —
pas de fourneau, pas de brassage). Ce sont de vraies recettes Minecraft, pas des simplifications.

---

### Expérience proposée A — Full TextCraft training (444 items)

**Idée** : remplacer le train set actuel (374 items) par tous les 444 items non-test.
Gain principal : 8 exemples depth 4 en training au lieu de 1 (×8).

**Ce qui change** :
- Générer un nouveau `data/train/textcraft_train_full.json` avec les 444 IDs non-test
  (items 0–532 hors test, + items 536, 538–543)
- Passer `--max-items 444` (ou pointer sur le nouveau fichier)
- Test set inchangé → résultats directement comparables au papier

**Intérêt** :
- Coût nul (même jeu, même serveur)
- Isole proprement l'effet de la couverture depth 4 en training
- Espérance : le modèle voit 8× plus de chaînes depth 4 → devrait progresser sur les
  3 items test depth 4 (actuellement 0/3 même pour AgentGym-RL-3B)

**Risque** : faible. Les 7 items exclus sont structurellement similaires aux items existants
(banners = même structure dye chain). Le modèle aura vu des patterns très proches.

---

### Expérience proposée B — Génération synthétique de recettes (style SCPO)

**Idée** : créer de nouvelles recettes inventées pour augmenter artificiellement la densité
d'exemples depth 4+ en training, sans toucher au test set.

**Principe** (inspiré de SCPO — Self-play with Critic-driven Policy Optimization) :
1. Générer des arbres de recettes synthétiques en combinant des items existants :
   `synthetic_item_A = lectern + hopper` (depth 5),
   `synthetic_item_B = lodestone + polished_granite_stairs` (depth 6), etc.
2. Ajouter ces recettes comme fichiers JSON dans `agentenv_textcraft/recipes/`
3. Le serveur TextCraft les charge automatiquement (CraftingTree parse tous les JSON)
4. Entraîner sur ces items synthétiques + les 444 items réels

**Intérêt** :
- Contrôle total sur la difficulté : on peut créer des recettes depth 5, 6, 7...
- Potentiellement utile pour le curriculum (ScalingInter sur des depths qu'on contrôle)
- Direction de recherche originale, pas dans le papier

**Risques** :
- Recettes synthétiques peut-être mal formées (cycles, items inaccessibles)
- Distribution shift : le modèle s'entraîne sur des items "non-Minecraft" → effet sur
  la généralisation incertain
- Engineering non trivial : il faut valider que CraftingTree accepte les nouvelles recettes
  et que le serveur les résout correctement

**Ordre de priorité** : faire A d'abord (gain certain, coût nul), puis B si A donne des
résultats mais que depth 4 reste bloqué à 0/3.

---

---

## Session 2026-05-20 — Eval Gemini 3.5 Flash sur TextCraft + scripts eval API externes

### Objectif

Mesurer les performances d'un modèle SOTA propriétaire (Gemini 3.5 Flash) sur TextCraft
sans fine-tuning, pour fixer une borne supérieure pratique. Créer une infrastructure
d'évaluation réutilisable pour tout modèle API (Kimi, DeepSeek, OpenAI, etc.).

### Nouveaux fichiers

| Fichier | Rôle |
|---|---|
| `src/eval/eval_gemini.py` | Eval multi-tours via API Google Gemini (google-genai) |
| `src/eval/eval_openai_compat.py` | Eval générique pour toute API OpenAI-compatible |
| `src/eval/analyze_gemini.py` | Rapport par depth + comparaison baseline |
| `runs/exp_gemini_baseline/config.yaml` | Config expérience Gemini |
| `runs/exp_gemini_gemini_3_5_flash/` | Logs épisodes (100 JSON) |
| `runs/exp_kimi_k2_6/config.yaml` | Config Kimi K2.6 (noms API à vérifier) |
| `runs/exp_deepseek_v4_pro_max/config.yaml` | Config DeepSeek V4 Pro Max (noms API à vérifier) |

### Résultats Gemini 3.5 Flash

```
Depth    Items  Solved  Pass@1   Rds moy   Dur moy
depth 1     31      31  100.0%       5.6      30.2s
depth 2     41      41  100.0%       8.3      49.6s
depth 3     25      24   96.0%      15.6      81.9s
depth 4      3       3  100.0%      26.0     140.0s
TOTAL      100      99   99.0%
```

| Modèle | Pass@1 | depth 3 | depth 4 |
|---|---|---|---|
| Gemini 3.5 Flash (0 training) | **99%** | 96% | 100% |
| AgentGym-RL paper (Qwen 3B full FT, N=8) | 75% | ~64% | ~33% |
| Qwen 2.5-3B baseline (0 training) | 18% | 0% | 0% |

**Gemini 3.5 Flash écrase le papier de référence** (99% vs 75%). Sert de borne
supérieure — confirme que TextCraft est quasi-entièrement résolvable par un grand modèle.

### Analyse de l'unique échec (textcraft_422, depth 3)

Item : `minecraft:pink_banner` (recette : 6 pink_wool + 1 stick).
Comportement : Gemini boucle indéfiniment sur `craft 1 pink_wool using 1 pink_dye, 1 white_wool`
pendant les 30 tours sans jamais assembler le banner. Cause probable : perte de l'objectif
final dans un contexte chargé (~60 messages) + manque de stick non détecté.
Ce comportement "boucle sur sous-objectif" est identique à ce qu'on observe sur Qwen,
mais Qwen le fait sur 25/25 items depth-3 vs 1/25 pour Gemini.

### Problèmes techniques rencontrés

1. **Quota free tier Gemini 3.5 Flash** : limite de 20 requêtes/jour. Résolu en activant
   la facturation Google AI Studio (coût total ~quelques centimes pour 100 items).

2. **503 UNAVAILABLE persistants** : Gemini 3.5 Flash venait de sortir et était surchargé.
   Résolu en lançant une boucle de retry overnight (`while true; do ...; sleep 600; done`).
   Les 503 se sont dissipés vers 19h UTC.

3. **Timeout API Gemini** : sans timeout, certains appels pendaient indéfiniment (>20 min).
   Fix : passer un client `httpx.Client(timeout=90.0)` au constructeur `genai.Client()` via
   `http_options=genai_types.HttpOptions(httpx_client=httpx.Client(timeout=90.0))`.
   Note : le champ `timeout=N` de `HttpOptions` est en millisecondes (N=90 → 90ms → trop court),
   et ne peut pas être passé directement à `generate_content()`.

### Pour lancer Kimi K2.6 / DeepSeek V4 Pro Max

Les deux utilisent `src/eval/eval_openai_compat.py` (API OpenAI-compatible).
Il faut : (1) une clé API, (2) vérifier le nom exact du modèle :
```bash
curl https://api.moonshot.cn/v1/models -H "Authorization: Bearer $API_KEY"   # Kimi
curl https://api.deepseek.com/v1/models -H "Authorization: Bearer $API_KEY"  # DeepSeek
```

---

## TODO prochaine session training

- [ ] **Ablation LoRA vs full FT** : lancer un run LoRA (r=16) avec exactement les mêmes hyperparamètres que Exp 6 (N=8, batch=8, max_response_length=4096, 4× A100) pour isoler proprement l'effet du fine-tuning method. Si LoRA atteint le même Pass@1 → les optimizer states libérés (~10-12 Go/GPU) peuvent être réinvestis en `max_response_length` plus grand (8192 voire 10240) et `max_model_len` plus grand. Si LoRA rate → confirme que full FT est nécessaire pour cette tâche.
- [ ] **Augmenter `max_response_length`** : passer de 4096 à 8192 dans `textcraft_train.4gpu.sh`. Les épisodes depth-2 qui nécessitent >8 tours étaient tronqués pendant le training → pas de signal de reward → le modèle n'apprend pas à chaîner les crafts intermédiaires. C'est probablement la cause principale du gap depth-2 (24% vs 90%).
- [ ] **Continuer Exp 6 jusqu'à ~200 steps** pour voir si la courbe converge vers 75/100 (papier). À 50 steps on a vu seulement 400 queries vs 4800 au checkpoint d'éval du papier (step 150).

