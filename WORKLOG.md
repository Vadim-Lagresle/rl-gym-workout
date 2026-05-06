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


