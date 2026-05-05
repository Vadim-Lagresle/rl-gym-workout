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

