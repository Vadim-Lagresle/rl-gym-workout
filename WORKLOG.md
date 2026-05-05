# Worklog — rl-gym-workout

Journal de bord pour reprendre le projet sans perdre de contexte.

## Setup (fait)

- VM GCP `vadimagent`, **A100 40 Go**, driver 580 / CUDA 13 (compatible binaries cu124).
- Submodule `AgentGym/` initialisé (`git submodule update --init --recursive`).
- Miniconda installé dans `~/miniconda3`.
- Deux envs conda **persistent sur le disque** :
  - `**agentgym-rl`** (Python 3.10) — la stack d'entraînement
  (torch 2.4 cu124, flash-attn 2.7.3, vllm dev, ray 2.55.1, transformers 4.51.3, verl, agentenv).
  - `**agentenv-textcraft**` (Python 3.10) — le serveur de jeu HTTP
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

