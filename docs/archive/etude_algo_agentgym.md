# Étude algo AgentGym-RL (verl)

Document de travail — session 2026-06-02.
On parcourt le code verl fichier par fichier pour comprendre ce qu'ils font
exactement et pourquoi ça donne 38/100 là où notre stack TRL donne 15/100.

---

## Architecture `external/AgentGym-RL/verl/`

```
verl/
├── agent_trainer/          ← POINT D'ENTRÉE du training
│   ├── main_ppo.py         ← script lancé par le .sh (parse config, init Ray)
│   └── ppo/
│       ├── ray_trainer.py  ← BOUCLE PRINCIPALE GRPO (le chef d'orchestre)
│       └── core_algos.py   ← calcul du loss GRPO (les maths pures)
│
├── workers/                ← LES WORKERS (s'exécutent sur chaque GPU via Ray)
│   ├── agent_fsdp_workers.py       ← instancie acteur + critique avec FSDP
│   ├── agent_actor/
│   │   └── dp_actor.py             ← modèle de politique (forward, update)
│   ├── agent_critic/
│   │   └── dp_critic.py            ← value function (pas utilisée en GRPO pur)
│   ├── rollout/
│   │   └── agent_vllm_rollout/
│   │       └── vllm_rollout.py     ← ROLLOUT MULTI-TOUR (joue les épisodes)
│   ├── reward_manager/
│   │   └── naive.py                ← appelle le serveur TextCraft, renvoie reward
│   └── sharding_manager/
│       └── fsdp_vllm.py            ← sync poids FSDP → vLLM entre chaque step
│
├── protocol.py             ← structures de données partagées (DataProto, etc.)
├── utils/
│   ├── agent_dataset/
│   │   └── rl_dataset.py   ← charge les items TextCraft, les met en batch
│   └── agentgym/
│       └── client.py       ← client HTTP vers le serveur TextCraft (port 36005)
│
└── third_party/vllm/       ← wrappers vLLM (versions 0.3, 0.4, 0.5) — plomberie
```

---

## Résumé fichier par fichier (première passe)

**`main_ppo.py`** — le `__main__`. Lit le fichier YAML de config (learning rate, N,
batch size…), initialise Ray sur les GPUs, instancie le `RayPPOTrainer` et appelle
`trainer.fit()`. C'est là que tout démarre.

**`ppo/ray_trainer.py`** — le chef d'orchestre. La méthode `fit()` contient la boucle
`for step in range(total_steps)`. À chaque step elle :
1. dit au rollout worker de jouer N épisodes
2. récupère les trajectoires
3. calcule les rewards via `reward_manager`
4. calcule les avantages GRPO
5. envoie au actor worker pour faire le gradient update

**`ppo/core_algos.py`** — les maths de GRPO. Contient les fonctions qui calculent le
ratio importance sampling, le clip PPO, et la normalisation des avantages par groupe.
C'est ici que réside la différence GRPO vs PPO classique.

**`vllm_rollout.py`** — joue les épisodes multi-tours. Pour chaque item du dataset,
il boucle sur les tours : génère une action avec vLLM → envoie au serveur TextCraft →
récupère l'observation → recommence jusqu'à `done` ou `max_rounds`. Retourne la
trajectoire complète (tokens + masques + rewards).

**`agent_fsdp_workers.py`** — crée les workers Ray avec FSDP (Fully Sharded Data
Parallel). FSDP répartit les poids du modèle sur les N GPUs — chaque GPU ne stocke
qu'une fraction des paramètres, et les recompose à la volée pendant le
forward/backward.

**`dp_actor.py`** — le modèle de politique côté training. Reçoit une trajectoire,
calcule les logprobs avec le modèle courant (`π_actuelle`), appelle `core_algos` pour
le loss, fait le backward.

**`fsdp_vllm.py`** — synchronise les poids entre FSDP (training) et vLLM (génération)
après chaque update. C'est la colle entre les deux mondes.

**`naive.py` (reward_manager)** — appelle le client TextCraft pour chaque épisode,
récupère le reward final (0 ou 1), et peut appliquer un reward shaping (ScalingInter
dans les expés du papier).

**`protocol.py`** — définit `DataProto`, la structure de données qui circule entre tous
les workers (tenseurs de tokens, masques, rewards, logprobs…). C'est le format
d'échange universel.

**`rl_dataset.py`** — charge `textcraft_train.json`, construit les batchs d'items à
entraîner, gère le shuffle et la pagination entre steps.

**`client.py`** — client HTTP léger vers le serveur FastAPI TextCraft (port 36005).
Expose `reset(item_idx)`, `step(action)`, `observe()`.

---

## Analyse détaillée par fichier

*(à remplir au fil de la session)*