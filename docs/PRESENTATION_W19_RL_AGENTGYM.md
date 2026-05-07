# RL multi-turn sur LLM — point d'avancement semaine 19

*Auteur : Vadim Lagresle — semaine du 2026-05-04 au 2026-05-07*

> **TL;DR** — Setup infra complet sur VM A100, eval baseline **Qwen2.5-3B-Instruct sur TextCraft** : **Pass@1 = 18 / 100**. Tentative de training avec le framework upstream **AgentGym-RL/verl** échouée en single-GPU (NCCL/FSDP). **Pivot vers TRL+GRPO + LoRA** : pipeline de training interactif multi-tour fonctionnel, validé sur 2 et 3 steps. Prochaine étape : training plus long + eval de contrôle.

---

## 1. Contexte et objectif

- **Objectif scientifique** : entraîner des **LLM agents multi-turn** par RL (PPO/GRPO) dans un environnement interactif.
- **Choix d'environnement** : **TextCraft** (jeu de crafting type Minecraft, accessible via HTTP, reward sparse 0/1).
- **Choix de framework** : **AgentGym-RL** (papier ScalingInter-RL, multi-env, plug & play en théorie). Construit sur deux briques : **AgentGym** (les environnements) et **verl** (RL training).
- **Modèle de base** : Qwen2.5-3B-Instruct (compromis taille / VRAM A100 40 Go).

---

## 2. Setup infrastructure

| Composant | État |
|---|---|
| VM GCP A100 40 Go (`vadimagent`, zone `europe-west4-b`) | OK, snapshot pour repartir d'un état stable |
| Tunnel SSH vers `gitlab.crto.in:8443` | OK |
| Submodule `AgentGym/` (14 environnements + agentenv) | OK |
| Env conda `agentgym-rl` (torch 2.4 cu124, vllm, ray, transformers 4.51.3, verl) | OK |
| Env conda `agentenv-textcraft` (FastAPI/uvicorn, jeu) | OK |
| Env conda `trl-grpo` (torch 2.6, transformers 4.57.1, trl 0.19, peft) | OK (créé après le pivot) |
| Modèle Qwen2.5-3B-Instruct téléchargé (5.8 Go) | OK |

**Architecture runtime** :

```
┌───────────────────────────┐  HTTP  ┌──────────────────────────┐
│ Trainer (agentgym-rl)     │  ───▶  │ Env server (agentenv-...)│
│   verl + vLLM + Ray       │  ◀───  │ FastAPI + uvicorn         │
│   GPU                     │        │ port 36005                │
└───────────────────────────┘        └──────────────────────────┘
```

5 endpoints REST exposés par le serveur d'env :
`POST /create`, `POST /reset`, `POST /step`, `GET /observation`, `POST /close`.
Côté Python, c'est `TextCraftEnvClient` qui les wrap (méthodes `reset`, `observe`, `step`, `close`).

---

## 3. Phase 1 — Eval baseline Qwen-3B sur TextCraft

### Pourquoi un script custom et pas le upstream
Le script upstream `examples/eval/textcraft_eval.sh` utilise `verl.agent_trainer.main_generation` qui suppose **multi-GPU + FSDP + checkpoint au format spécifique**. En single-GPU avec un modèle Hugging Face brut, **NCCL crashe** (`ActorDiedError` Ray) — la pile `verl.third_party.vllm` insiste pour faire du `dummy_dtensor` sharding même sur 1 GPU.

→ Décision : **`scratch/03_eval_qwen.py`** = vLLM standard + `TextCraftEnvClient` + boucle multi-turn maison.

### Boucle d'eval (très simplifiée)
```python
for item in dataset:
    env.reset(item.id)
    obs = env.observe()
    msgs = [system, manuel, ack, obs]
    for _ in range(max_rounds):
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        out = vllm.generate(prompt, ...)
        msgs.append({"role": "assistant", "content": out})
        step = env.step(out)            # parse "Action: ..." côté serveur
        msgs.append({"role": "user", "content": step.state})
        if step.done: break
    log(item.id, reward=step.reward, transcript=msgs)
```

Features ajoutées : **mode resume** (skip items déjà loggés via cache JSON), `MAX_ITEMS` env var pour debug.

### Résultats

| Métrique | Valeur |
|---|---|
| Pass@1 | **18 / 100 = 18%** |
| Avg reward | 0.18 |
| Items évalués sur l'A100 spot | 77 (1ère VM préemptée) |
| Items complétés sur l'A100 stable | 23 (resume) |

### Analyse qualitative des échecs (`scratch/04_analyze_eval.py`)
3 catégories dominantes identifiées sur les 82 échecs :
- **Multi-action dans un même message** (~30%) : Qwen écrit plusieurs `Action:` d'un coup, le serveur n'en exécute qu'une et renvoie un message d'erreur.
- **Action loops** (~20%) : répétition de la même action ≥3 fois (>40% des actions de l'épisode), souvent un `inventory` ou `craft` invalide.
- **Env complaints** (~25%) : actions au mauvais format (`could not parse`, `wrong item format`).

→ Le modèle 3B brut est limité par sa **rigueur formelle**, pas tant par sa stratégie. Cible idéale pour du RL : reward shaping qui pénalise multi-action et loops.

---

## 4. Phase 2 — Tentative de training upstream (verl)

### Smoke test
- Script local : `examples/train/AgentGym-RL/textcraft_train.local.sh`
- Overrides Hydra forcés : `n_gpus_per_node=1`, `total_training_steps=1`, `adv_estimator=grpo`, batch sizes minimaux.
- **Crash identique à l'eval** : `ActorDiedError` au moment où `verl.third_party.vllm` charge les poids en `dummy_dtensor`.

### Diagnostic
- `verl` est **un fork lourd de vLLM** spécifique multi-GPU + FSDP.
- Le code suppose qu'il y a au moins 2 GPUs pour faire le sharding ; en single-GPU, le code path n'est pas testé et NCCL refuse d'initialiser un groupe à 1 rank.
- Patcher le fork prendrait des jours sans garantie de résultat.

### Décision (avec hard cap 4h respecté)
**Pivot vers TRL+GRPO** plutôt que de fork verl. Coût d'opportunité accepté : on perd la feature ScalingInter-RL "out of the box" (curriculum sur `max_rounds`), à reimplémenter à la main si pertinent.

Doc complémentaire produite : `docs/AGENTGYM_RL_TRAINING_DEEPDIVE.md` (RayPPOTrainer, FSDP, vLLM rollout, hooks de customisation, hyperparamètres clés).

---

## 5. Phase 3 — Pipeline TRL+GRPO custom (où on est aujourd'hui)

### Stack
- Env conda dédiée `trl-grpo` (isolée des conflits transformers/torch de la stack `agentgym-rl`).
- **GRPOTrainer** de `trl 0.19.1`.
- **LoRA** (rank 16, alpha 32, sur tous les `*_proj`) : indispensable pour tenir Qwen-3B + grad + optimizer + KV cache sur 40 Go.
- `bf16`, `per_device_train_batch_size=1`, `max_completion_length=128`, `num_generations=2`.

### Le point critique : le `rollout_func` interactif

**Problème** : par défaut, `GRPOTrainer` génère **une completion d'un coup** (open-loop), puis on calcule un reward dessus. Ça ne reflète pas le vrai jeu, où chaque action change l'observation suivante.

**Solution implémentée** (`scratch/07_trl_grpo_textcraft_smoke.py`) : on passe à TRL un `rollout_func` custom qui fait le vrai loop multi-tour pendant la génération :

```python
def textcraft_rollout_func(prompts, trainer):
    # 1. crée un TextCraftEnvClient par sample, reset(item_id)
    # 2. pour chaque tour (jusqu'à MAX_SIM_ROUNDS=20) :
    #    a. render le prompt chat (états + observation courante)
    #    b. trainer._generate_single_turn(...) → completion + logprobs
    #    c. extract "Action: X" de la completion
    #    d. env.step(...) → nouvelle obs + reward + done
    #    e. append assistant turn + user turn dans le state
    #    f. concat completion_ids et logprobs sur tous les tours
    # 3. retourne {prompt_ids, completion_ids, logprobs,
    #              episode_reward, invalid_steps}
```

La **reward function** consomme `episode_reward` (sparse 0/1) + `invalid_steps` (count des erreurs serveur) et applique un shaping léger :
- `+0.02` si exactement 1 action par message
- `-0.05` si 0 ou >1 actions par message
- `-0.01` par invalid step

### Pourquoi c'est non-trivial
- L'API `rollout_func` de TRL est **expérimentale** (`UserWarning` à chaque run, pas documentée).
- Une approche alternative `environment_factory` existe mais **requiert `transformers>=5.2.0`** (pas encore released) → on contourne en gérant l'env directement dans `rollout_func`.
- Concaténer correctement les `logprobs` sur plusieurs tours, en respectant les masks de prompt vs completion, est piégeux ; bug dans cette concat = gradient incorrect = training silencieusement faux.

### Validation
| Run | Setup | Résultat |
|---|---|---|
| Smoke 1 step | open-loop, multi-action simulé | OK, exit_code=0 |
| Stability 20 steps | open-loop | OK, train_runtime ~88s |
| Smoke 2 steps | **interactive rollout_func** | OK, reward moyen ~0.36, step_time ~111s |
| Run 10 steps | interactive rollout_func | lancé puis interrompu (changement de GPU) — atteint 2/10 sans crash |

---

## 6. Synthèse pour les encadrants

### Ce qu'on a appris
1. **AgentGym-RL est un environnement-zoo solide, mais sa partie training (`verl` fork) est conçue pour multi-GPU.** Pas un sandbox pédagogique — pas adapté à un setup single-GPU sans investir des jours de dette technique.
2. **Le baseline Qwen-3B fait 18% sur TextCraft.** Marge de progression réelle, en particulier sur la rigueur de format (multi-action et loops sont des "low hanging fruits" pour le RL).
3. **TRL+GRPO + LoRA + custom rollout_func** est un setup **viable, contrôlable et fast iteration** pour faire du RL multi-turn en single-GPU. C'est notre stack de travail.

### Ce qui marche maintenant
- Pipeline d'eval reproductible avec resume.
- Pipeline de training GRPO interactif (vrai loop multi-tour avec env entre chaque action).
- Outils d'analyse qualitative des transcripts (`scratch/04_analyze_eval.py`, `scratch/05_view_log.py` avec annotation des "vrais auteurs").
- Tout commité/pushé sur `gitlab.crto.in:8443/v.lagresle/rl-gym-workout`, dernier commit `9a47da4`.

### Risques et zones d'ombre à challenger
- **API `rollout_func` expérimentale** : peut casser à un upgrade trl. À watch.
- **Reward shaping** : pour l'instant très simple. Reste à valider qu'il donne un gradient utile sur 50-100 steps.
- **VRAM** : on est à la limite avec LoRA sur A100 40 Go. Si on veut tester Qwen-7B il faudra changer de GPU ou passer en QLoRA.

### Question matérielle pour les encadrants
Pour la suite (runs >100 steps, scaling sur Qwen-7B, ajout d'un 2e env), est-ce qu'on peut viser un GPU plus gros — **A100 80 Go**, **H100**, voire **B200** si disponible ? L'A100 40 Go reste workable pour Qwen-3B mais limite vite l'horizon expérimental (notamment full fine-tuning et batch sizes plus grands pour stabiliser GRPO).

### Prochaines étapes (par ordre de priorité)
1. **Run training de référence** (50-100 steps) sur A100, avec checkpoints réguliers.
2. **Eval de contrôle** (subset 30 items) après training pour mesurer le delta vs Pass@1=18%.
3. Si signal positif : **reward shaping plus fin** (pénaliser explicitement les boucles, valoriser les chemins de craft optimaux).
4. Si signal négatif : revoir la **distribution des items** (le dataset train est-il trop dur pour 3B ?), ou tester un **curriculum** type ScalingInter (max_rounds croissant).
5. À moyen terme : ajouter un **2e env** (WebShop ou ALFWorld) pour valider la généricité du pipeline.

---

## Annexe A — FAQ technique

Ces clarifications ont été demandées en réunion de cadrage interne. Elles sont conservées ici pour éviter d'avoir à les ré-expliquer.

### A.1 — C'est quoi NCCL ?

**NVIDIA Collective Communications Library** : la lib bas-niveau qui permet à plusieurs GPUs (sur la même machine ou en cluster) de se synchroniser pendant un training distribué — all-reduce des gradients, all-gather des poids shardés, broadcast, etc. Sans NCCL, pas de training multi-GPU efficace. Notre crash : `verl` essaie d'initialiser un process group NCCL (à 2+ ranks) même quand on a 1 GPU. Le code part du principe qu'il y a du sharding à faire ; à 1 rank, NCCL refuse d'initialiser → `ActorDiedError` Ray.

### A.2 — verl, en détail

**verl = Volcano Engine Reinforcement Learning**, framework RL pour LLMs développé par **ByteDance** (Volcano Engine = leur cloud). C'est *la* référence pour faire du RLHF à grande échelle (utilisé pour entraîner DeepSeek-R1, Qwen, etc.). Concrètement il apporte : les algos RL (PPO, GRPO, ReMax, RLOO, DPO), le **sharding multi-GPU** (FSDP natif PyTorch ou Megatron 3D parallelism, permet d'entraîner des modèles >70B), le **rollout rapide** (intégration vLLM pour générer les completions vite — le rollout est le bottleneck en PPO), l'**orchestration Ray** (actor / critic / reference / reward = 4 workers Ray séparés sur des GPUs différents), et la **config Hydra** (YAML hiérarchiques pour gérer 200+ hyperparamètres). Très puissant **mais conçu pour un cluster**, pas pour 1 GPU. C'est notre problème.

### A.3 — AgentGym-RL = verl + ScalingInter ? Pas seulement.

AgentGym-RL est composé de **trois briques** :

1. **AgentGym** (les environnements) : 14 envs multi-tour packagés en serveurs HTTP + leurs `EnvClient` Python (TextCraft, WebShop, ALFWorld, BabyAI…). C'est le "zoo".
2. **Un verl modifié** : ils ont patché verl pour supporter le **rollout multi-tour avec env-in-the-loop**. Le rollout vLLM standard de verl est single-turn ("ici un prompt, génère N tokens, fini"). AgentGym-RL ajoute le pattern "génère un assistant turn → call `env.step` → re-génère avec la nouvelle observation → ..." dans le rollout vLLM lui-même.
3. **L'algorithme ScalingInter-RL** : un curriculum sur `max_rounds` (commence à 5 tours, augmente progressivement). C'est leur contribution scientifique propre.

Donc AgentGym-RL ce n'est pas "juste pour s'éviter de réécrire PPO" : c'est un zoo d'environnements + le glue code env ↔ rollout multi-tour ↔ trainer + l'algo ScalingInter, le tout par-dessus la machinerie verl.

### A.4 — verl est un fork de vLLM ?

Précision importante : **vLLM n'est pas un truc RL**. vLLM = serveur d'inférence ultra-rapide pour LLMs (PagedAttention, prefix caching). Il a remplacé `transformers.generate` qui est lent et c'est devenu l'état de l'art pour servir un LLM.

Mais **verl embarque sa propre version modifiée de vLLM** dans `verl/third_party/vllm/`. Pourquoi ? Pendant le RL les poids du modèle changent à chaque step. vLLM standard ne supporte pas bien le hot-reload des poids (conçu pour servir un modèle figé). Verl a besoin de lire les poids depuis FSDP (format DTensor sharded) et de les pousser dans vLLM sans repasser sur disque. Verl ajoute donc des hooks pour synchroniser actor ↔ rollout après chaque update.

→ verl ≠ vLLM, mais **verl bundle un vLLM patché** (`verl.third_party.vllm`). C'est ce fork patché qui plante en single-GPU (le code path FSDP n'existe pas à 1 rank). Un vLLM standard, lui, marche très bien à 1 GPU — c'est ce qu'on utilise dans nos scripts custom.

### A.5 — TRL vs verl

**TRL = Transformer Reinforcement Learning**, par **Hugging Face**. Même but que verl (RL sur LLMs : PPO, DPO, GRPO, KTO…) mais philosophie inverse :

| | TRL | verl |
|---|---|---|
| Cible | Prototype, single-process, single-GPU friendly | Cluster, scale |
| Sharding | Via `accelerate` (FSDP, DeepSpeed) — optionnel | FSDP/Megatron — obligatoire |
| Rollout | `transformers.generate` ou vLLM standard | vLLM forké, intégré profondément |
| API | Pythonique, customizable, peu de YAML | Hydra YAML lourd |
| Mainteneur | HF, très actif | ByteDance, actif mais moins ouvert |
| Maturité multi-turn | Récent et expérimental (`rollout_func`) | Plus mature mais via AgentGym-RL |

→ TRL c'est "easy mode", verl c'est "production mode". Pour 1 GPU et de la recherche flexible, TRL gagne.

### A.6 — Pourquoi pas tout coder à la main ?

Question légitime, surtout maintenant qu'on est sur 1 GPU. **PPO/GRPO en eux-mêmes sont ~500-800 lignes propres** ; ce n'est pas le code qui pose problème, ce sont les **détails numériques** qui rendent le RL sur LLM instable :

- **Advantage estimation** correcte (GAE pour PPO, group-relative pour GRPO) → bug = gradient mort.
- **Importance sampling ratio** + clipping PPO → bug = explosion des gradients ou updates bloquées.
- **KL divergence** vers le modèle de référence (anti reward-hacking) → bug = collapse du modèle vers une politique dégénérée.
- **Value loss** (PPO) ou normalisation des rewards par groupe (GRPO).
- **Mixed precision** (bf16 pour matmul, fp32 pour la loss) → mauvais cast = NaN silencieux.
- **Reward whitening / advantage normalization** → souvent la différence entre "ça apprend" et "ça apprend pas".
- **Gradient checkpointing** + **LoRA** + **ZeRO/FSDP** intégrés correctement.

Recoder GRPO from scratch est faisable en quelques jours. Le faire **sans bug numérique** prend typiquement **plusieurs semaines** — le RL est connu pour être instable et silencieusement faux. TRL a déjà payé ce coût (battle-tested, métriques cohérentes, intégration `peft` / `accelerate`).

**Position pragmatique** : on utilise TRL pour la **machinerie GRPO** (loss, advantage, KL, clipping, optimizer), mais on **code soi-même tout ce qui est interaction multi-tour avec l'environnement** via `rollout_func`. C'est précisément le point intéressant de notre projet et il n'est pas couvert "out of the box". On garde donc l'avantage des frameworks (stabilité numérique) sans subir leurs hypothèses (multi-GPU, rollout single-turn).

→ **Pas besoin de verl sur 1 GPU. Pas besoin non plus de tout coder. TRL est le sweet spot.**

### A.7 — LoRA : on ne fine-tune pas tout Qwen ?

Non, et c'est intentionnel. **Sans LoRA**, fine-tuner Qwen-3B en bf16 :

| Composant | Mémoire |
|---|---|
| Poids du modèle (bf16) | ~6 Go |
| Gradients (même dtype) | ~6 Go |
| Optimizer Adam (m + v en fp32) | ~12 Go |
| Activations (selon batch et seq_len) | 5-15 Go |
| KV cache de vLLM pour le rollout | 5-10 Go |
| **Total** | **~35-50 Go** |

→ ne tient pas sur A100 40 Go avec en plus un rollout vLLM en parallèle.

**Avec LoRA (rank 16)** : le modèle de base reste **figé** (pas de gradient → pas de mémoire pour ses grads/optimizer). On ajoute des **petits adaptateurs** (matrices basse-dimension `r=16`) sur les couches d'attention et MLP. Adaptateurs = ~10-20 Mo (vs 6 Go pour le full). Gradients + optimizer **uniquement sur les adaptateurs** : ~50 Mo total. On tient à l'aise.

**Trade-off** : LoRA touche moins de paramètres que le full fine-tuning, donc capacité d'apprentissage moindre. Pour un fine-tuning instruct ou RL léger, c'est largement suffisant — c'est le standard practice dans 90% des fine-tunings publiés. Si on veut apprendre une nouvelle compétence radicalement différente, on voudra du full FT (et donc plus de GPU). À terme on peut **merger les LoRA dans le base model** pour récupérer un modèle "pur" si besoin.

---

## Annexe B — Fichiers clés du repo (descriptions détaillées)

**`scratch/01_minicycle.py`** — Premier mini-script de validation (≈60 lignes) qui instancie un `TextCraftEnvClient`, fait un `reset(0)`, affiche l'observation initiale, envoie une action `inventory` factice et imprime `state / reward / done`. Aucun LLM impliqué. But : valider que le serveur HTTP TextCraft tourne et que le client Python parle bien avec lui, avant d'introduire la complexité de vLLM ou du training.

**`scratch/02_vllm_standalone.py`** — Smoke test de vLLM standard avec Qwen2.5-3B-Instruct, sans aucun lien avec TextCraft ou le RL. Charge le modèle, applique un chat template basique, génère 100 tokens, imprime la sortie. Sert à isoler les problèmes : si ça marche ici mais pas dans la pile complète, le bug vient de l'intégration verl/AgentGym, pas de vLLM lui-même. C'est ce script qui a confirmé que vLLM standard tourne très bien sur 1 GPU et que le problème était le fork `verl.third_party.vllm`.

**`scratch/03_eval_qwen.py`** — Le pipeline d'évaluation custom qui remplace `examples/eval/textcraft_eval.sh` (qui crashait). Charge le split test TextCraft (100 items), instancie vLLM standard avec Qwen-3B, et pour chaque item fait : reset env, observe, puis boucle `apply_chat_template → vllm.generate → env.step` jusqu'à `done` ou `max_rounds=20`. À chaque tour, l'observation du serveur devient un `user` turn dans la conversation. À la fin, on log un JSON par item dans `scratch/eval_logs/` (transcript complet + reward + nb tours). Mode resume implémenté : au démarrage, on lit les logs existants et on skip les items déjà évalués (utile après préemption Spot ou pour partitionner les runs). Variable d'env `MAX_ITEMS` pour eval partielle, `FORCE_REDO=1` pour ignorer le cache. Reporte à la fin un Pass@1 et la durée totale.

**`scratch/04_analyze_eval.py`** — Outil d'analyse qualitative post-hoc des logs produits par `03`. Pour chaque transcript, compte le nombre de messages assistant, détecte les messages contenant 0 ou >1 lignes `Action: ...` (échecs de format), repère les "env complaints" (regex sur les patterns `could not`, `error:`, `wrong item format`), et identifie les action loops (action répétée ≥3 fois et représentant >40% des actions de l'épisode). Sort un résumé agrégé sur les 100 items + une dizaine d'exemples concrets pour chaque catégorie d'échec. C'est cet outil qui a permis de quantifier les 3 modes d'échec dominants (multi-action ~30%, loops ~20%, format errors ~25%).

**`scratch/05_view_log.py`** — Petit visualiseur de transcripts qui résout un point de confusion : dans les logs, tous les messages sont étiquetés `system`, `user` ou `assistant` au sens OpenAI Chat API, alors que l'auteur "réel" est différent selon le tour. Le script reclassifie : le premier `system` est annoté "📜 setup" (consigne fixe), le premier `user` est "📚 manuel" (les règles du jeu de TextCraft), le `assistant` "OK. I'll follow your instructions and try my best to solve the task." est "📚 manuel (ack scripté)" — c'est une réponse en dur dans le harness, pas Qwen. Les `user` suivants sont "🎮 env" (le serveur TextCraft) et les `assistant` suivants "🤖 Qwen". Ça rend les logs lisibles d'un coup d'œil.

**`scratch/06_grpo_trl_skeleton.py`** — Squelette initial d'un pipeline TRL+GRPO, écrit juste après la décision de pivot pour cadrer l'architecture (chargement dataset, env client, signature `rollout_func`, contrat reward). Volontairement non fonctionnel, il sert de "design doc exécutable". Conservé pour traçabilité, superseded par `07`.

**`scratch/07_trl_grpo_textcraft_smoke.py`** — **Le pipeline de training TRL+GRPO qui marche aujourd'hui** (≈290 lignes). Il fait : chargement du dataset train TextCraft, construction des prompts chat (manuel + ack + observation initiale), config `GRPOTrainer` avec LoRA (rank 16 sur tous les `*_proj`), bf16, batch=1, completion_length=128, num_generations=2. Le cœur de la logique est la fonction `textcraft_rollout_func(prompts, trainer)` qu'on injecte dans le trainer : pour chaque sample, elle crée un `TextCraftEnvClient`, fait `reset(item_id)`, puis pour chaque tour appelle `trainer._generate_single_turn(...)` avec le prompt rendu via `apply_chat_template`, parse la première ligne `Action:` de la completion, fait `env.step(...)`, ajoute l'observation comme nouveau `user` turn, et concatène les `completion_ids` + `logprobs` de tous les tours dans des listes par sample. Renvoie un dict `{prompt_ids, completion_ids, logprobs, episode_reward, invalid_steps}` que TRL utilise pour calculer la loss GRPO. La fonction `textcraft_reward` consomme `episode_reward` + `invalid_steps` et applique un shaping (`+0.02` si exactement 1 action par message, `-0.05` sinon, `-0.01` par invalid step). C'est l'équivalent maison de ce qu'AgentGym-RL fait dans son fork de verl, mais en single-GPU et lisible.

**`examples/eval/textcraft_eval.local.sh`** — Copie locale du script d'eval upstream, modifié pour single-GPU + raw HF model (pas de checkpoint mergé). Conservé pour mémoire après le crash NCCL ; il ne tourne pas, mais montre l'écart entre les hypothèses du framework upstream et notre setup.

**`examples/train/AgentGym-RL/textcraft_train.local.sh`** — Idem côté training. Overrides Hydra ajoutés (`n_gpus_per_node=1`, `total_training_steps=1`, `adv_estimator=grpo`, batch sizes minimaux, modèle local, suppression du `model_merger.py`). Conservé pour traçabilité du smoke test verl qui a échoué.

**`WORKLOG.md`** — Journal de bord live, sections par phase, contient tous les commands lancés, les erreurs rencontrées, les diagnostics et les décisions. Lecture obligatoire si tu reprends le projet après une coupure.

**`docs/SESSION_2026-05-05.md`** — Récap autonome de la première grosse session (setup + eval baseline). Format narratif, accessible à un lecteur extérieur.

**`docs/AGENTGYM_RL_TRAINING_DEEPDIVE.md`** — Deep dive technique sur le code de training d'AgentGym-RL (`RayPPOTrainer`, intégration vLLM, `RoundsScheduler` pour ScalingInter, hyperparamètres clés, hooks de customisation, risques single-GPU). Produit avant le pivot pour bien comprendre ce qu'on quittait et identifier ce qu'il faudrait reimplémenter à la main si on en a besoin (ex : le curriculum sur `max_rounds`).
