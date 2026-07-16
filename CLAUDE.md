# CLAUDE.md — rl-gym-workout


## Contexte du projet

Travail de recherche sur les **agents LLM self-improving par RL multi-tour**.

**Question centrale** : pourquoi les performances s'effondrent à depth 4 sur TextCraft, et
comment y remédier avec les outils modernes (curriculum, SCPO, BOND, CoT, mix SFT+RL) ?

## Stack technique

| Composant | Détail |
|---|---|
| Modèle base | Qwen2.5-3B-Instruct (cible paper) |
| Framework RL | **TRL GRPO + vLLM** (stack principale, toutes nouvelles expés) |
| Framework RL legacy | verl (fork AgentGym-RL) — expés 1-6 uniquement, ne pas utiliser pour les nouvelles |
| Env benchmark | TextCraft via serveur HTTP FastAPI (port 36005) |
| GPU VM | **B200 192 Go HBM3e** (single GPU, nouvelles expés) — ex-A100 40 Go pour les anciens runs |
| Envs conda | `agentgym-rl` (entraînement TRL), `agentenv-textcraft` (serveur jeu + label_depths.py) |

## Architecture du projet (refacto 2026-07-16)

```
rl-gym-workout/
├── src/
│   ├── train/              # entraînement — 8 modules courts + 2 points d'entrée
│   │   ├── train_grpo.py         # ENTRYPOINT GRPO (CLI + assemblage)
│   │   ├── train_grpo_snis.py    # ENTRYPOINT SNIS (parser partagé + --snis-*)
│   │   ├── rollout.py            # LA boucle env par tour + contrat TRL + reward
│   │   ├── snis.py               # recombinaison SNIS (config, scoring, selftest)
│   │   ├── vllm_engine.py        # moteur vLLM in-process + sync de poids
│   │   ├── periodic_eval.py      # éval test périodique + save du best
│   │   ├── data.py / schedules.py / diagnostics.py
│   │   └── run_curriculum_staged.sh, run_benchmark_baselines.sh
│   ├── eval/
│   │   ├── eval_textcraft.py     # ENTRYPOINT éval multi-tour (--backend auto|vllm|hf)
│   │   ├── eval_oracle.py        # pass@k best-of-N (2 backends, passes résumables)
│   │   ├── textcraft_common.py   # boucle épisode + logs + pass_at_k (source unique)
│   │   ├── llm_chat.py           # ChatGenerator (serveur vLLM | HF in-process)
│   │   ├── single_turn/          # pipeline exp16 (collect + replay des plans)
│   │   └── api/                  # bornes SOTA (Gemini, OpenAI-compatible)
│   ├── analysis/           # analyze_eval.py (taxonomie erreurs), dashboards, plots oracle
│   └── utils/              # start_vllm_server.sh, merge_lora.py, label_depths.py, vllm_supports.py
├── runs/                   # registre expérimental par FAMILLE — voir runs/INDEX.md
│   ├── 0_baselines/  1_api/  2_grpo_fullft/  3_scalinginter/  4_curriculum/
│   ├── 5_lora_warmstart/  6_snis/  7_oracle/
│   ├── 8_single_turn_exp16/    # + README.md (les 4 axes exp16)
│   └── 9_legacy_pre_b200/      # exp2-6 (verl/A100) + prototypes
├── archive/                # code retiré du chemin actif (README par sous-dossier)
├── external/               # code externe (AgentGym, verl, scripts papier) + USAGE.md
├── data/                   # datasets train/test par env (JSON)
├── docs/                   # RESULTS.md, hebdo/, dashboard/, archive/
├── saves/ · models/        # checkpoints et poids — gitignorés
└── WORKLOG.md              # journal (historique ≤ juin ; le suivi vit dans docs/hebdo/)
```

**Fichiers clés à connaître :**
- `src/train/train_grpo.py` — **entraînement GRPO** (TRL) ; `train_grpo_snis.py` pour la variante SNIS
- `src/eval/eval_textcraft.py` — **évaluation active** : `--backend vllm` (serveur, rapide) ou `hf` (universel, ex. Qwen3.5)
- `src/utils/start_vllm_server.sh` — démarre le serveur vLLM sur un checkpoint (port 8001)
- `runs/INDEX.md` — registre de TOUS les runs (familles, scores, statuts)
- `runs/<famille>/<run>/config.yaml` — config, hyperparamètres et résultats de chaque run
- `external/USAGE.md` — quels fichiers on utilise dans les dépendances externes
- `docs/RESULTS.md` — tableau de résultats structuré (référence)
- `docs/GERRIT_WORKFLOW.md` — comment pousser le snapshot hebdo sur Gerrit (`research/vadim-lagresle/`)

## Résultats actuels (résumé — détail dans runs/INDEX.md et docs/RESULTS.md)

| Run | Pass@1 /100 | Note |
|---|---|---|
| Baseline Qwen2.5-3B (0 training) | 18 | référence |
| exp6 verl 4×A100 full-FT | 38 | meilleur pré-B200 (stack legacy) |
| exp9 curriculum depth | 14 | régression — mur depth≥3 = capacité |
| **exp10.8 LoRA warm-start (best du projet)** | **54** | re-éval indépendante (58 = pic bruité) |
| exp17 SNIS v1 | 15 | interrompu step ~53 (crash vLLM) |
| **Papier AgentGym-RL-3B** | **75** | objectif |
| Qwen3.5-4B sans training | 77 | dépasse l'objectif sans RL |
| Gemini 3.5 Flash (API) | 99 | borne haute SOTA |


## Commandes de démarrage de session

```bash
# Panneau 1 — serveur TextCraft
conda activate agentenv-textcraft
textcraft --host 127.0.0.1 --port 36005

# Panneau 2 — entraînement ou eval
conda activate agentgym-rl
cd ~/rl-gym-workout
```

## Stack d'évaluation (à utiliser systématiquement)

**Un seul point d'entrée : `src/eval/eval_textcraft.py`.** Le backend se choisit par
`--backend` : `vllm` (serveur, KV cache, ~15 min/100 items — checkpoints 3B standard)
ou `hf` (HuggingFace in-process, lent mais universel — OBLIGATOIRE pour les archis que
vLLM 0.9.1 ne sert pas, ex. Qwen3.5 ; ajouter `--no-thinking`). `auto` choisit seul.

vLLM est opérationnel sur ce serveur (glibc 2.28) grâce à la version 0.9.1 manylinux1 du mirror
Criteo PyPI + 2 patches Python. Voir `docs/hebdo/5juin/session_2026-06-01.md` pour les détails.

```bash
# Étape 1 — lancer le serveur vLLM sur le checkpoint à évaluer (backend vllm seulement)
bash src/utils/start_vllm_server.sh saves/trl_grpo/<run>/checkpoint-<N>

# Étape 2 — eval (100 items, ~15 min) ; --run-name peut préfixer la famille runs/
python src/eval/eval_textcraft.py \
    --model saves/trl_grpo/<run>/checkpoint-<N> \
    --run-name <famille>/<run_name>

# Arrêter le serveur après
kill $(cat /tmp/vllm_server.pid)

# Archi non servable par vLLM (ex. Qwen3.5-4B) : pas de serveur, backend hf
python src/eval/eval_textcraft.py --model models/Qwen3.5-4B --backend hf \
    --no-thinking --run-name <famille>/<run_name>
```

Oracle pass@k (marge exploitable par le RL) : `src/eval/eval_oracle.py`, mêmes backends,
passes résumables (`passes.jsonl`). Pipeline exp16 : `src/eval/single_turn/`.

## Stack d'entraînement (à utiliser systématiquement)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python src/train/train_grpo.py \
    --full-ft \
    --num-generations 8 \
    --max-completion-length 512 \
    --max-items 0 \
    --max-steps <N> \
    --use-vllm-inprocess \
    --run-name <run_name> \
    > logs/<run_name>.log 2>&1 &
```

**Paramètres clés actuels** (validés sur B200, ne pas changer sans raison) :
- `--full-ft` : full fine-tuning (pas LoRA)
- `--use-vllm-inprocess` : vLLM in-process avec KV cache (×6 vs HF generate)
- `optim=adamw_bnb_8bit` : 8-bit Adam (libère ~18 Go vs fp32 Adam)
- `gpu_memory_utilization=0.17` pour vLLM (marge suffisante pour les saves checkpoint ~8.4 Go)
- `attn_implementation=sdpa` : attention optimisée PyTorch (flash-attn bloqué par glibc 2.28)

## Philosophie de travail avec Claude

**Une tâche à la fois, expliquée avant d'être exécutée.**

- Je (Claude) ai un rôle dorénavant d'aide, de conseil, d'explication, de proposition de pistes, mais plus de leader sur toute une stratégie, et ce certainement pas sur les aspects scientifiques. Pour du changement de code, mon rôle est de bien t'expliquer les fichiers, les fonctions, les libraiires, l'architecture et les appels à fonction. Toujours privilégier la pédagogie à un lead perso sur un problème. 
- Je (Claude) propose ce que je vais faire et pourquoi, tu  valides avant que je code.
- On ne fait pas de grosse refacto ou d'abstraction sans raison explicite.
- Les scripts restent simples, lisibles, avec le minimum de dépendances.
- Quand je modifie quelque chose, je dis exactement quelle ligne change et pourquoi.
- On ne lance pas une expérience sur GPU avant d'avoir validé le code "à sec" (dry-run ou smoke test CPU).

### Mode d'explication obligatoire (pédagogie avant tout)

- Claude est un **assistant d'aide** : il explique de manière organisée, il ne balance
  pas des problèmes scientifiques et techniques en 3 lignes par problème.
- Chaque problème (technique ou scientifique) est introduit **dans son contexte** :
  1. de quelle fonction il s'agit, d'où elle est appelée / de qui elle hérite ;
  2. quelle est son influence sur le reste du système ;
  3. son code, cité avec fichier:lignes (exhaustif sur le passage concerné) ;
  4. enfin le problème principal à soulever.
- Pour tout travail de fond (review, design, debug), Claude commence par écrire un
  **plan des points à couvrir**, puis on avance point par point sur plusieurs échanges,
  en rappelant à chaque échange où on en est du plan de review.
- Ne pas tout coder / tout faire d'un coup : rien ne sert de produire si l'utilisateur
  ne comprend pas. Une étape à la fois, comprise et validée avant la suivante.

### Traçabilité des modifications (obligatoire à chaque prompt)

À la fin de **chaque réponse** où j'ai exécuté du code, Claude fournit une section
**« Trace des modifications »** en français, pédagogique, qui récapitule :

1. **Chaque commande bash exécutée** qui a un effet (lancement de run, kill, merge,
   déplacement de fichiers…) — les commandes de lecture pure (ls, cat, grep) peuvent
   être omises ou résumées en une ligne.
2. **Chaque fichier créé ou modifié**, avec les lignes de code principales (extrait,
   pas le fichier entier) et une phrase expliquant *pourquoi* ce changement.

Le but : garder une trace exploitable de tout ce qui a été fait, reconstruisible
depuis la trace de raisonnement, sans avoir à relire les tool calls. Format type :

```markdown
## Trace des modifications
### Commandes bash
- `nohup python src/train/train_grpo.py ... &` — lancement du run exp10.8
### Fichiers modifiés
- `src/train/train_grpo.py:746` — `default=1e-6` → explication du changement
```

## Directions de recherche en cours

1. **Comprendre depth 4** — isoler pourquoi 0/100 même après training (papier Table 3)
2. **Répliquer le papier** — run trl qui reproduit leur code verl multi-GPU avec les vrais hyperparamètres (N=8, Full FT, FSDP)
3. **Idées à tester** (dans l'ordre croissant de complexité) :
   - CoT structuré au tour 1 (plan explicite avant les actions)
   - Curriculum de difficulté (depth 1→2→3→4)
   - SCPO pour générer plus de données d'entraînement synthétiques
   - Mix SFT + RL (ablation : N exemples SFT vs N steps RL)
   - BOND pour distiller Best-of-N dans le modèle

## Contraintes importantes

- Pas de force-push, pas de commit sans demander.
- Toujours vérifier le code **avant** de lancer sur les gros GPU (B200 192 Go) — les ressources sont rares.
- `gcloud compute config-ssh` écrase parfois le `RemoteForward 8443` dans `~/.ssh/config` sur le Mac.
- Le remote GitLab pointe sur `https://gitlab.crto.in:8443/v.lagresle/rl-gym-workout.git` (tunnel SSH requis).
- Push hebdo sur **Gerrit** (en plus du daily GitLab) : clone séparé dans `~/ai-agentic-commerce-incubation/`, sous-dossier `research/vadim-lagresle/`. Détails dans `docs/GERRIT_WORKFLOW.md`.
- Les nouvelles expés (exp7+) tournent sur **B200** avec **TRL + vLLM** — ne pas utiliser verl pour les nouveaux runs.
- Le curriculum par depth nécessite `data/train/textcraft_train_with_depth.json` généré par `src/utils/label_depths.py` (env `agentenv-textcraft` ; `--split test` pour le fichier test).
- Le code obsolète vit dans `archive/` (README par sous-dossier : raison + remplaçant) — ne pas l'importer depuis `src/`.
- Serveur TextCraft : à lancer depuis `external/AgentGym/agentenv-textcraft/` (chemin relatif `agentenv_textcraft/recipes/`).
