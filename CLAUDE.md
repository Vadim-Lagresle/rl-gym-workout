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

## Architecture du projet

```
rl-gym-workout/
├── src/                    # scripts principaux (train, eval, analyse, utils)
├── runs/                   # un sous-dossier par expérience (config.yaml + eval_logs)
│   ├── exp1_baseline/
│   ├── exp2_grpo_v2/
│   ├── exp3_grpo_v3/
│   ├── exp4_grpo_v4_scalinginter/
│   ├── exp6_verl_4gpu/
│   └── prototypes/         # scripts d'exploration archivés
├── external/               # tout le code qu'on n'a pas écrit
│   ├── AgentGym/           # submodule — envs + clients HTTP TextCraft
│   ├── AgentGym-RL/        # framework verl (trainer, rollout, PPO/GRPO)
│   ├── agentgym_rl_paper/  # scripts de référence du papier (eval/ + train/)
│   └── USAGE.md            # quels fichiers on utilise réellement dans external/
├── data/                   # datasets train/test par env (JSON)
├── docs/                   # RESULTS.md, guides techniques, references/ (PDFs)
├── saves/                  # checkpoints locaux — gitignorés (trop lourds)
├── models/                 # modèle de base Qwen — gitignorés
├── setup/                  # requirements + setup.sh
└── WORKLOG.md              # journal chronologique de session
```

**Fichiers clés à connaître :**
- `src/train/train_grpo.py` — **script d'entraînement actif** (TRL GRPO, toujours utiliser celui-là)
- `src/eval/eval_vllm.py` — **script d'évaluation actif** (vLLM via serveur HTTP, ~15 min/100 items)
- `src/utils/start_vllm_server.sh` — démarre le serveur vLLM sur un checkpoint (port 8001)
- `src/eval/eval_fullft.py` — évaluation lente via HuggingFace generate() — **NE PAS UTILISER**, remplacé par eval_vllm.py
- `runs/expN_*/config.yaml` — config, hyperparamètres et résultats de chaque run
- `external/USAGE.md` — quels fichiers on utilise dans les dépendances externes
- `docs/RESULTS.md` — tableau de résultats structuré (référence)
- `docs/GERRIT_WORKFLOW.md` — comment pousser le snapshot hebdo sur Gerrit (`research/vadim-lagresle/`)
- `WORKLOG.md` — contexte de session, procédure de reprise

## Résultats actuels (résumé)

| Run | Pass@1 /100 | Note |
|---|---|---|
| Baseline Qwen2.5-3B (0 training) | 18 | référence |
| GRPO v2 step10 (shaping bugué) | 18 | =baseline |
| GRPO v2 step50 (shaping bugué) | 14 | régression |
| GRPO v3 step50 (shaping corrigé) | 8 | régression |
| GRPO v4 ScalingInter sparse 0/1 | 14 | régression |
| **Papier AgentGym-RL-3B** | **75** | objectif |


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

**Toujours utiliser `eval_vllm.py` + serveur vLLM, jamais `eval_fullft.py` directement.**

vLLM est opérationnel sur ce serveur (glibc 2.28) grâce à la version 0.9.1 manylinux1 du mirror
Criteo PyPI + 2 patches Python. Voir `docs/hebdo/5juin/session_2026-06-01.md` pour les détails.

```bash
# Étape 1 — lancer le serveur vLLM sur le checkpoint à évaluer
bash src/utils/start_vllm_server.sh saves/trl_grpo/<run>/checkpoint-<N>

# Étape 2 — eval (100 items, ~15 min)
python src/eval/eval_vllm.py \
    --model saves/trl_grpo/<run>/checkpoint-<N> \
    --run-name <run_name>

# Arrêter le serveur après
kill $(cat /tmp/vllm_server.pid)
```

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
- exp9 (curriculum) nécessite `data/train/textcraft_train_with_depth.json` généré par `src/utils/label_depths.py` dans l'env `agentenv-textcraft`.
