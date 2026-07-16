# rl-gym-workout

Stage de recherche (2026, Criteo CAIL) sur les **agents LLM self-improving par RL multi-tour**.

**Question centrale** : peut-on entraîner Qwen2.5-3B-Instruct à résoudre des tâches de planification séquentielles (TextCraft) par RL en ligne, et pourquoi les performances s'effondrent à depth 4 ?

Référence : [AgentGym-RL (2025)](docs/references/agentgym_rl_paper.pdf) — Pass@1 = 75/100 avec full fine-tuning, N=8, au moins 4 GPU A100 et Ascend 910B NPUs.

---

## Focus

Le projet se concentre exclusivement sur **TextCraft** : un environnement de crafting Minecraft en texte où l'agent doit assembler des objets en plusieurs étapes via des commandes (`Action: get oak log`, `Action: craft plank`, etc.). Chaque épisode = une recette à résoudre en ≤ 20 tours.

---

## Architecture (refacto 2026-07-16 — détail dans CLAUDE.md)

```
rl-gym-workout/
├── src/
│   ├── train/                  # GRPO/SNIS : 2 entrypoints + modules (rollout, snis,
│   │                           #   vllm_engine, periodic_eval, data, schedules…)
│   ├── eval/                   # eval_textcraft.py (--backend vllm|hf), eval_oracle.py,
│   │                           #   tronc commun, single_turn/ (exp16), api/ (SOTA)
│   ├── analysis/               # analyze_eval (taxonomie erreurs) + dashboards
│   └── utils/                  # serveur vLLM, merge LoRA, labels depth
├── runs/                       # Registre expérimental par famille — voir runs/INDEX.md
├── archive/                    # Code retiré du chemin actif (README par sous-dossier)
├── data/                       # Datasets train/test TextCraft (JSON)
├── models/ · saves/            # Poids et checkpoints (gitignorés)
├── external/                   # Code tiers (non modifié)
│   ├── AgentGym/               # Client HTTP TextCraft + serveur de jeu
│   ├── AgentGym-RL/            # Framework verl — legacy (exp1-6 uniquement)
│   └── agentgym_rl_paper/      # Scripts de référence du papier
├── docs/                       # RESULTS.md (référence), hebdo/, dashboard/, archive/
├── setup/                      # requirements + script d'installation
└── WORKLOG.md                  # Journal (historique ≤ juin ; suivi dans docs/hebdo/)
```

---

## Stack

| Composant | Détail |
|---|---|
| Modèle | Qwen2.5-3B-Instruct |
| Entraînement | TRL GRPO full-FT/LoRA + vLLM 0.9.1 in-process (sync de poids par step) |
| Éval | serveur vLLM (KV cache) ou HF generate (archis récentes, ex. Qwen3.5) |
| Environnement | TextCraft via serveur HTTP FastAPI (port 36005) |
| GPU | B200 192 Go single GPU (env conda `agentgym-rl`) |

---

## Dépendances externes (`external/`)

- **`AgentGym/agentenv`** — package Python installé en editable (`pip install -e`). Fournit `TextCraftEnvClient`, le client HTTP utilisé dans tous nos scripts pour interagir avec le serveur de jeu.
- **`AgentGym/agentenv-textcraft`** — le serveur FastAPI du jeu TextCraft. Tourne dans un env conda séparé (`agentenv-textcraft`), indépendant de notre stack d'entraînement.
- **`AgentGym-RL/`** — fork verl de ByteDance adapté pour les agents multi-tours. Utilisé pour l'Exp 6 (4 GPU) uniquement. Incompatible single GPU.
- **`agentgym_rl_paper/`** — scripts originaux du papier (référence pour reproduire les résultats).

---

## Démarrage rapide

```bash
# 1. Lancer le serveur TextCraft (terminal séparé)
conda activate agentenv-textcraft
cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005

# 2. Lancer un entraînement (env agentgym-rl)
conda activate agentgym-rl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train/train_grpo.py --full-ft --num-generations 8 \
    --max-completion-length 512 --max-items 0 --max-steps 200 \
    --use-vllm-inprocess --run-name <run_name>

# 3. Évaluer un checkpoint
bash src/utils/start_vllm_server.sh saves/trl_grpo/<run>/checkpoint-<N>
python src/eval/eval_textcraft.py --model saves/trl_grpo/<run>/checkpoint-<N> \
    --run-name <famille>/<run_name>
```

Voir `docs/RESULTS.md` pour les résultats, `runs/INDEX.md` pour le registre des
expériences, et `CLAUDE.md` pour les conventions de travail.