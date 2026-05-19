# rl-gym-workout

Stage de recherche (2026, Criteo CAIL) sur les **agents LLM self-improving par RL multi-tour**.

**Question centrale** : peut-on entraîner Qwen2.5-3B-Instruct à résoudre des tâches de planification séquentielles (TextCraft) par RL en ligne, et pourquoi les performances s'effondrent à depth 4 ?

Référence : [AgentGym-RL (2025)](docs/references/agentgym_rl_paper.pdf) — Pass@1 = 75/100 avec full fine-tuning, N=8, au moins 4 GPU A100 et Ascend 910B NPUs.

---

## Focus

Le projet se concentre exclusivement sur **TextCraft** : un environnement de crafting Minecraft en texte où l'agent doit assembler des objets en plusieurs étapes via des commandes (`Action: get oak log`, `Action: craft plank`, etc.). Chaque épisode = une recette à résoudre en ≤ 20 tours.

---

## Architecture

```
rl-gym-workout/
├── src/
│   ├── train/train_grpo.py     # Entraînement GRPO (TRL) — script principal
│   └── eval/                   # Évaluation, analyse, visualisation
├── runs/                       # Un dossier par expérience (config + eval_logs)
├── data/                       # Datasets train/test TextCraft (JSON)
├── models/                     # Qwen2.5-3B-Instruct (gitignored)
├── saves/                      # Checkpoints LoRA (gitignored)
├── external/                   # Code tiers (non modifié)
│   ├── AgentGym/               # Client HTTP TextCraft + serveur de jeu
│   ├── AgentGym-RL/            # Framework verl (PPO/GRPO multi-GPU, à delete)
│   └── agentgym_rl_paper/      # Scripts de référence du papier, pour inspiration
├── docs/                       # Résultats, guides, PDFs
├── setup/                      # requirements + script d'installation
└── WORKLOG.md                  # Journal de session
```

---

## Stack

| Composant | Détail |
|---|---|
| Modèle | Qwen2.5-3B-Instruct |
| Entraînement | TRL 1.3.0 + GRPO + LoRA (notre stack) |
| Génération rapide | vLLM ≥ 0.12.0 |
| Environnement | TextCraft via serveur HTTP FastAPI (port 36005) |
| GPU cible | B200 single GPU (env `trl-b200`) |

---

## Dépendances externes (`external/`)

- **`AgentGym/agentenv`** — package Python installé en editable (`pip install -e`). Fournit `TextCraftEnvClient`, le client HTTP utilisé dans tous nos scripts pour interagir avec le serveur de jeu.
- **`AgentGym/agentenv-textcraft`** — le serveur FastAPI du jeu TextCraft. Tourne dans un env conda séparé (`agentenv-textcraft`), indépendant de notre stack d'entraînement.
- **`AgentGym-RL/`** — fork verl de ByteDance adapté pour les agents multi-tours. Utilisé pour l'Exp 6 (4 GPU) uniquement. Incompatible single GPU.
- **`agentgym_rl_paper/`** — scripts originaux du papier (référence pour reproduire les résultats).

---

## Démarrage rapide

```bash
# 1. Installer l'env d'entraînement (B200)
CUDA=cu126 bash setup/setup_trl_b200.sh

# 2. Lancer le serveur TextCraft (terminal séparé)
conda activate agentenv-textcraft
cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005

# 3. Lancer l'entraînement
conda activate trl-b200
python src/train/train_grpo.py --use-vllm --num-generations 8 --max-steps 200 --run-name exp7_b200
```

Voir `docs/` pour les résultats des expériences et `WORKLOG.md` pour le contexte de session.