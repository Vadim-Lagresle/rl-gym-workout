# CLAUDE.md — rl-gym-workout

## Briefing automatique de début de session

**A chaque nouvelle conversation, faire ceci en premier sans attendre que Vadim le demande :**
1. Lire `WORKLOG.md`
2. Lire `docs/RESULTS.md`
3. Produire un rapport court (10-15 lignes max) avec : dernière expérience connue, résultat, prochaine étape identifiée, et une question ou point de vigilance si pertinent.


## Contexte du projet

Stage de recherche (2026, Criteo CAIL) sur les **agents LLM self-improving par RL multi-tour**.
Encadrants : Alberto Lumbreras, Patrick Gallinari.

**Question centrale** : pourquoi les performances s'effondrent à depth 4 sur TextCraft, et
comment y remédier avec les outils modernes (curriculum, SCPO, BOND, CoT, mix SFT+RL) ?

## Stack technique

| Composant | Détail |
|---|---|
| Modèle base | Qwen2.5-3B-Instruct (cible paper) |
| Framework RL | verl (fork AgentGym-RL) + TRL GRPO pour les expés légères |
| Env benchmark | TextCraft via serveur HTTP FastAPI (port 36005) |
| GPU VM | A100 40 Go (single ou multi via GCP) |
| Envs conda | `agentgym-rl` (entraînement), `agentenv-textcraft` (serveur jeu) |

## Architecture du projet

```
rl-gym-workout/
├── AgentGym/           # submodule — envs + clients HTTP
├── AgentGym-RL/        # submodule — verl fork (trainer, rollout, PPO/GRPO)
├── AgentEval/          # datasets train/test par env (JSON)
├── scratch/            # scripts numérotés 01_…10_ + utilitaires
├── examples/           # eval/ et train/ — scripts de référence papier
├── docs/               # RESULTS.md (résultats structurés), guides techniques
├── outputs/            # checkpoints par date
├── saves/              # sauvegardes modèles LoRA
├── setup/              # requirements + setup.sh
└── WORKLOG.md          # journal chronologique de session
```

**Fichiers clés à connaître :**
- `scratch/07_trl_grpo_textcraft_smoke.py` — script principal d'entraînement (toutes les versions v2→v4)
- `scratch/03_eval_qwen.py` / `scratch/08_eval_qwen_lora.py` — évaluation baseline et LoRA
- `docs/RESULTS.md` — tableau de résultats structuré (référence)
- `WORKLOG.md` — contexte de session, procédure de reprise
- `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` — rollout multi-tour
- `AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py` — boucle PPO + ScalingInter

## Résultats actuels (résumé)

| Run | Pass@1 /100 | Note |
|---|---|---|
| Baseline Qwen2.5-3B (0 training) | 18 | référence |
| GRPO v2 step10 (shaping bugué) | 18 | =baseline |
| GRPO v2 step50 (shaping bugué) | 14 | régression |
| GRPO v3 step50 (shaping corrigé) | 8 | régression |
| GRPO v4 ScalingInter sparse 0/1 | 14 | régression |
| **Papier AgentGym-RL-3B** | **75** | objectif |

**Gap de 57 points** dû principalement à : LoRA vs Full FT, N=2 vs N=8 (→ 70% des steps
sans gradient à N=2 et p≈0.18), batch=1 vs 32, max_tokens=128 vs 512.

## Commandes de démarrage de session

```bash
# Panneau 1 — serveur TextCraft
conda activate agentenv-textcraft
textcraft --host 127.0.0.1 --port 36005

# Panneau 2 — entraînement ou eval
conda activate agentgym-rl
cd ~/rl-gym-workout
```

## Philosophie de travail avec Claude

**Une tâche à la fois, expliquée avant d'être exécutée.**

- Je (Claude) propose ce que je vais faire et pourquoi, tu valides avant que je code.
- On ne fait pas de grosse refacto ou d'abstraction sans raison explicite.
- Les scripts restent simples, lisibles, avec le minimum de dépendances.
- Quand je modifie quelque chose, je dis exactement quelle ligne change et pourquoi.
- On ne lance pas une expérience sur GPU avant d'avoir validé le code "à sec" (dry-run ou smoke test CPU).

## Directions de recherche en cours

1. **Comprendre depth 4** — isoler pourquoi 0/100 même après training (papier Table 3)
2. **Répliquer le papier** — run verl multi-GPU avec les vrais hyperparamètres (N=8, Full FT, FSDP)
3. **Idées à tester** (dans l'ordre croissant de complexité) :
   - CoT structuré au tour 1 (plan explicite avant les actions)
   - Curriculum de difficulté (depth 1→2→3→4)
   - SCPO pour générer plus de données d'entraînement synthétiques
   - Mix SFT + RL (ablation : N exemples SFT vs N steps RL)
   - BOND pour distiller Best-of-N dans le modèle

## Contraintes importantes

- Pas de force-push, pas de commit sans demander.
- Toujours vérifier le code **avant** de lancer sur les gros GPU (A100/B200) — les ressources sont rares.
- `gcloud compute config-ssh` écrase parfois le `RemoteForward 8443` dans `~/.ssh/config` sur le Mac.
- Le remote GitLab pointe sur `https://gitlab.crto.in:8443/v.lagresle/rl-gym-workout.git` (tunnel SSH requis).
