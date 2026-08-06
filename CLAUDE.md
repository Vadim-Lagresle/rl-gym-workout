# CLAUDE.md — rl-gym-workout


## Contexte du projet

Travail de recherche sur les **agents LLM self-improving par RL multi-tour**.

**Question centrale** : pourquoi les performances s'effondrent à depth 4 sur TextCraft, et
comment y remédier avec les outils modernes (curriculum, SCPO, BOND, CoT, mix SFT+RL) ?

## Stack technique

| Composant | Détail |
|---|---|
| Modèle base | Qwen2.5-3B-Instruct (cible paper) — **sur `/tmp/models/` depuis 2026-07-29** (volatil, retélécharger via `setup/ensure_qwen_tmp.sh`) |
| Framework RL | **TRL ≥1.9 GRPO + vLLM ≥0.25 colocate + flash-attention** (stack v2, 2026-07-22) |
| Framework RL legacy | verl (exp1-6) ; TRL 1.4/vLLM 0.9.1 (exp7-19.2 — env rollback supprimé le 2026-07-29) |
| Env benchmark | TextCraft via serveur HTTP FastAPI (port 36005) |
| GPU VM | **B200 192 Go HBM3e** (single GPU) — VM migrée CentOS Stream 10 / glibc 2.39 (2026-07-20) |
| Envs Python | **v2 : `/tmp/envs/agentgym-rl-v2`** (⚠ /tmp volatil — reconstruire via `setup/setup_agentgym_rl_v2.sh`, ~10 min grâce au wheel flash-attn dans `saves/wheels/` ; base pyenv 3.11.7) ; `~/envs/agentenv-textcraft` (serveur jeu, home) |
| Disque home (35 Go) | Politique 2026-07-29 : ne stocker QUE les best adapters (+ optimizer) — `saves/keep_best/` (archive, voir son README), `saves/trl_grpo/<run>_besttrain` (runs en cours), `saves/wheels/`. Modèles de base et checkpoints périodiques sur `/tmp` (retéléchargeables/reconstructibles) |

## Architecture du projet (refacto 2026-07-16)

```
rl-gym-workout/
├── src/
│   ├── train/              # entraînement — 8 modules courts + 2 points d'entrée
│   │   ├── train_grpo.py         # ENTRYPOINT GRPO (CLI + assemblage)
│   │   ├── train_grpo_snis.py    # ENTRYPOINT SNIS (parser partagé + --snis-*)
│   │   ├── rollout.py            # LA boucle env par tour + contrat TRL + reward
│   │   ├── snis.py               # recombinaison SNIS (config, scoring, selftest)
│   │   ├── vllm_engine.py        # accès au moteur vLLM colocate de TRL + save modèle
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
# Panneau 1 — serveur TextCraft (depuis le dossier du package : chemin relatif recipes/)
source ~/envs/agentenv-textcraft/bin/activate
cd ~/rl-gym-workout/external/AgentGym/agentenv-textcraft
textcraft --host 127.0.0.1 --port 36005

# Panneau 2 — entraînement ou eval (env v2 + modèle ; les reconstruire si /tmp a été purgé)
[ -d /tmp/envs/agentgym-rl-v2 ] || bash ~/rl-gym-workout/setup/setup_agentgym_rl_v2.sh
bash ~/rl-gym-workout/setup/ensure_qwen_tmp.sh    # Qwen2.5-3B → /tmp/models (no-op si présent)
export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"   # requis : les sous-process vLLM cherchent ninja dans le PATH
cd ~/rl-gym-workout
```

## Stack d'évaluation (à utiliser systématiquement)

**Un seul point d'entrée : `src/eval/eval_textcraft.py`.** Le backend se choisit par
`--backend` : `vllm` (serveur, KV cache, rapide — Qwen3.5 inclus depuis la stack v2)
ou `hf` (HuggingFace in-process, lent mais universel, repli sans serveur ;
`--no-thinking` pour Qwen3/3.5). `auto` choisit seul.

Depuis la migration VM (glibc 2.39, 2026-07-22), la stack v2 utilise une vLLM récente
standard — les archis récentes (Qwen3.5…) sont servables. Pour servir avec l'env v2 :
`PYTHON=/tmp/envs/agentgym-rl-v2/bin/python bash src/utils/start_vllm_server.sh <modèle>`.
(Historique glibc 2.28 / vLLM 0.9.1 patchée : `docs/hebdo/5juin/session_2026-06-01.md`.)

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

**Toujours lancer avec `setsid nohup ... &`** (leçon 2026-07-30) : la fermeture
de la session Cursor tue le groupe de processus ~45 min après — `nohup` seul ne
protège pas (SIGHUP seulement), `setsid` détache le run dans sa propre session.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python src/train/train_grpo.py \
 --full-ft \
    --num-generations 8 \
    --max-completion-length 512 \
    --max-items 0 \
    --max-steps <N> \
    --use-vllm-inprocess \
    --run-name <run_name> \
    > logs/<run_name>.log 2>&1 &
```

**Paramètres clés actuels** (stack v2 validée par smokes le 2026-07-22) :
- `--full-ft` : full fine-tuning (pas LoRA)
- `--use-vllm-inprocess` : moteur vLLM colocate GÉRÉ PAR TRL — sync des poids EN MÉMOIRE
  par TRL à chaque step (PEFT inclus), plus d'écriture disque ni de recréation de moteur
- `optim=adamw_bnb_8bit` : 8-bit Adam (libère ~18 Go vs fp32 Adam) — auto en full-ft
- `--vllm-gpu-util 0.17` : marge validée sur B200 pour le 3B
- `attn_implementation=flash_attention_2` (défaut ; `--attn-implementation sdpa` = ancien repli)
- `vllm_importance_sampling_correction=False` figé dans le code : parité de sémantique avec la
  lignée exp10/19 (on-policy ratio ≡ 1) — à réactiver seulement comme ablation contrôlée

### Règles d'entraînement (politique 2026-07-29)

**1. Persistance : ne sauver sur le home QUE les best (+ optimizers)**

| Artefact | Sélection | Emplacement | Taille typique (LoRA r=64) |
|---|---|---|---|
| **Best test** | Pass@1 périodique sur le test set (`TestEvalCallback`) | `saves/trl_grpo/<run>_best` | ~480 Mo (adapter seul) |
| **Best train** | max de la moyenne roulante du reward TRAIN (`RewardAdaptiveLrCallback`) | `saves/trl_grpo/<run>_besttrain` | ~480 Mo adapter + ~960 Mo `optimizer.pt` |
| Archive des best clos | — | `saves/keep_best/` (README par run) | idem |

- Checkpoints périodiques (`--output-root /tmp/trl_grpo_runs`) : sur `/tmp`, volatils — ne
  **pas** compter dessus pour reprendre après purge pod.
- Modèle de base : `/tmp/models/` (retélécharger via `setup/ensure_qwen_tmp.sh`).
- À chaque nouveau best train, l'adapter est remplacé (swap atomique `.tmp` → dir) ;
  `.besttrain_info` trace step/epoch/mean/lr/beta.
- **Toujours activer `--lr-adaptive-save-optimizer`** sur les runs adaptatifs LoRA :
  l'état Adam (moments fp32) est sauvé avec le best et **restauré à chaque coupe de LR**
  (poids + moments cohérents, reprise exacte après coupure infra). Sans ce flag, les
  moments sont remis à zéro au restore (exp22.1).

**2. Stabilisation LR + beta KL — trois modes (exclusifs)**

Tous divisent LR et beta par le même facteur. Source unique :
`apply_lr_beta()` dans `src/train/schedules.py`.

| Mode | CLI | Quand couper | Leçon |
|---|---|---|---|
| **Paliers calendaires** | `--lr-stage-every-epochs 3` | toutes les N epochs, quoi qu'il arrive | exp20 : stabilise, pic 58, mais coupe des dynamiques en cours ; exp22.4 : consolide la dérive post-pic |
| **Adaptatif au reward train** | `--lr-adaptive` (+ flags ci-dessous) | si la moyenne roulante stagne sous la référence pendant `patience` checks | exp22/22.1 : restore utile, mais coupes sur bruit si critère trop laxiste → exp22.2 ; exp22.3 : coupe APRÈS le décrochage |
| **Calendaire long + restore du best de palier** | `--lr-stage-every-epochs 10 --lr-stage-restore-best` | toutes les 10 epochs, en REPARTANT du best du palier (poids + optimizer, `<run>_stagebest`) ; best global promu dans `<run>_besttrain` | exp22.5 (2026-07-31) : la dérive de fin de palier est jetée au lieu d'être consolidée |

**Run adaptatif standard (exp22.2 — à utiliser pour les nouveaux runs LoRA few-shot) :**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python src/train/train_grpo.py \
 --lora-r 64 \
    --num-generations 8 --gradient-accumulation-steps 64 \
    --max-completion-length 512 --max-items 0 \
    --num-epochs 100 --learning-rate 5e-6 --beta 0.01 \
    --lr-adaptive --lr-adaptive-restore-best --lr-adaptive-save-optimizer \
    --lr-adaptive-eps 0.02 --lr-adaptive-ref-median-k 5 \
    --lr-adaptive-min-stage-epochs 3 \
    --lr-adaptive-min-lr 1e-7 --lr-adaptive-stop-at-floor \
    --fewshot 10 --vllm-max-len 20480 \
    --eval-every 50 --eval-items 100 \
    --save-steps 47 --save-total-limit 20 \
    --output-root /tmp/trl_grpo_runs \
    --use-vllm-inprocess --run-name <run_name> \
    > logs/<run_name>.log 2>&1 &
```

Mécanique adaptative (détail dans `RewardAdaptiveLrCallback`, `schedules.py`) :
- **Fenêtre** : moyenne roulante du reward train sur 1 epoch (~47 steps × 64 traj).
- **Check** : toutes les ½ epoch. **Patience** : 3 checks consécutifs sous la référence.
- **Référence de stagnation** : médiane des 5 derniers checks du palier (`--lr-adaptive-ref-median-k 5`) — pas le max historique (biaisé +1-2σ, leçon exp22.1).
- **Tolérance** : `eps=0.02` (~2σ du bruit inter-fenêtres mesuré).
- **Palier minimum** : `--lr-adaptive-min-stage-epochs 3` — une coupe déclenchée avant est *retenue* ; exécutée au premier check suivant si la stagnation persiste (vraies mesures stables par palier).
- **À la coupe** : LR et beta ÷3 → **restore des poids du best train** (+ optimizer si activé) → cooldown (historique vidé, prochain check après 1 fenêtre complète post-coupe). TRL resync vLLM au step suivant.
- **Plancher** : `--lr-adaptive-min-lr 1e-7 --lr-adaptive-stop-at-floor` — arrêt propre du run (plus de epochs à LR≈0).
- Le best train est 100 % TRAIN : aucun peeking test ; le Pass@1 périodique reste un estimateur honnête.

**Surveillance** (lignes `[reward-adaptive-lr]` dans le log + wandb `adaptive/*`) :
- `best train sauvé` / `poids RESTAURÉS` / `moments Adam restaurés (état du best)`
- `coupe retenue : palier X ep < min 3 ep` = signal avant palier complet
- `>>> COUPE #N` = LR réduit + restore
- KL saine ~0.001 ; dérive soutenue > 0.05 avant la 1re coupe = critère trop laxiste
- Entropie : collapse < 0.15 dès ~step 400 = politique quasi déterministe, starvation d'exploration GRPO

**LoRA** : défaut historique r=16/α=32 ; exp22.2 teste r=64/α=128 (`--lora-r`, `--lora-alpha`).
Ancre KL = Qwen nu (adapter recréé à zéro ou poids injectés via `set_peft_model_state_dict`
sans recréer d'adapter — cf. doc hebdo 21/07 continuation d'adapter).


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
