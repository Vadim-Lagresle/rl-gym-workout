# Session 2026-07-22 (suite) — Migration stack v2 : TRL 1.9 + vLLM 0.25.1 colocate + flash-attention

Suite directe de la validation isolée du matin (`session_2026-07-22_vllm_recent_qwen35_valide.md`).
Décision Vadim : migrer tout le pipeline. **Fait, validé par 6 smokes, commité.**

## 1. Ce qui change dans le code (1 module réécrit, 3 retouches)

| Fichier | Avant | Après |
|---|---|---|
| `src/train/vllm_engine.py` | moteur vLLM maison : init manuelle + sync par écriture du modèle complet (~5,8 Go) sur /tmp + destruction/recréation du moteur à CHAQUE step (~15-20 s, 20-25 % du temps de step mesuré sur exp19) | mince adaptateur vers le moteur **colocate géré par TRL** : `get_engine(trainer)`, `generate_round` (même logique/logprobs), `sync_before_eval` (l'éval périodique a lieu à on_step_end, avant la sync paresseuse de TRL), `save_model_for_vllm` conservé |
| `src/train/train_grpo.py` | `use_vllm=False` + moteur maison | `use_vllm=True, vllm_mode="colocate"` (TRL construit ET synchronise le moteur, **en mémoire**, PEFT inclus) ; `attn_implementation=flash_attention_2` par défaut (`--attn-implementation sdpa` = repli) ; `--vllm-gpu-util` exposé ; **`vllm_importance_sampling_correction=False` figé** (parité de sémantique avec la lignée : la correction IS de TRL 1.9 utiliserait nos logprobs de rollout — 0.0 sur les tokens masqués — comme logprobs d'échantillonnage ; à réactiver un jour comme ablation contrôlée, pas par accident) |
| `src/train/rollout.py` | `vllm_engine.LLM_ENGINE` | `vllm_engine.get_engine(trainer)` — la boucle multi-tour est INCHANGÉE |
| `src/train/periodic_eval.py` | moteur global | moteur passé en argument + `sync_before_eval(trainer)` explicite avant chaque éval |

Supprimés : `init_engine`, `sync_trl_to_vllm`, `VllmSyncCallback`, l'état de module.
Le contrat `rollout_func` (prompt_ids/completion_ids/logprobs/env_mask) est identique
entre TRL 1.4 et 1.9 → `rollout.py`, `snis.py`, `data.py` n'ont pas bougé sur le fond.

## 2. Environnement v2

- **`/tmp/envs/agentgym-rl-v2`** : vLLM 0.25.1, TRL 1.9.0, flash-attn 2.8.3.post1,
  torch 2.11.0+cu130, transformers 5.14.1, peft 0.19.1, bitsandbytes 0.49.2, wandb,
  datasets, agentenv (client TextCraft, installé --no-deps).
- ⚠ /tmp est volatil (purgé au restart de pod, cf. exp19) → **`setup/setup_agentgym_rl_v2.sh`**
  (versionné) reconstruit l'env en ~10 min : le wheel flash-attn compilé (50 min de
  compilation) est conservé sur le home (`saves/wheels/flash_attn-2.8.3.post1-*.whl`, 232 Mo).
- **Piège PATH** : les sous-process vLLM V1 (EngineCore) cherchent `ninja` dans le PATH →
  toujours `export PATH="/tmp/envs/agentgym-rl-v2/bin:$PATH"` (documenté dans CLAUDE.md).
- L'ancien env `~/envs/agentgym-rl` (TRL 1.4/vLLM 0.9.1) est **conservé intact en rollback**
  — mais ne peut plus exécuter le code migré (champs GRPOConfig v2) : rollback = git + vieil env.

## 3. Validation — 6 smokes, tous exit 0

| Smoke | Ce qui est exercé | Résultat |
|---|---|---|
| à sec | py_compile src/**, selftest SNIS 7/7, --help ×2 | ✔ |
| A | LoRA + colocate TRL + FA2 + éval périodique + save-best adapter | ✔ éval 3 items en 9,4 s, best sur home |
| B | full-FT + adamw_bnb_8bit + ScalingInter (cap 3 tours) | ✔ |
| C | SNIS M=4/G=2 (scoring, recombinaison) | ✔ diagnostics ESS normaux |
| D | few-shot k=10 (111 tours, chat template transformers 5.14) | ✔ 2/2 épisodes résolus |
| E | chemin HF sans vLLM + schedule epoch | ✔ |
| F | serveur vLLM v2 (`PYTHON=… start_vllm_server.sh`) + `eval_textcraft.py` client ancien env | ✔ 2/2, 0,9 s/épisode |

Observation de vitesse : chaque smoke complet (chargement moteur + 1 step + éval) tourne
en **25-45 s** contre plusieurs minutes avant — la sync en mémoire remplace les ~15-20 s
d'écriture disque par step, et le moteur 0.25.1 démarre et génère plus vite (sm100 natif).
Le gain réel par step sur un vrai run (64 trajectoires) sera mesuré au premier lancement.

## 4. Ce que ça ouvre

- **Qwen3 / Qwen3.5 servables et entraînables** (registre v2) — l'éval de Qwen3.5-4B peut
  abandonner le fallback HF lent.
- **flash-attention** actif par défaut à l'entraînement (`--attn-implementation sdpa` si besoin).
- Le point n°2 des « pistes d'optimisation » du WORKLOG du 17/07 (sync adapter au lieu de
  reload complet) est résolu gratuitement par la migration ; le point n°1
  (per_device_train_batch_size > 1) reste ouvert.

## 5. Points de vigilance avant le premier run long

- Aucun run LONG n'a encore tourné sur la v2 — seulement des smokes 1 step. Le premier
  vrai run (ex. reprise exp19.x) doit être surveillé de près au début (mémoire GPU du
  colocate sur la durée, stabilité de la sync sur des centaines de steps).
- `vllm_importance_sampling_correction=False` : choix de parité assumé, documenté dans le
  code — décision à revisiter explicitement si on veut la version « corrigée ».
- Si /tmp est purgé : `bash setup/setup_agentgym_rl_v2.sh` (~10 min) puis relancer.
