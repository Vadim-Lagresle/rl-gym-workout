"""Entraînement GRPO multi-tour sur TextCraft avec TRL — point d'entrée.

Implémente une boucle RL interactive : à chaque step GRPO, le modèle joue N
épisodes complets (generate → env.step → observe → repeat), reçoit une récompense
sparse 0/1 de l'environnement, et met à jour ses poids par GRPO.

Architecture (refacto 2026-07-16) — ce fichier ne contient que la CLI et
l'assemblage ; la logique vit dans les modules de src/train/ :
    data.py          dataset train + prompts (curriculum depth inclus)
    schedules.py     ScalingInter (budget d'interaction progressif) et lr scheduler
    rollout.py       LA boucle env par tour + mise à plat contrat TRL + reward
    vllm_engine.py   moteur vLLM in-process + sync de poids par step
    periodic_eval.py éval test périodique + save du best checkpoint
    diagnostics.py   télémétrie mémoire GPU
    snis.py          recombinaison SNIS (utilisée par train_grpo_snis.py)

Pré-requis :
  - Serveur TextCraft lancé : source ~/envs/agentenv-textcraft/bin/activate &&
    cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005
  - Env v2 (TRL>=1.9 / vLLM>=0.25 / flash-attn — migration 2026-07-22) :
    /tmp/envs/agentgym-rl-v2, reconstructible via setup/setup_agentgym_rl_v2.sh.
    L'ancien env (~/envs/agentgym-rl, TRL 1.4/vLLM 0.9.1) ne peut PLUS exécuter ce
    script (champs GRPOConfig v2) — rollback possible via l'historique git.

Usage (commande de référence, voir CLAUDE.md ; --help pour tous les arguments) :
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_grpo.py --full-ft --num-generations 8 \
        --max-completion-length 512 --max-items 0 --max-steps 200 \
        --use-vllm-inprocess --run-name <run_name>

Nota : --use-vllm-inprocess active le moteur vLLM colocate GÉRÉ PAR TRL
(use_vllm=True, vllm_mode="colocate") — sync de poids en mémoire par TRL à
chaque step, PEFT inclus. L'ancien moteur maison (écriture disque + recréation,
~20-25 % du temps de step) a été retiré à la migration du 2026-07-22.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo → imports src.*

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.train import schedules, vllm_engine
from src.train.data import DEFAULT_MODEL_PATH, DEFAULT_SYSTEM_PROMPT, REPO_ROOT, \
    build_prompt_rows, check_server
from src.train.diagnostics import MemDiagCallback
from src.train.periodic_eval import TestEvalCallback
from src.train.rollout import grpo_rollout_func, textcraft_reward


class WarmStartOptimizerCallback(TrainerCallback):
    """Charge l'état d'optimiseur d'un warm-start AU DÉBUT du train.

    L'optimiseur n'existe pas encore à la construction du trainer : HF Trainer le
    crée dans train(), juste avant de déclencher on_train_begin — d'où ce callback.
    bnb 8-bit : load_state_dict est surchargé par bitsandbytes et préserve les dtypes
    des états quantifiés (uint8 + stats fp32), même mécanique que la reprise de
    checkpoint HF standard.
    """

    def __init__(self, optimizer_path: str) -> None:
        self.optimizer_path = optimizer_path
        self.trainer_ref = None  # injecté après la création du GRPOTrainer

    def on_train_begin(self, targs, state, control, **kwargs):
        import torch
        sd = torch.load(self.optimizer_path, map_location="cpu", weights_only=False)
        self.trainer_ref.optimizer.load_state_dict(sd)
        n_states = len(sd.get("state", {}))
        print(f"[warm-start] moments Adam chargés ({n_states} états de paramètres) "
              f"depuis {self.optimizer_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=float, default=0,
                        help="Nb d'epochs (passes complètes sur le dataset). Si > 0, "
                             "prioritaire sur --max-steps (qui passe alors à -1).")
    parser.add_argument("--num-generations", type=int, default=2,
                        help="Rollouts GRPO par prompt (N). Papier TextCraft: 8.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64,
                        help="Micro-steps avant 1 optimizer step. Trajectoires/step ≈ "
                             "grad_accum ; prompts/step = grad_accum / num_generations. "
                             "Papier TextCraft: 256 (= 32 prompts × N=8).")
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Max tokens per assistant turn (128 = smoke test, 512 = proper run).")
    parser.add_argument("--steps-per-generation", type=int, default=0,
                        help="Taille du buffer de rollouts en trajectoires (mode « vrai PPO », "
                             "exp23). 0 = défaut TRL (= grad_accum : 1 génération par optimizer "
                             "step, strictement on-policy, ratio ≡ 1). Si > grad_accum : TRL "
                             "génère steps_per_generation trajectoires d'un coup, calcule les "
                             "old_per_token_logps sous le modèle PRÉ-updates, puis consomme le "
                             "buffer en steps_per_generation/grad_accum optimizer steps CLIPPÉS "
                             "(epsilon 0.2) — la mécanique mini-batch de verl (ppo_mini_batch_size "
                             "× n = 64 traj/update, batch 256). Doit être un multiple de grad_accum.")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="Coefficient du bonus d'entropie dans la loss (loss -= coef × "
                             "entropie moyenne par token actif). verl/papier : 0.001 "
                             "(dp_actor.py:253). 0 = off (défaut, sémantique exp10-22).")
    parser.add_argument("--full-ft", action="store_true", default=False,
                        help="Full fine-tuning (no LoRA). Requires more VRAM — use on B200.")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="Rang LoRA (défaut 16 = lignée exp10-22.1). exp22.2 : 64 — plus de "
                             "capacité d'adaptation, adapter ~4x plus gros (~500 Mo).")
    parser.add_argument("--lora-alpha", type=int, default=0,
                        help="Alpha LoRA (0 = auto : 2×r, le ratio de la lignée exp10).")
    parser.add_argument("--moving-anchor-every-epochs", type=float, default=0,
                        help="LoRA uniquement (exp25) : ancre KL MOBILE par merge-and-restart. "
                             "Toutes les N epochs : snapshot de l'adapter (chaîne reconstructible "
                             "dans saves/trl_grpo/<run>_anchors), fusion dans la base, adapter "
                             "réinitialisé (A=kaiming, B=0), moments Adam LoRA purgés. La KL "
                             "mesure alors la distance au DERNIER ré-ancrage, plus à Qwen nu. "
                             "0 = off (ancre fixe historique). Détail : schedules.MovingAnchorCallback.")
    parser.add_argument("--moving-anchor-initial-cycle", type=int, default=0,
                        help="Reprise (--resume-from-checkpoint) d'un run à ancre mobile : nombre "
                             "de ré-ancrages DÉJÀ effectués par le run d'origine (cf. chain.jsonl). "
                             "Prochain ré-ancrage à l'epoch (N+1)×every. 0 = run neuf.")
    parser.add_argument("--beta", type=float, default=0.001,
                        help=(
                            "Coefficient de la penalite KL (ancrage a la reference). "
                            "Defaut 0.001 (aligne verl/papier, OK en full-ft LR=1e-6). "
                            "En LoRA + LR eleve, monter a ~0.01 pour eviter le runaway KL "
                            "qui a fait diverger exp10."
                        ))
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help=(
                            "LR optimizer. Défaut 1e-6 (aligné full-ft / papier). En LoRA, "
                            "le blog Thinking Machines 'LoRA Without Regret' recommande ~10x "
                            "le LR full-ft -> 1e-5 (voire 1.5e-5 pour les runs <100 steps)."
                        ))
    parser.add_argument("--lr-scheduler-type", type=str, default="constant",
                        choices=["constant", "constant_with_warmup", "linear",
                                 "cosine", "cosine_with_restarts"],
                        help=(
                            "Scheduler LR HF Trainer. Défaut 'constant' (aligné verl/papier, "
                            "cf. audit 11 juin 2026 : le défaut HF 'linear' décroît vers 0 et "
                            "divise le LR moyen par ~2 sur un run). Incompatible avec "
                            "--lr-stage-every-epochs (qui suppose un multiplicateur de schedule "
                            "constant ≡ 1)."
                        ))
    parser.add_argument("--optim", type=str, default="",
                        help=(
                            "Optimizer HF Trainer. Défaut intelligent : 'adamw_bnb_8bit' en "
                            "full-ft (économise ~18 Go sur les 3B params), 'adamw_torch' (fp32, "
                            "non quantisé) en LoRA — les params entraînables sont alors minuscules "
                            "donc l'optimizer fp32 tient sans souci. Override possible."
                        ))
    parser.add_argument("--max-depth", type=int, default=0,
                        help=(
                            "Keep only training items with depth <= max-depth. "
                            "0 = all items. Requires data/train/textcraft_train_with_depth.json "
                            "(generate with src/utils/label_depths.py)."
                        ))
    parser.add_argument("--depth-exact", type=str, default="",
                        help=(
                            "Garde uniquement les items dont depth ∈ liste fournie. "
                            "Ex: '1' (depth 1 seul) ou '3,4' (depth 3 et 4). "
                            "Exclusif avec --max-depth. Utilisé par le curriculum par stage."
                        ))
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to model dir (default: models/Qwen2.5-3B-Instruct).")
    parser.add_argument("--warm-start-dir", type=str, default="",
                        help="(full-FT) Injecte les poids (model.safetensors) de ce dossier dans la "
                             "POLITIQUE après la création du trainer, et charge son optimizer.pt "
                             "(s'il existe) au début du train. L'ancre KL (modèle de référence) et "
                             "le tokenizer restent ceux de --model-path : à la différence de "
                             "--model-path <best>, la KL continue de mesurer la distance à la BASE.")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt. Pass '' for models without system role (e.g. Gemma-3).")
    parser.add_argument("--run-name", type=str, default="trl_grpo_textcraft_smoke")
    parser.add_argument(
        "--max-rounds-schedule",
        type=str,
        default="",
        help=(
            "ScalingInter curriculum piloté par STEP, format '<rounds>:<step>', "
            "e.g. '5:0,10:13,15:26,20:38' = max_rounds=5 dès step 0, =10 dès step 13, "
            "=15 dès step 26, =20 dès step 38. Empty = cap fixe MAX_SIM_ROUNDS."
        ),
    )
    parser.add_argument(
        "--max-rounds-schedule-epochs",
        type=str,
        default="",
        help=(
            "ScalingInter curriculum piloté par EPOCH (et non step), format "
            "'<rounds>:<epoch>', e.g. '6:0,11:2,16:4,21:6,25:8' = 6 tours dès "
            "l'epoch 0, 11 dès l'epoch 2, ... Exclusif avec --max-rounds-schedule."
        ),
    )
    parser.add_argument(
        "--max-completion-schedule-epochs",
        type=str,
        default="",
        help=(
            "Curriculum du budget de sortie PAR TOUR (exp35), piloté par epoch, "
            "format '<max_tokens>:<epoch>', e.g. '256:0,512:15,1024:35'. Ne pilote "
            "que max_tokens de la génération vLLM (rollout) ; --max-completion-length "
            "reste la valeur passée à TRL (la fixer au MAX du schedule). Backend "
            "vLLM in-process uniquement (le repli HF ignore le schedule)."
        ),
    )
    parser.add_argument(
        "--depth-schedule-epochs",
        type=str,
        default="",
        help=(
            "Curriculum DEPTH par paliers DANS un seul run (exp33), format "
            "'<depth>:<epoch>', ex. '1:0,2:6,3:20,4:45' = items depth<=1 dès l'epoch 0, "
            "<=2 dès l'epoch 6... Implémenté par échantillonnage pondéré (magellan.py "
            "DepthScheduleProvider) : le dataset reste entier, les items hors palier ont "
            "une probabilité nulle. Exclusif avec --max-depth/--depth-exact/--goal-sampler."
        ),
    )
    parser.add_argument(
        "--depth-schedule-auto",
        action="store_true",
        help=(
            "Curriculum DEPTH AUTO-DÉCLENCHÉ (exp33.1) : palier suivant si le reward "
            "train moyen sur la dernière epoch complète >= --depth-auto-threshold, "
            "sinon après --depth-auto-max-epochs au palier courant "
            "(magellan.py DepthAutoScheduleProvider). Même masque uniforme depth<=palier "
            "que --depth-schedule-epochs ; exclusif avec lui et avec --goal-sampler."
        ),
    )
    parser.add_argument("--depth-auto-threshold", type=float, default=0.8,
                        help="Seuil de reward train moyen (1 epoch) déclenchant le palier suivant.")
    parser.add_argument("--depth-auto-max-epochs", type=float, default=10.0,
                        help="Durée max d'un palier en epochs avant passage forcé.")
    parser.add_argument("--depth-auto-start-stage", type=int, default=1,
                        help="Palier de départ (reprise après purge : mettre le dernier "
                             "palier atteint avant l'interruption, lu dans les logs "
                             "'[depth-auto] >>> PALIER depth<=X').")
    parser.add_argument("--goal-sampler", type=str, default="uniform",
                        choices=["uniform", "online", "ek_depth", "magellan"],
                        help=(
                            "Autocurriculum (exp34) : échantillonnage des items train ∝ progrès "
                            "d'apprentissage (port MAGELLAN, src/train/magellan.py). 'uniform' = "
                            "GRPO standard ; 'online'/'ek_depth' = ALP par item/par depth sans "
                            "réseau ; 'magellan' = estimateur appris sur les embeddings du LLM "
                            "(tête SR + adapters LoRA séparés + compétence retardée par snapshots)."
                        ))
    parser.add_argument("--magellan-N", type=int, default=100,
                        help="Horizon du retard de compétence, en optimizer steps (leur N=100).")
    parser.add_argument("--magellan-epsilon-start", type=float, default=1.0)
    parser.add_argument("--magellan-epsilon-end", type=float, default=0.2)
    parser.add_argument("--magellan-epsilon-decay", type=float, default=320,
                        help="Décroissance exponentielle de l'ε-greedy, en steps (eux : 320).")
    parser.add_argument("--magellan-buffer-size", type=int, default=5000,
                        help="Buffer (but, succès) pour l'entraînement de la tête SR.")
    parser.add_argument("--magellan-batch-size", type=int, default=256,
                        help="Buts par sr_update (échantillon pondéré récence).")
    parser.add_argument("--magellan-recompute-freq", type=int, default=32,
                        help="Steps entre deux recomputes de l'ALP sur tous les buts.")
    parser.add_argument("--magellan-sr-lr", type=float, default=1e-4,
                        help="LR Adam de l'estimateur SR (leur rl_script_args.lr=1e-4 — passé "
                             "tel quel à sr_update, updater.py:93).")
    parser.add_argument("--magellan-sr-lora-r", type=int, default=16,
                        help="Rang des adapters SR séparés (leur lora_r=16).")
    parser.add_argument("--magellan-sr-lora-alpha", type=int, default=32)
    parser.add_argument("--lr-stage-every-epochs", type=float, default=0,
                        help="Paliers LR/beta : divise le LR (et beta, sauf --lr-stage-keep-beta) "
                             "par --lr-stage-factor toutes les N epochs. 0 = off. Automatise "
                             "l'échelle de LR de la lignée exp10 en un seul run (exp20).")
    parser.add_argument("--lr-stage-factor", type=float, default=3.0,
                        help="Facteur de division du LR (et beta) à chaque palier (défaut 3).")
    parser.add_argument("--lr-stage-keep-beta", action="store_true", default=False,
                        help="Avec --lr-stage-every-epochs : ne divise QUE le LR, beta reste "
                             "constant (ablation ; défaut : beta suit le LR).")
    parser.add_argument("--lr-stage-restore-best", action="store_true", default=False,
                        help="Avec --lr-stage-every-epochs (exp22.5, LoRA uniquement) : suit la "
                             "moyenne roulante du reward train, sauve le best du PALIER "
                             "(adapter + optimizer) dans saves/trl_grpo/<run>_stagebest et le "
                             "best GLOBAL dans <run>_besttrain ; à chaque frontière de palier, "
                             "RECHARGE le best du palier écoulé (poids + moments Adam) avant "
                             "d'appliquer le LR réduit.")
    parser.add_argument("--lr-adaptive", action="store_true", default=False,
                        help="Décroissance de LR ADAPTATIVE au reward train (exp22) : divise "
                             "LR et beta par --lr-stage-factor quand la moyenne roulante du "
                             "reward train (fenêtre --lr-adaptive-window-epochs) reste sous le "
                             "best pendant --lr-adaptive-patience demi-epochs consécutives. "
                             "Exclusif avec --lr-stage-every-epochs.")
    parser.add_argument("--lr-adaptive-window-epochs", type=float, default=1.0,
                        help="Fenêtre de la moyenne roulante du reward train, en epochs (défaut 1).")
    parser.add_argument("--lr-adaptive-check-every-epochs", type=float, default=0.5,
                        help="Période des checks, en epochs (défaut 0.5 = demi-epoch).")
    parser.add_argument("--lr-adaptive-patience", type=int, default=3,
                        help="Nb de checks consécutifs sous le best avant de couper le LR (défaut 3).")
    parser.add_argument("--lr-adaptive-eps", type=float, default=0.005,
                        help="Tolérance de bruit sous le best (échelle reward 0-1, défaut 0.005 "
                             "≈ std de la moyenne sur ~3000 trajectoires).")
    parser.add_argument("--lr-adaptive-min-lr", type=float, default=1e-9,
                        help="Plancher de LR : plus aucune coupe en dessous (défaut 1e-9).")
    parser.add_argument("--lr-adaptive-restore-best", action="store_true", default=False,
                        help="Avec --lr-adaptive (exp22.1, LoRA uniquement) : sauve l'adapter à "
                             "chaque nouveau best de moyenne roulante du reward TRAIN "
                             "(saves/trl_grpo/<run>_besttrain, home persistant, remplacé à chaque "
                             "nouveau best) et, à chaque coupe de LR, RECHARGE ces poids + remet "
                             "les moments Adam à zéro : on consolide le meilleur état connu au "
                             "lieu de figer la dérive post-pic (leçon exp22).")
    parser.add_argument("--lr-adaptive-min-stage-epochs", type=float, default=0.0,
                        help="Durée MINIMALE d'un palier de LR, en epochs (exp22.2 : 3). Tant "
                             "qu'elle n'est pas écoulée, une coupe déclenchée par la patience est "
                             "retenue (et exécutée au premier check suivant si le signal persiste). "
                             "0 = off (sémantique exp22.1).")
    parser.add_argument("--lr-adaptive-ref-median-k", type=int, default=0,
                        help="K>0 : la détection de stagnation compare la fenêtre courante à la "
                             "MÉDIANE des K derniers checks du palier (robuste au bruit) au lieu "
                             "du max historique (biaisé +1-2 sigma, leçon exp22.1 : coupes sur du "
                             "bruit à cadence minimale). Le max reste utilisé pour SAVE le best. "
                             "0 = max historique (sémantique exp22.1). exp22.2 : 5.")
    parser.add_argument("--lr-adaptive-save-optimizer", action="store_true", default=False,
                        help="Avec --lr-adaptive-restore-best : sauve AUSSI l'état de l'optimizer "
                             "(moments Adam fp32, ~2-4x la taille de l'adapter) avec le best, et "
                             "le restaure à la coupe au lieu de remettre les moments à zéro — "
                             "reprise cohérente poids+optimizer, et resume exact après coupure.")
    parser.add_argument("--lr-adaptive-stop-at-floor", action="store_true", default=False,
                        help="Arrête le run quand une coupe passerait sous --lr-adaptive-min-lr "
                             "(leçon exp22.1 : 18 epochs brûlées à LR ~0). À combiner avec un "
                             "plancher réaliste, ex. --lr-adaptive-min-lr 1e-7.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Éval Pass@1 sur le test set tous les N steps, logguée dans wandb "
                             "sous eval/pass_at_1 (0 = off ; nécessite --use-vllm-inprocess)")
    parser.add_argument("--eval-items", type=int, default=0,
                        help="Nb d'items du test set pour l'éval périodique (0 = tous)")
    parser.add_argument("--best-init-score", type=float, default=-1.0,
                        help="Seuil initial du save-best (Pass@1, 0..1). Mettre au score du "
                             "checkpoint de reprise (ex. 0.32) pour ne jamais sauver pire que lui. "
                             "Défaut -1 = le 1er eval sauve toujours.")
    parser.add_argument("--best-delete-before-save", action="store_true", default=False,
                        help="Supprime l'ancien best AVANT d'écrire le nouveau (pic disque = "
                             "1 modèle au lieu de 2). Requis pour le full-FT 7B (exp26) : le "
                             "home 35 Go ne peut pas héberger deux modèles de 15 Go pendant le "
                             "swap. Fenêtre de quelques minutes sans best complet, assumée.")
    parser.add_argument("--save-best-optimizer", action="store_true", default=False,
                        help="Sauve aussi l'état de l'optimiseur (optimizer.pt) avec le best test. "
                             "Full-FT 3B + adamw_bnb_8bit : ~6.2 Go en plus des 5.8 Go du modèle — "
                             "vérifier l'espace disque avant (politique 2026-07-29 : best + optimizer "
                             "sur le home). Permet une reprise avec moments Adam cohérents après purge /tmp.")
    parser.add_argument("--save-steps", type=int, default=0,
                        help="Période de sauvegarde des checkpoints (0 = défaut : 50 en full-ft, 5 sinon)")
    parser.add_argument("--save-total-limit", type=int, default=0,
                        help="Nb de checkpoints conservés (0 = défaut : 1 en full-ft, 3 sinon). "
                             "Augmenter pour garder l'historique, ex. un par point d'eval "
                             "(attention au disque : ~5.8 Go par checkpoint)")
    parser.add_argument("--save-best-only", action="store_true", default=False,
                        help="N'écrit JAMAIS le dernier checkpoint : ni les checkpoints HF "
                             "périodiques/de fin (save_strategy='no', donc pas d'optimizer.pt), "
                             "ni le modèle final dans out_dir. Seul le callback d'éval sauve "
                             "<run>_best, et uniquement quand le Pass@1 test s'améliore. "
                             "Évite de saturer le disque et de perdre le best (cf. exp10.3).")
    parser.add_argument("--use-vllm-inprocess", action="store_true", default=False,
                        help="Moteur vLLM colocate GÉRÉ PAR TRL (use_vllm=True, vllm_mode='colocate') : "
                             "génération rapide avec prefix caching, poids synchronisés EN MÉMOIRE "
                             "par TRL à chaque step (PEFT inclus). Requiert TRL>=1.9 / vLLM>=0.25 "
                             "(env v2 : setup/setup_agentgym_rl_v2.sh). Sans ce flag : génération HF.")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.17,
                        help="gpu_memory_utilization du moteur vLLM colocate (0.17 validé sur B200 "
                             "pour le 3B : laisse la marge aux saves de checkpoints)")
    parser.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa"],
                        default="flash_attention_2",
                        help="Implémentation d'attention du modèle TRL (défaut : flash_attention_2, "
                             "débloqué par la migration VM glibc 2.39 ; sdpa = ancien repli)")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help=(
            "Path to a TRL checkpoint dir (e.g. saves/.../checkpoint-25) "
            "to resume training state from. Empty = fresh run."
        ),
    )
    parser.add_argument("--fewshot", type=int, default=0,
                        help="Nb d'exemples résolus (recettes hors train/test, exp18/19) injectés "
                             "dans le prompt de CHAQUE rollout ET de l'éval périodique. 0 = zero-shot.")
    parser.add_argument("--fewshot-file", type=str,
                        default=str(REPO_ROOT / "data" / "eval" / "textcraft_fewshot_examples.json"))
    parser.add_argument("--fewshot-format", choices=["dialogue", "bloc"], default="dialogue",
                        help="dialogue = tours user/assistant (recommandé, cf. exp18)")
    parser.add_argument("--vllm-max-len", type=int, default=16384,
                        help="max_model_len du moteur vLLM in-process (monter à ~20480 avec "
                             "--fewshot >= 10 : les exemples ajoutent ~2-5k tokens au contexte)")
    parser.add_argument("--plan-mode", action="store_true", default=False,
                        help="Single-turn « pure reasoning » (exp21) : le modèle émet en UNE "
                             "complétion le raisonnement + la séquence d'actions complète, "
                             "rejouée telle quelle dans l'env (rollout_plan.py). Remplace la "
                             "boucle multi-tour, les prompts ET l'éval périodique. "
                             "Incompatible avec --max-rounds-schedule* et --fewshot-format bloc.")
    parser.add_argument("--output-root", type=str, default="",
                        help="Racine des checkpoints périodiques (défaut : saves/trl_grpo, disque home). "
                             "Ex. /tmp/trl_grpo_runs pour garder les checkpoints resumables sur l'overlay "
                             "sans saturer le home — le best adapter reste TOUJOURS sur le home "
                             "(saves/trl_grpo/<run>_best, via le callback d'éval).")
    return parser


def main(args: argparse.Namespace | None = None, rollout_func=None) -> None:
    """Assemble et lance le run GRPO.

    rollout_func : None = GRPO pur (rollout.grpo_rollout_func) ; train_grpo_snis.py
    passe ici sa rollout_func SNIS — c'est le SEUL point de variation entre les deux."""
    if args is None:
        args = build_parser().parse_args()
    if args.plan_mode:
        if rollout_func is not None:
            raise SystemExit("--plan-mode est incompatible avec une rollout_func externe (SNIS).")
        if args.max_rounds_schedule or args.max_rounds_schedule_epochs:
            raise SystemExit("--plan-mode est single-turn : pas de schedule de max_rounds.")
        from src.train.rollout_plan import plan_rollout_func
        rollout_func = plan_rollout_func
    if rollout_func is None:
        rollout_func = grpo_rollout_func

    depth_in: set[int] | None = None
    if args.depth_exact:
        if args.max_depth:
            raise SystemExit("--depth-exact et --max-depth sont exclusifs.")
        depth_in = {int(x) for x in args.depth_exact.split(",") if x.strip()}

    if args.max_rounds_schedule and args.max_rounds_schedule_epochs:
        raise SystemExit("--max-rounds-schedule et --max-rounds-schedule-epochs sont exclusifs.")
    if args.max_rounds_schedule:
        schedules.MAX_ROUNDS_SCHEDULE = schedules.parse_max_rounds_schedule(args.max_rounds_schedule)
        print(f"[scaling-inter] schedule (step-based) = {schedules.MAX_ROUNDS_SCHEDULE}", flush=True)
    if args.max_rounds_schedule_epochs:
        schedules.MAX_ROUNDS_SCHEDULE_EPOCHS = \
            schedules.parse_max_rounds_schedule_epochs(args.max_rounds_schedule_epochs)
        print(f"[scaling-inter] schedule (epoch-based) = {schedules.MAX_ROUNDS_SCHEDULE_EPOCHS}", flush=True)
    if args.max_completion_schedule_epochs:
        sched = schedules.parse_max_completion_schedule_epochs(args.max_completion_schedule_epochs)
        if max(tok for _, tok in sched) > args.max_completion_length:
            raise SystemExit(
                f"--max-completion-schedule-epochs dépasse --max-completion-length "
                f"({args.max_completion_length}) : fixer --max-completion-length au "
                f"MAX du schedule (budgets TRL/vLLM dimensionnés dessus).")
        if not args.use_vllm_inprocess:
            raise SystemExit("--max-completion-schedule-epochs requiert --use-vllm-inprocess "
                             "(le repli HF ignore le schedule).")
        schedules.MAX_COMPLETION_SCHEDULE_EPOCHS = sched
        print(f"[completion-schedule] schedule (epoch-based) = {sched}", flush=True)

    fewshot_block, fewshot_messages = (None, None)
    if args.fewshot > 0:
        if args.plan_mode:
            # Mêmes exemples résolus qu'exp18/19/20, reformatés single-turn
            # (tâche → Thought + séquence d'actions complète).
            from src.train.rollout_plan import load_plan_fewshot
            fewshot_messages = load_plan_fewshot(args.fewshot_file, args.fewshot)
            print(f"[fewshot] k={args.fewshot} exemples single-turn (plan) injectés dans "
                  f"chaque rollout + éval périodique", flush=True)
        else:
            from src.eval.textcraft_common import load_fewshot
            fewshot_block, fewshot_messages = load_fewshot(
                args.fewshot_file, args.fewshot, args.fewshot_format)
            print(f"[fewshot] k={args.fewshot} exemples ({args.fewshot_format}) injectés dans "
                  f"chaque rollout + éval périodique ({len(fewshot_messages or [])} tours)", flush=True)

    model_path = Path(args.model_path) if args.model_path else DEFAULT_MODEL_PATH
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    # Le moteur vLLM est désormais construit et synchronisé PAR TRL (use_vllm=True,
    # vllm_mode="colocate" dans GRPOConfig ci-dessous) — plus d'init manuelle ici.
    print(f"[train] Model: {model_path}"
          + (" — vLLM colocate géré par TRL" if args.use_vllm_inprocess else " — génération HF"),
          flush=True)

    check_server()
    prompts_per_step = args.gradient_accumulation_steps // args.num_generations
    if args.gradient_accumulation_steps % args.num_generations != 0:
        raise SystemExit(
            f"gradient_accumulation_steps ({args.gradient_accumulation_steps}) must be "
            f"divisible by num_generations ({args.num_generations})."
        )
    print(
        f"[train] batch GRPO: {args.gradient_accumulation_steps} traj/step, "
        f"{prompts_per_step} prompts/step, N={args.num_generations}",
        flush=True,
    )
    if args.steps_per_generation:
        if args.steps_per_generation % args.gradient_accumulation_steps != 0:
            raise SystemExit(
                f"--steps-per-generation ({args.steps_per_generation}) doit être un multiple de "
                f"--gradient-accumulation-steps ({args.gradient_accumulation_steps}).")
        if args.steps_per_generation % args.num_generations != 0:
            raise SystemExit(
                f"--steps-per-generation ({args.steps_per_generation}) doit être un multiple de "
                f"--num-generations ({args.num_generations}).")
        print(
            f"[train] vrai PPO : buffer de {args.steps_per_generation} traj "
            f"({args.steps_per_generation // args.num_generations} prompts), consommé en "
            f"{args.steps_per_generation // args.gradient_accumulation_steps} optimizer steps "
            f"clippés (eps 0.2) — old_logps figées au modèle pré-updates (mécanique verl).",
            flush=True,
        )
    if args.plan_mode:
        from src.train.rollout_plan import build_plan_prompt_rows
        if args.max_depth or depth_in:
            raise SystemExit("--plan-mode ne supporte pas (encore) le filtrage par depth.")
        rows = build_plan_prompt_rows(max_items=args.max_items,
                                      system_prompt=args.system_prompt,
                                      fewshot_messages=fewshot_messages)
    else:
        rows = build_prompt_rows(max_items=args.max_items, max_depth=args.max_depth,
                                 system_prompt=args.system_prompt, depth_in=depth_in,
                                 fewshot_block=fewshot_block, fewshot_messages=fewshot_messages)
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_root = Path(args.output_root) if args.output_root else REPO_ROOT / "saves" / "trl_grpo"
    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", "rl-gym-workout")
    # Le compte wandb v-lagresle n'a pas d'entity par défaut (CommError sinon) ;
    # l'entity utilisable est l'équipe v-lagresle-criteo (cf. wandb.Api().viewer.teams).
    os.environ.setdefault("WANDB_ENTITY", "v-lagresle-criteo")

    # Smoke test = run minuscule (≤10 steps ET pas de mode epochs) : on coupe wandb
    # et la sauvegarde des checkpoints. Un run en --num-epochs n'est JAMAIS un smoke.
    is_smoke = (args.num_epochs <= 0 and args.max_steps <= 10)

    # Optimizer : défaut intelligent selon full-ft vs LoRA (override possible via --optim).
    optim_choice = args.optim or ("adamw_bnb_8bit" if args.full_ft else "adamw_torch")
    print(f"[train] Optimizer: {optim_choice} ({'LoRA' if not args.full_ft else 'full-ft'})", flush=True)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        # Skip wandb for smoke tests to avoid cluttering the project.
        report_to=[] if is_smoke else ["wandb"],
        # bs=1 (micro-batch) keeps the forward pass at 1×seq×vocab to avoid OOM on B200.
        # 1 optimizer step = grad_accum micro-steps × 1 prompt × num_generations rollouts.
        # → trajectoires/step ≈ grad_accum ; prompts/step = grad_accum / num_generations.
        # Papier AgentGym-RL TextCraft: train_batch_size=32, n=8 → grad_accum=256, N=8.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Optimizer : voir optim_choice calculé plus haut.
        optim=optim_choice,
        learning_rate=args.learning_rate,
        # Explicit clip (default is also 1.0, but make intent clear after the
        # grad_norm=1765 spike at step 35 of v2 — see WORKLOG step50 anomaly).
        max_grad_norm=1.0,
        # KL regularization: paper uses kl_loss_coef=0.001 (low_var_kl type).
        # TRL beta is equivalent; TRL already uses the same k3 estimator as verl low_var_kl.
        # Configurable: en LoRA + LR eleve, beta=0.001 est trop faible -> runaway KL
        # (cf. effondrement exp10). Remonter beta resserre l'ancrage a la reference.
        beta=args.beta,
        # The paper's script sets ppo_inner_epochs=2 BUT the AgentGym-RL verl fork ignores
        # it: update_policy() does a single pass over the batch (ppo_epochs only appears in
        # the MFU metric, agent_fsdp_workers.py:413). The actual paper run is 1 epoch.
        # With num_iterations=1 and steps_per_generation == grad_accum (default), TRL skips
        # the old_logprobs forward and trains fully on-policy (ratio ≡ 1).
        num_iterations=1,
        # --steps-per-generation > grad_accum (exp23) = mini-batching PPO de verl : le
        # buffer de rollouts est consommé en plusieurs optimizer steps ; TRL détecte le
        # désalignement (grad_accum % (steps_per_generation × num_iterations) != 0) et
        # calcule les old_per_token_logps → ratio clippé réel dès la 2e update.
        steps_per_generation=(args.steps_per_generation or None),
        # Bonus d'entropie (verl entropy_coeff=0.001) : loss -= coef × entropie moyenne
        # par token actif (même signe/forme que dp_actor.py:253). 0.0 = terme absent.
        entropy_coef=args.entropy_coef,
        # verl (repro papier, cf. audit 11 juin 2026) utilise un schedule constant ;
        # le défaut HF Trainer ("linear") décroît vers 0 et divise le LR moyen par
        # ~2. Configurable via --lr-scheduler-type (défaut "constant" : aucun
        # changement de comportement si le flag n'est pas passé).
        lr_scheduler_type=args.lr_scheduler_type,
        # Explicit: "dapo" is the TRL 1.4 default. Its token-level aggregation
        # (sum / total_tokens) matches verl's masked_mean closer than loss_type="grpo"
        # (per-sequence mean), pinned here for reproducibility across TRL versions.
        loss_type="dapo",
        # Moteur vLLM colocate GÉRÉ PAR TRL (migration 2026-07-22, TRL>=1.9 requis) :
        # TRL construit le moteur dans le process et synchronise les poids EN MÉMOIRE
        # (merge->push->unmerge, PEFT inclus) avant chaque rollout_func — remplace
        # l'ancien vllm_engine maison (écriture 5.8 Go/step + recréation moteur).
        use_vllm=args.use_vllm_inprocess,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        vllm_max_model_length=args.vllm_max_len,
        # Parité avec la lignée exp10/exp19 (TRL 1.4 : use_vllm=False → pas de
        # correction IS, ratio ≡ 1 en on-policy). La correction IS de TRL 1.9
        # utiliserait nos logprobs de rollout (0.0 sur les tokens template/obs
        # masqués) comme logprobs d'échantillonnage → sémantique différente.
        # À réactiver un jour comme ablation contrôlée, pas par accident.
        vllm_importance_sampling_correction=False,
        # --num-epochs > 0 => piloter par epochs (max_steps=-1), sinon par max_steps.
        max_steps=(-1 if args.num_epochs > 0 else args.max_steps),
        num_train_epochs=(args.num_epochs if args.num_epochs > 0 else 3),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        logging_steps=1,
        # Disable saving for smoke tests to avoid wasting 6 GB per run.
        # For real runs: save every ~epoch, keep 1 checkpoint (peak disk = 12 GB during write).
        save_strategy="no" if (is_smoke or args.save_best_only) else "steps",
        save_steps=args.save_steps or (50 if args.full_ft else 5),
        save_total_limit=args.save_total_limit or (1 if args.full_ft else 3),
        save_only_model=args.full_ft,
        eval_strategy="no",
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True,
                           # flash_attention_2 débloqué par la migration VM glibc 2.39
                           # (2026-07-22) ; sdpa reste dispo via --attn-implementation.
                           "attn_implementation": args.attn_implementation},
    )

    lora_alpha = args.lora_alpha or 2 * args.lora_r
    peft_config = None if args.full_ft else LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.full_ft:
        print("[train] Full fine-tuning (no LoRA).", flush=True)
    else:
        print(f"[train] LoRA r={args.lora_r} alpha={lora_alpha}", flush=True)

    callbacks = [MemDiagCallback()]
    # (Plus de VllmSyncCallback : la sync par step est faite nativement par TRL,
    #  et TestEvalCallback force sa propre sync avant chaque éval.)

    warm_opt_cb = None
    if args.warm_start_dir:
        if not args.full_ft:
            raise SystemExit("--warm-start-dir ne supporte que --full-ft (en LoRA, utiliser la "
                             "continuation d'adapter — doc hebdo 21/07).")
        if args.resume_from_checkpoint:
            raise SystemExit("--warm-start-dir et --resume-from-checkpoint sont exclusifs "
                             "(deux mécanismes de reprise différents).")
        if not (Path(args.warm_start_dir) / "model.safetensors").exists():
            raise SystemExit(f"--warm-start-dir : model.safetensors introuvable dans "
                             f"{args.warm_start_dir}")
        warm_opt_path = Path(args.warm_start_dir) / "optimizer.pt"
        if warm_opt_path.exists():
            warm_opt_cb = WarmStartOptimizerCallback(str(warm_opt_path))
            callbacks.append(warm_opt_cb)
        else:
            print(f"[warm-start] pas d'optimizer.pt dans {args.warm_start_dir} — "
                  f"les moments Adam repartiront de zéro.", flush=True)

    lr_stage_cb = None
    if args.lr_stage_every_epochs > 0 and args.lr_adaptive:
        raise SystemExit("--lr-stage-every-epochs et --lr-adaptive sont exclusifs "
                         "(deux pilotes du même LR).")
    if args.lr_adaptive_restore_best and not args.lr_adaptive:
        raise SystemExit("--lr-adaptive-restore-best n'a de sens qu'avec --lr-adaptive.")
    if args.lr_stage_restore_best and args.lr_stage_every_epochs <= 0:
        raise SystemExit("--lr-stage-restore-best n'a de sens qu'avec --lr-stage-every-epochs.")
    if args.lr_stage_restore_best and args.full_ft:
        raise SystemExit("--lr-stage-restore-best nécessite LoRA (le save/restore porte "
                         "sur l'adapter seul, pas sur un modèle full-FT).")
    if args.lr_adaptive_save_optimizer and not args.lr_adaptive_restore_best:
        raise SystemExit("--lr-adaptive-save-optimizer n'a de sens qu'avec --lr-adaptive-restore-best.")
    if (args.lr_stage_every_epochs > 0 or args.lr_adaptive) and args.lr_scheduler_type != "constant":
        raise SystemExit(
            "Les paliers LR (calendaires ou adaptatifs) supposent lr_scheduler_type="
            "'constant' (modification de base_lrs avec multiplicateur de schedule ≡ 1) ; "
            f"--lr-scheduler-type={args.lr_scheduler_type!r} n'est pas supporté."
        )
    if args.lr_stage_every_epochs > 0 and args.lr_stage_restore_best:
        stagebest_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{args.run_name}_stagebest")
        globalbest_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{args.run_name}_besttrain")
        lr_stage_cb = schedules.StagedBestRestoreCallback(
            base_lr=args.learning_rate, base_beta=args.beta,
            every_epochs=args.lr_stage_every_epochs, factor=args.lr_stage_factor,
            window_epochs=args.lr_adaptive_window_epochs,
            check_every_epochs=args.lr_adaptive_check_every_epochs,
            scale_beta=not args.lr_stage_keep_beta,
            stage_dir=stagebest_dir, global_dir=globalbest_dir, save_optimizer=True)
        callbacks.append(lr_stage_cb)
        print(f"[lr-stage-restore] LR{' et beta' if not args.lr_stage_keep_beta else ''} "
              f"÷{args.lr_stage_factor:g} toutes les {args.lr_stage_every_epochs:g} epochs, "
              f"restore du best du palier (poids + optimizer) à chaque frontière "
              f"(départ lr={args.learning_rate:g}, beta={args.beta:g}) — "
              f"stage best: {stagebest_dir}, best global: {globalbest_dir}", flush=True)
    elif args.lr_stage_every_epochs > 0:
        lr_stage_cb = schedules.StagedLrBetaCallback(
            base_lr=args.learning_rate, base_beta=args.beta,
            every_epochs=args.lr_stage_every_epochs, factor=args.lr_stage_factor,
            scale_beta=not args.lr_stage_keep_beta)
        callbacks.append(lr_stage_cb)
        print(f"[lr-beta-stage] LR{' et beta' if not args.lr_stage_keep_beta else ''} "
              f"÷{args.lr_stage_factor:g} toutes les {args.lr_stage_every_epochs:g} epochs "
              f"(départ lr={args.learning_rate:g}, beta={args.beta:g})", flush=True)
    elif args.lr_adaptive:
        besttrain_dir = None
        if args.lr_adaptive_restore_best:
            if args.full_ft:
                raise SystemExit("--lr-adaptive-restore-best nécessite LoRA (le save/restore "
                                 "porte sur l'adapter seul, pas sur un modèle full-FT).")
            besttrain_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{args.run_name}_besttrain")
        lr_stage_cb = schedules.RewardAdaptiveLrCallback(
            base_lr=args.learning_rate, base_beta=args.beta,
            factor=args.lr_stage_factor,
            window_epochs=args.lr_adaptive_window_epochs,
            check_every_epochs=args.lr_adaptive_check_every_epochs,
            patience=args.lr_adaptive_patience, eps=args.lr_adaptive_eps,
            min_lr=args.lr_adaptive_min_lr,
            scale_beta=not args.lr_stage_keep_beta,
            best_dir=besttrain_dir,
            min_stage_epochs=args.lr_adaptive_min_stage_epochs,
            ref_median_k=args.lr_adaptive_ref_median_k,
            save_optimizer=args.lr_adaptive_save_optimizer,
            stop_at_floor=args.lr_adaptive_stop_at_floor)
        callbacks.append(lr_stage_cb)
        print(f"[reward-adaptive-lr] LR{' et beta' if not args.lr_stage_keep_beta else ''} "
              f"÷{args.lr_stage_factor:g} si la moyenne roulante du reward train "
              f"(fenêtre {args.lr_adaptive_window_epochs:g} ep) reste sous le best pendant "
              f"{args.lr_adaptive_patience} checks (1 check / {args.lr_adaptive_check_every_epochs:g} ep) "
              f"(départ lr={args.learning_rate:g}, beta={args.beta:g})"
              + (f" — save/restore du best train dans {besttrain_dir}" if besttrain_dir else ""),
              flush=True)

    anchor_cb = None
    if args.moving_anchor_every_epochs > 0:
        if args.full_ft:
            raise SystemExit("--moving-anchor-every-epochs nécessite LoRA : avec PEFT l'ancre "
                             "KL est « adapter désactivé », c'est elle qu'on déplace par "
                             "merge-and-restart (en full-FT, utiliser sync_ref_model de TRL).")
        if args.beta <= 0:
            raise SystemExit("--moving-anchor-every-epochs sans pénalité KL (--beta 0) n'a "
                             "pas d'objet.")
        if args.lr_adaptive or args.lr_stage_every_epochs > 0:
            raise SystemExit("--moving-anchor-every-epochs est incompatible avec les paliers "
                             "LR à restore : un restore d'adapter croiserait les ré-ancrages "
                             "(l'adapter restauré serait relatif à une ancienne base).")
        if args.moving_anchor_initial_cycle > 0 and not args.resume_from_checkpoint:
            raise SystemExit("--moving-anchor-initial-cycle > 0 n'a de sens qu'avec "
                             "--resume-from-checkpoint (reprise d'un run à ancre mobile).")
        anchors_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{args.run_name}_anchors")
        anchor_cb = schedules.MovingAnchorCallback(
            every_epochs=args.moving_anchor_every_epochs, anchors_dir=anchors_dir,
            initial_cycle=args.moving_anchor_initial_cycle)
        callbacks.append(anchor_cb)
        print(f"[moving-anchor] ancre KL mobile : merge-and-restart toutes les "
              f"{args.moving_anchor_every_epochs:g} epochs — snapshots de chaîne dans "
              f"{anchors_dir}"
              + (f" (reprise : {args.moving_anchor_initial_cycle} cycles déjà faits, "
                 f"prochain à l'epoch "
                 f"{(args.moving_anchor_initial_cycle + 1) * args.moving_anchor_every_epochs:g})"
                 if args.moving_anchor_initial_cycle else ""), flush=True)

    test_eval_cb = None
    if args.eval_every > 0:
        if not args.use_vllm_inprocess:
            raise SystemExit("--eval-every nécessite --use-vllm-inprocess (le moteur vLLM sert aussi à l'éval)")
        plan_eval_fn = None
        if args.plan_mode:
            from src.train.rollout_plan import run_plan_test_eval
            plan_eval_fn = run_plan_test_eval
        test_eval_cb = TestEvalCallback(eval_every=args.eval_every, eval_items=args.eval_items,
                                        best_init_score=args.best_init_score, run_name=args.run_name,
                                        fewshot_block=fewshot_block, fewshot_messages=fewshot_messages,
                                        eval_fn=plan_eval_fn, save_optimizer=args.save_best_optimizer,
                                        delete_before_save=args.best_delete_before_save)
        callbacks.append(test_eval_cb)
        print(f"[test_eval] Éval test set tous les {args.eval_every} steps "
              f"({args.eval_items or 'tous les'} items).", flush=True)

    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=textcraft_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=rollout_func,
        callbacks=callbacks,
    )
    if test_eval_cb is not None:
        test_eval_cb.trainer_ref = trainer
    if lr_stage_cb is not None:
        lr_stage_cb.trainer_ref = trainer
    if warm_opt_cb is not None:
        warm_opt_cb.trainer_ref = trainer
    if anchor_cb is not None:
        anchor_cb.trainer_ref = trainer

    # --- Curriculum par échantillonnage : depth par paliers (exp33) ou
    # --- autocurriculum ALP (exp34, port MAGELLAN — src/train/magellan.py) ---
    if args.depth_schedule_epochs or args.depth_schedule_auto or args.goal_sampler != "uniform":
        import json as _json
        from types import MethodType

        from src.train import magellan as mgl

        n_modes = sum([bool(args.depth_schedule_epochs), bool(args.depth_schedule_auto),
                       args.goal_sampler != "uniform"])
        if n_modes > 1:
            raise SystemExit("--depth-schedule-epochs, --depth-schedule-auto et "
                             "--goal-sampler sont mutuellement exclusifs.")
        if args.max_depth or args.depth_exact:
            raise SystemExit("Le curriculum par échantillonnage travaille sur le dataset "
                             "ENTIER : incompatible avec --max-depth/--depth-exact.")
        if getattr(cfg, "dataloader_num_workers", 0):
            raise SystemExit("Le sampler pondéré lit les poids en direct : "
                             "dataloader_num_workers doit rester 0.")

        depth_file = REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"
        with depth_file.open() as f:
            depth_map = _json.load(f)
        depths = [int(depth_map.get(r["item_id"], 99)) for r in rows]
        goals = list(range(len(rows)))          # un but = une ligne du dataset
        depth_of = dict(enumerate(depths))

        magellan_cb = None
        if args.depth_schedule_auto:
            # exp33.1 — palier déclenché au reward train (fenêtre = 1 epoch de steps).
            steps_per_epoch = max(1, len(rows) * args.num_generations
                                  // args.gradient_accumulation_steps)
            provider = mgl.DepthAutoScheduleProvider(
                depths, steps_per_epoch=steps_per_epoch,
                threshold=args.depth_auto_threshold,
                max_stage_epochs=args.depth_auto_max_epochs,
                start_depth=args.depth_auto_start_stage)
            trainer.add_callback(provider)   # reçoit reward/epoch via on_log
            prob_fn = provider.probabilities
            print(f"[depth-auto] curriculum depth déclenché au succès : seuil "
                  f"{args.depth_auto_threshold} sur {steps_per_epoch} steps (1 epoch), "
                  f"cap {args.depth_auto_max_epochs:g} epochs/palier"
                  + (f", REPRISE au palier {args.depth_auto_start_stage}"
                     if args.depth_auto_start_stage > 1 else ""), flush=True)
        elif args.depth_schedule_epochs:
            provider = mgl.DepthScheduleProvider(args.depth_schedule_epochs, depths)
            provider.trainer_ref = trainer
            prob_fn = provider.probabilities
            print(f"[depth-schedule] curriculum depth par paliers : "
                  f"{args.depth_schedule_epochs} (échantillonnage pondéré, dataset entier)",
                  flush=True)
        else:
            eps = dict(epsilon_start=args.magellan_epsilon_start,
                       epsilon_end=args.magellan_epsilon_end,
                       epsilon_decay=args.magellan_epsilon_decay)
            estimator = None
            if args.goal_sampler == "online":
                sampler = mgl.OnlineGoalSampler(goals, **eps,
                                                buffer_size=args.magellan_buffer_size)
            elif args.goal_sampler == "ek_depth":
                sampler = mgl.EKOnlineGoalSampler(goals, depth_of, **eps,
                                                  buffer_size=args.magellan_buffer_size)
            else:  # magellan
                if args.full_ft:
                    raise SystemExit("--goal-sampler magellan nécessite LoRA (adapters SR "
                                     "séparés sur le modèle PEFT de la politique).")
                estimator = mgl.MagellanEstimator(
                    sr_lr=args.magellan_sr_lr, batch_size=args.magellan_batch_size,
                    gradient_batch_size=mgl.MAGELLAN_DEFAULTS["gradient_batch_size"],
                    sr_lora_r=args.magellan_sr_lora_r,
                    sr_lora_alpha=args.magellan_sr_lora_alpha)
                texts = mgl.load_goal_texts(
                    rows, str(REPO_ROOT / "data" / "train" / "textcraft_goal_texts.json"))
                goal_ids = [tokenizer(t, truncation=True, max_length=2048)["input_ids"]
                            for t in texts]
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                estimator.attach(unwrapped, goal_ids,
                                 tokenizer.pad_token_id or tokenizer.eos_token_id)
                sampler = mgl.MAGELLANGoalSampler(
                    goals, estimator, N=args.magellan_N, **eps,
                    recompute_freq=args.magellan_recompute_freq)
            magellan_cb = mgl.MagellanCallback(
                sampler, estimator, buffer_size=args.magellan_buffer_size,
                batch_size=args.magellan_batch_size, depth_of=depth_of,
                log_path=str(REPO_ROOT / "logs" / f"magellan_{args.run_name}.jsonl"))
            magellan_cb.trainer_ref = trainer
            trainer.add_callback(magellan_cb)
            trainer._goal_recorder = magellan_cb.record
            trainer._item_idx_to_row = {r["item_idx"]: i for i, r in enumerate(rows)}
            prob_fn = sampler.sampling_probs
            print(f"[magellan] goal sampler '{args.goal_sampler}' actif : "
                  f"{len(goals)} buts, ε {args.magellan_epsilon_start}→"
                  f"{args.magellan_epsilon_end} (decay {args.magellan_epsilon_decay}), "
                  f"recompute /{args.magellan_recompute_freq} steps", flush=True)

        def _weighted_sampler(self, dataset=None):
            ds = dataset if dataset is not None else self.train_dataset
            return mgl.WeightedRepeatSampler(
                data_source=ds,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                prob_fn=prob_fn, seed=self.args.seed)

        trainer._get_train_sampler = MethodType(_weighted_sampler, trainer)

    if args.warm_start_dir:
        # APRÈS la création du trainer : le ref model (ancre KL) et le moteur vLLM ont
        # été construits depuis --model-path (la base). On n'écrase que la POLITIQUE.
        # TRL resynchronise vLLM avant la 1re génération (global_step != _last_loaded_step
        # à l'init) → les rollouts du step 0 utilisent bien les poids injectés.
        from safetensors.torch import load_file
        ws_path = Path(args.warm_start_dir) / "model.safetensors"
        ws_state = load_file(str(ws_path))
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        missing, unexpected = unwrapped.load_state_dict(ws_state, strict=False)
        if unexpected:
            raise SystemExit(f"[warm-start] clés inattendues dans {ws_path} : {unexpected[:5]}...")
        # lm_head.weight absent = normal (tied embeddings Qwen : suit embed_tokens).
        bad_missing = [k for k in missing if k != "lm_head.weight"]
        if bad_missing:
            raise SystemExit(f"[warm-start] poids manquants dans {ws_path} : {bad_missing[:5]}...")
        print(f"[warm-start] politique initialisée depuis {ws_path} "
              f"(missing={missing or 'aucun'}) — ancre KL et tokenizer restent {model_path}",
              flush=True)

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)
    print("[train] TRL+GRPO training run finished", flush=True)

    # Sauvegarde finale garantie (sauf smoke). On matérialise les poids finaux à la racine
    # de out_dir comme un modèle HF COMPLET (pas juste l'adapter en LoRA) pour que l'éval
    # (start_vllm_server.sh + eval) et le chaînage --model-path fonctionnent direct.
    if not is_smoke and not args.save_best_only:
        final_model = trainer.accelerator.unwrap_model(trainer.model)
        vllm_engine.save_model_for_vllm(final_model, str(out_dir))  # full-ft: direct ; LoRA: merged
        tokenizer.save_pretrained(str(out_dir))
        kind = "merged LoRA" if hasattr(final_model, "merge_adapter") else "full-ft"
        print(f"[train] Final model ({kind}) saved to {out_dir}", flush=True)
    elif args.save_best_only:
        print("[train] --save-best-only : pas de save du dernier modèle ; "
              "seul <run>_best (meilleur Pass@1) est conservé.", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
