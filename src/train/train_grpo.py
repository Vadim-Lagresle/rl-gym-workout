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
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.train import schedules, vllm_engine
from src.train.data import DEFAULT_MODEL_PATH, DEFAULT_SYSTEM_PROMPT, REPO_ROOT, \
    build_prompt_rows, check_server
from src.train.diagnostics import MemDiagCallback
from src.train.periodic_eval import TestEvalCallback
from src.train.rollout import grpo_rollout_func, textcraft_reward


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
                                        eval_fn=plan_eval_fn)
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
