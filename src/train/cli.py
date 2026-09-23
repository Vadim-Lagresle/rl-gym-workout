"""Command-line arguments of the GRPO training entry point (train_grpo.py).

In plain words: every knob of a training run is declared here, with its default and
a short explanation: batch shape (group size G, trajectories per update), LoRA or
full fine-tuning, learning rate and KL coefficient, the moving KL anchor, the
curricula (horizon, token budget, depth), evaluation and checkpointing. Read this
file to know what a run can do; read train_grpo.py to see how the pieces are
assembled. The reference commands are in the README.

Notes (FR) : les textes d'aide gardent l'historique des décisions (numéros de runs
exp20, exp33…) pour que chaque option puisse être reliée au registre runs/INDEX.md.
"""

from __future__ import annotations

import argparse

from src.train.data import DEFAULT_SYSTEM_PROMPT, REPO_ROOT


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
    parser.add_argument("--kl-clamp", type=float, default=0.0,
                        help="Borne haute de l'estimateur KL k3 PAR TOKEN dans la loss (0 = off, "
                             "défaut TRL, sémantique exp10-42). verl/papier : 10 (low_var_kl clampé "
                             "à [-10, 10], core_algos.py:381). Nécessite le patch "
                             "setup/patch_trl_kl_clamp.py (vérifié au démarrage). Décision 15/09.")
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
                             "0 = off (ancre fixe historique). Détail : kl_anchor.MovingAnchorCallback.")
    parser.add_argument("--moving-anchor-mode", choices=["merge", "ref"], default="merge",
                        help="Mécanisme de l'ancre mobile. 'merge' (défaut, exp25→43) : "
                             "merge-and-restart = ReLoRA (rang accumulé, B·A et Adam remis à zéro). "
                             "'ref' (exp46) : UN adaptateur vivant jamais fusionné ni réinitialisé, "
                             "et un adaptateur figé 'ref' recopié depuis le vivant toutes les N "
                             "epochs, lu par TRL comme référence KL. Sépare « référence mobile » "
                             "de « merge-and-restart ». Détail : kl_anchor.MovingRefAdapterCallback.")
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
            "<=2 dès l'epoch 6... Implémenté par échantillonnage pondéré (sampling.py "
            "DepthScheduleProvider) : le dataset reste entier, les items hors palier ont "
            "une probabilité nulle. Exclusif avec --max-depth/--depth-exact/--depth-balance."
        ),
    )
    parser.add_argument(
        "--depth-schedule-auto",
        action="store_true",
        help=(
            "Curriculum DEPTH AUTO-DÉCLENCHÉ (exp33.1) : palier suivant si le reward "
            "train moyen sur la dernière epoch complète >= --depth-auto-threshold, "
            "sinon après --depth-auto-max-epochs au palier courant "
            "(sampling.py DepthAutoScheduleProvider). Même masque uniforme depth<=palier "
            "que --depth-schedule-epochs ; exclusif avec lui et avec --depth-balance."
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
    parser.add_argument("--train-file", type=str, default="",
                        help="Fichier de train alternatif (liste d'item_id), relatif au repo ou absolu. "
                             "Son fichier de depths est <stem>_with_depth.json à côté. Défaut : "
                             "data/train/textcraft_train.json. exp48 : "
                             "data/train/textcraft_train_plus_reservoir.json (train + réservoir few-shot).")
    parser.add_argument("--depth-balance", choices=["none", "uniform", "sqrt"], default="none",
                        help="Échantillonnage pondéré FIXE par profondeur (exp48, F). 'uniform' : chaque "
                             "profondeur pèse 1/4 des tirages ; 'sqrt' : poids d'une profondeur ∝ √n_d "
                             "(n_d = nb d'items de cette profondeur). Exclusif avec les curriculums "
                             "par échantillonnage.")
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
    parser.add_argument("--output-root", type=str, default="",
                        help="Racine des checkpoints périodiques (défaut : saves/trl_grpo, disque home). "
                             "Ex. /tmp/trl_grpo_runs pour garder les checkpoints resumables sur l'overlay "
                             "sans saturer le home — le best adapter reste TOUJOURS sur le home "
                             "(saves/trl_grpo/<run>_best, via le callback d'éval).")
    return parser
