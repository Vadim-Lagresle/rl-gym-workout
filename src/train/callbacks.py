"""Assembles the training callbacks from the command-line arguments.

In plain words: a callback is a piece of code that TRL calls at fixed moments of
training (every step, every log, at the start). This file decides which ones a run
gets: GPU-memory telemetry, the periodic test evaluation that also saves the best
model, the moving KL anchor, the (legacy) learning-rate schedules, and the
warm start of a full fine-tuning run from a previous model and optimizer. It also
checks that incompatible options are not combined, before any GPU work starts.

Used by train_grpo.py: build_callbacks() before the trainer exists,
attach_callbacks() right after, apply_warm_start() just before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from src.train import kl_anchor, lr_adaptive, lr_schedules
from src.train.data import REPO_ROOT
from src.train.diagnostics import MemDiagCallback
from src.train.periodic_eval import TestEvalCallback


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


def build_callbacks(args: argparse.Namespace, fewshot_block: str | None,
                    fewshot_messages: list[dict] | None) -> tuple[list[Any], list[Any]]:
    """Returns (callbacks for the trainer, callbacks that need a trainer reference)."""
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
        lr_stage_cb = lr_schedules.StagedBestRestoreCallback(
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
        lr_stage_cb = lr_schedules.StagedLrBetaCallback(
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
        lr_stage_cb = lr_adaptive.RewardAdaptiveLrCallback(
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
        anchor_cls = (kl_anchor.MovingRefAdapterCallback if args.moving_anchor_mode == "ref"
                      else kl_anchor.MovingAnchorCallback)
        anchor_cb = anchor_cls(
            every_epochs=args.moving_anchor_every_epochs, anchors_dir=anchors_dir,
            initial_cycle=args.moving_anchor_initial_cycle)
        callbacks.append(anchor_cb)
        print(f"[moving-anchor] ancre KL mobile : "
              f"{'recopie default→ref (adaptateur figé)' if args.moving_anchor_mode == 'ref' else 'merge-and-restart'}"
              f" toutes les "
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
        test_eval_cb = TestEvalCallback(eval_every=args.eval_every, eval_items=args.eval_items,
                                        best_init_score=args.best_init_score, run_name=args.run_name,
                                        fewshot_block=fewshot_block, fewshot_messages=fewshot_messages,
                                        save_optimizer=args.save_best_optimizer,
                                        delete_before_save=args.best_delete_before_save)
        callbacks.append(test_eval_cb)
        print(f"[test_eval] Éval test set tous les {args.eval_every} steps "
              f"({args.eval_items or 'tous les'} items).", flush=True)
    needs_trainer = [cb for cb in (test_eval_cb, lr_stage_cb, warm_opt_cb, anchor_cb) if cb is not None]
    return callbacks, needs_trainer


def attach_callbacks(trainer: Any, needs_trainer: list[Any]) -> None:
    """Gives each callback its trainer; the 'ref' anchor adds its frozen adapter here."""
    for cb in needs_trainer:
        cb.trainer_ref = trainer
        if hasattr(cb, "attach"):   # mode 'ref' : ajoute l'adaptateur figé AVANT train()
            cb.attach(trainer)


def apply_warm_start(trainer: Any, args: argparse.Namespace, model_path: Path) -> None:
    """(full-FT) Loads the policy weights of --warm-start-dir; the KL reference stays the base."""
    if not args.warm_start_dir:
        return
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
