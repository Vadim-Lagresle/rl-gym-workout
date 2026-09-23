"""GRPO training of a TextCraft agent with TRL: the entry point.

In plain words: this script trains a language model to solve TextCraft crafting
tasks by trial and error. At each update the model plays G complete episodes per
task (generate an action, the environment answers, repeat), each episode gets a
reward of 1 if the item is crafted and 0 otherwise, and GRPO pushes the model
towards the episodes that did better than the others of their group. This file
only reads the arguments and assembles the pieces; each piece lives in its own
module of src/train/:

    cli.py               all command-line options, with their meaning
    data.py              the training tasks and their prompts
    rollout.py           the episode loop (model <-> environment) and the reward
    vllm_engine.py       access to the vLLM generation engine that TRL runs in-process
    grpo_config.py       the TRL configuration (batch, loss, KL, checkpoints)
    callbacks.py         which callbacks the run gets (eval, anchor, LR schedules)
    kl_anchor.py         the moving KL reference (merge-and-restart, or frozen copy)
    horizon_schedules.py the horizon and token-budget curricula
    sampling.py          the depth curriculum and depth rebalancing
    periodic_eval.py     test evaluation every N updates and saving of the best model
    lr_schedules.py, lr_adaptive.py   legacy learning-rate schedules (exp20-exp23)

Prerequisites: the TextCraft server on port 36005 and the v2 environment
(setup/setup_agentgym_rl_v2.sh). Reference commands: README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo → imports src.*

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOTrainer  # noqa: E402

from src.train import horizon_schedules, vllm_engine  # noqa: E402
from src.train.callbacks import apply_warm_start, attach_callbacks, build_callbacks  # noqa: E402
from src.train.cli import build_parser  # noqa: E402
from src.train.data import DEFAULT_MODEL_PATH, REPO_ROOT, build_prompt_rows, check_server  # noqa: E402
from src.train.grpo_config import make_grpo_config, make_lora_config  # noqa: E402
from src.train.rollout import grpo_rollout_func, textcraft_reward  # noqa: E402
from src.train.sampling import install_weighted_sampler  # noqa: E402


def main() -> None:
    """Parses the arguments, builds dataset, config, callbacks and trainer, then trains."""
    args = build_parser().parse_args()
    if args.kl_clamp > 0:
        # Borne k3 : la ligne vit dans TRL (patchée par setup/patch_trl_kl_clamp.py) et ne
        # s'active que via la variable d'environnement — on vérifie AVANT tout chargement GPU.
        import importlib.util
        spec = importlib.util.spec_from_file_location("patch_trl_kl_clamp", REPO_ROOT / "setup" / "patch_trl_kl_clamp.py")
        patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
        if patch.MARKER not in patch.trl_file().read_text():
            raise SystemExit(f"--kl-clamp {args.kl_clamp} demandé mais TRL n'est pas patché "
                             f"({patch.trl_file()}). Lancer : python setup/patch_trl_kl_clamp.py")
        os.environ["TRL_KL_CLAMP"] = str(args.kl_clamp)
        print(f"[kl-clamp] TRL_KL_CLAMP={args.kl_clamp} (borne k3 par token, comme verl)", flush=True)
    depth_in: set[int] | None = None
    if args.depth_exact:
        if args.max_depth:
            raise SystemExit("--depth-exact et --max-depth sont exclusifs.")
        depth_in = {int(x) for x in args.depth_exact.split(",") if x.strip()}

    if args.max_rounds_schedule and args.max_rounds_schedule_epochs:
        raise SystemExit("--max-rounds-schedule et --max-rounds-schedule-epochs sont exclusifs.")
    if args.max_rounds_schedule:
        horizon_schedules.MAX_ROUNDS_SCHEDULE = horizon_schedules.parse_max_rounds_schedule(args.max_rounds_schedule)
        print(f"[scaling-inter] schedule (step-based) = {horizon_schedules.MAX_ROUNDS_SCHEDULE}", flush=True)
    if args.max_rounds_schedule_epochs:
        horizon_schedules.MAX_ROUNDS_SCHEDULE_EPOCHS = \
            horizon_schedules.parse_max_rounds_schedule_epochs(args.max_rounds_schedule_epochs)
        print(f"[scaling-inter] schedule (epoch-based) = {horizon_schedules.MAX_ROUNDS_SCHEDULE_EPOCHS}", flush=True)
    if args.max_completion_schedule_epochs:
        sched = horizon_schedules.parse_max_completion_schedule_epochs(args.max_completion_schedule_epochs)
        if max(tok for _, tok in sched) > args.max_completion_length:
            raise SystemExit(
                f"--max-completion-schedule-epochs dépasse --max-completion-length "
                f"({args.max_completion_length}) : fixer --max-completion-length au "
                f"MAX du schedule (budgets TRL/vLLM dimensionnés dessus).")
        if not args.use_vllm_inprocess:
            raise SystemExit("--max-completion-schedule-epochs requiert --use-vllm-inprocess "
                             "(le repli HF ignore le schedule).")
        horizon_schedules.MAX_COMPLETION_SCHEDULE_EPOCHS = sched
        print(f"[completion-schedule] schedule (epoch-based) = {sched}", flush=True)

    fewshot_block, fewshot_messages = (None, None)
    if args.fewshot > 0:
        from src.eval.textcraft_common import load_fewshot
        fewshot_block, fewshot_messages = load_fewshot(args.fewshot_file, args.fewshot, args.fewshot_format)
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
    train_path = None
    if args.train_file:
        train_path = Path(args.train_file)
        if not train_path.is_absolute():
            train_path = REPO_ROOT / train_path
    rows = build_prompt_rows(max_items=args.max_items, max_depth=args.max_depth,
                             system_prompt=args.system_prompt, depth_in=depth_in,
                             fewshot_block=fewshot_block, fewshot_messages=fewshot_messages,
                             train_path=train_path)
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

    cfg = make_grpo_config(args, out_dir, is_smoke)
    peft_config = make_lora_config(args)
    callbacks, needs_trainer = build_callbacks(args, fewshot_block, fewshot_messages)

    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=textcraft_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=grpo_rollout_func,
        callbacks=callbacks,
    )
    attach_callbacks(trainer, needs_trainer)
    install_weighted_sampler(trainer, args, rows)
    apply_warm_start(trainer, args, model_path)

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
