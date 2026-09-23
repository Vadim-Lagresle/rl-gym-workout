"""Builds the TRL GRPO configuration and the LoRA adapter configuration.

In plain words: this is where the training recipe becomes concrete settings for the
TRL library: one prompt per micro-batch, G trajectories per prompt, 64 trajectories
per update, strictly on-policy updates, the KL coefficient beta, the token-level
loss ("dapo"), the vLLM generation engine running inside the same process, and the
checkpoint policy. Each setting carries a comment saying why it has this value and,
where relevant, how it matches the reference paper's verl code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from peft import LoraConfig
from trl import GRPOConfig

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_grpo_config(args: argparse.Namespace, out_dir: Path, is_smoke: bool) -> GRPOConfig:
    """GRPOConfig for a run (see the comments for the choice of each value)."""
    # Optimizer : défaut intelligent selon full-ft vs LoRA (override possible via --optim).
    optim_choice = args.optim or ("adamw_bnb_8bit" if args.full_ft else "adamw_torch")
    print(f"[train] Optimizer: {optim_choice} ({'LoRA' if not args.full_ft else 'full-ft'})", flush=True)

    return GRPOConfig(
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
        # TRL beta is equivalent; TRL uses the same k3 estimator as verl low_var_kl MAIS sans
        # la borne [-10, 10] par token de verl (core_algos.py:381) — cf. --kl-clamp (15/09/2026).
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


def make_lora_config(args: argparse.Namespace) -> LoraConfig | None:
    """LoRA on the seven attention and MLP projections; None in full fine-tuning."""
    lora_alpha = args.lora_alpha or 2 * args.lora_r
    if args.full_ft:
        print("[train] Full fine-tuning (no LoRA).", flush=True)
        return None
    print(f"[train] LoRA r={args.lora_r} alpha={lora_alpha}", flush=True)
    return LoraConfig(r=args.lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
