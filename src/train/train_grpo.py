"""Entraînement GRPO multi-tour sur TextCraft avec TRL.

Implémente une boucle RL interactive : à chaque step GRPO, le modèle joue N
épisodes complets (generate → env.step → observe → repeat) via textcraft_rollout_func,
reçoit une récompense sparse 0/1 de l'environnement, et met à jour ses poids par GRPO.

Pré-requis :
  - Serveur TextCraft lancé : conda activate agentenv-textcraft &&
    cd external/AgentGym/agentenv-textcraft && textcraft --host 127.0.0.1 --port 36005
  - Env conda trl-b200 (B200) ou agentgym-rl (A100 sans vLLM)

Usage :
    python src/train/train_grpo.py --use-vllm --num-generations 8 --max-steps 200 --run-name exp7_b200
"""

#Dependecies :
from __future__ import annotations # for Python 3.10+ type hinting, ie more flexible 

import argparse 
import json
import os
import re
from pathlib import Path
from typing import Any

import requests # HTTP client for env interaction in rollout_func
from datasets import Dataset # transforms jsons into datasets for trl
from transformers import AutoTokenizer # tokenizer of Qwen 2.5 3B
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from agentenv.envs import TextCraftEnvClient



# Paths
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"


# Constants and utils for the interactive rollout and reward shaping.
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_SIM_ROUNDS = 30  # aligned with paper (AgentGym-RL textcraft_train.sh rounds=30)
ITEM_TAG_RE = re.compile(r"^<ITEM_IDX:(\d+)>$")

# ScalingInter curriculum: list of (step_threshold, max_rounds) sorted by step.
# Populated from --max-rounds-schedule (e.g. "5:0,10:13,15:26,20:38" for the
# 4-step curriculum from AgentGym-RL). None means "use MAX_SIM_ROUNDS constant".
MAX_ROUNDS_SCHEDULE: list[tuple[int, int]] | None = None


def parse_max_rounds_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse e.g. '5:0,10:13,15:26,20:38' into [(0,5),(13,10),(26,15),(38,20)]."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[tuple[int, int]] = []
    for p in parts:
        rounds_str, step_str = p.split(":")
        out.append((int(step_str), int(rounds_str)))
    out.sort(key=lambda x: x[0])
    return out


def current_max_rounds(trainer: Any) -> int:
    """Resolve the active max_rounds cap given the trainer's global step."""
    if MAX_ROUNDS_SCHEDULE is None:
        return MAX_SIM_ROUNDS
    step = 0
    state = getattr(trainer, "state", None)
    if state is not None:
        step = int(getattr(state, "global_step", 0) or 0)
    cap = MAX_SIM_ROUNDS
    for thr_step, thr_rounds in MAX_ROUNDS_SCHEDULE:
        if step >= thr_step:
            cap = thr_rounds
    return cap


def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int, max_depth: int = 0) -> list[dict[str, Any]]:
    with TRAIN_PATH.open() as f:
        rows = json.load(f)
    if max_depth > 0:
        depth_file = REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"
        if not depth_file.exists():
            raise FileNotFoundError(
                f"--max-depth requires {depth_file}. "
                "Generate it with: conda run -n agentenv-textcraft python src/utils/label_depths.py"
            )
        with depth_file.open() as f:
            depth_map: dict[str, int] = json.load(f)
        rows = [r for r in rows if depth_map.get(r["item_id"], 99) <= max_depth]
        print(f"[curriculum] max_depth={max_depth} → {len(rows)} items retained.", flush=True)
    if max_items > 0:
        rows = rows[:max_items]

    probe = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(rows), timeout=60)
    manual_human = probe.conversation_start[0]["value"]
    manual_ack = probe.conversation_start[1]["value"]

    out: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        idx = item_id_to_idx(item_id)
        # Hidden marker only for rollout_func bookkeeping (removed before generation).
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": manual_human},
            {"role": "assistant", "content": manual_ack},
            {"role": "user", "content": f"<ITEM_IDX:{idx}>"},
        ]
        out.append({"prompt": prompt, "item_id": item_id, "item_idx": idx})

    try:
        probe.close()
    except Exception:
        pass
    return out


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(completion)


def count_actions(text: str) -> int:
    return len(re.findall(r"Action:\s*(.*?)(?=\n|$)", text, flags=re.DOTALL))


def extract_actions(text: str) -> list[str]:
    actions = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, flags=re.DOTALL)
    out: list[str] = []
    for a in actions:
        # Keep only first line of each extracted action and normalize spaces.
        a = " ".join(a.strip().split())
        if a:
            out.append(a)
    return out


def first_action_or_empty(text: str) -> str:
    acts = extract_actions(text)
    return acts[0] if acts else ""


def textcraft_rollout_func(prompts: list[list[dict[str, str]]], trainer: GRPOTrainer) -> dict[str, Any]:
    """True interactive rollout: generate -> env.step -> observation -> repeat."""
    tokenizer = trainer.processing_class

    n = len(prompts)
    states = [list(p) for p in prompts]
    done = [False] * n
    final_rewards = [0.0] * n
    invalid_counts = [0] * n
    # Per-episode counters for syntax shaping at the message level (vs full
    # concatenated completion). See WORKLOG §1 "Notes méthodologiques (2026-05-11)".
    actions_per_turn_sum = [0] * n
    n_active_turns = [0] * n
    env_clients: list[TextCraftEnvClient] = []

    # Bootstrap one env per sample with the item idx encoded in the last user turn.
    for i in range(n):
        marker = str(states[i][-1]["content"]).strip()
        m = ITEM_TAG_RE.match(marker)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}] last user message: {marker!r}")
        idx = int(m.group(1))

        env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(idx)
        obs = env.observe()
        states[i][-1]["content"] = obs  # replace marker by initial observation
        env_clients.append(env)

    # Outputs expected by TRL rollout_func contract.
    prompt_ids_out: list[list[int]] = [[] for _ in range(n)]
    completion_ids_out: list[list[int]] = [[] for _ in range(n)]
    logprobs_out: list[list[float]] = [[] for _ in range(n)]

    cap = current_max_rounds(trainer)
    if MAX_ROUNDS_SCHEDULE is not None:
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        print(f"[scaling-inter] step={step} max_rounds={cap}", flush=True)
    for round_idx in range(cap):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break

        round_prompt_ids: list[list[int]] = []
        for i in active:
            rendered = tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            round_prompt_ids.append(prompt_ids)
            if round_idx == 0:
                prompt_ids_out[i] = prompt_ids

        # Use trainer internal generator to get completion token IDs + per-token logprobs.
        round_completion_ids, round_logprobs = trainer._generate_single_turn(
            round_prompt_ids, images=None, multimodal_fields={}
        )

        # Update env state one sample at a time.
        for j, i in enumerate(active):
            ids = round_completion_ids[j]
            lps = round_logprobs[j] if round_logprobs is not None else [0.0] * len(ids)

            text = tokenizer.decode(ids, skip_special_tokens=True)
            states[i].append({"role": "assistant", "content": text})

            # Count actions inside *this* assistant message (per-turn signal).
            # Same parser as `count_actions` so it is consistent with action extraction.
            n_actions_this_turn = count_actions(text)
            actions_per_turn_sum[i] += n_actions_this_turn
            n_active_turns[i] += 1

            action = first_action_or_empty(text)
            step = env_clients[i].step(f"Action: {action}")
            obs = step.state
            rew = float(step.reward)
            is_done = bool(step.done)
            states[i].append({"role": "user", "content": obs})

            completion_ids_out[i].extend(ids)
            logprobs_out[i].extend(lps[: len(ids)])
            final_rewards[i] = max(final_rewards[i], rew)
            low = obs.lower()
            if "could not" in low or "error:" in low or "wrong item format" in low:
                invalid_counts[i] += 1
            done[i] = is_done

    for env in env_clients:
        try:
            env.close()
        except Exception:
            pass

    # Guarantee non-empty completion/logprobs arrays.
    for i in range(n):
        if not completion_ids_out[i]:
            completion_ids_out[i] = [tokenizer.eos_token_id]
            logprobs_out[i] = [0.0]
        if not prompt_ids_out[i]:
            prompt_ids_out[i] = [tokenizer.eos_token_id]

    # Aggregate per-message action counts to a per-episode mean (1.0 = ideal).
    # Default to 1.0 when an episode produced 0 active turns (extremely rare; keeps shaping neutral).
    mean_actions_per_turn = [
        (actions_per_turn_sum[i] / n_active_turns[i]) if n_active_turns[i] > 0 else 1.0
        for i in range(n)
    ]

    print(
        f"[rollout] n={n} mean_n_actions_per_turn="
        f"{[round(x, 2) for x in mean_actions_per_turn]} "
        f"n_active_turns={n_active_turns} "
        f"episode_reward={final_rewards} invalid_steps={invalid_counts}",
        flush=True,
    )

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        # Extra fields forwarded to reward function through reward_kwargs
        "episode_reward": final_rewards,
        "invalid_steps": invalid_counts,
        "mean_n_actions_per_turn": mean_actions_per_turn,
    }


def textcraft_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    mean_n_actions_per_turn: list[float],
    **kwargs: Any,
) -> list[float]:
    """Sparse outcome reward only (matches AgentGym-RL TextCraft training recipe).

    Removes the v2/v3 shaping that empirically degraded Pass@1 (18 -> 14 -> 8 over
    baseline -> v2 -> v3). AgentGym-RL uses only the env 0/1 scalar at episode end,
    KL handled in the loss via use_kl_loss. See worker investigation 2026-05-12.

    invalid_steps and mean_n_actions_per_turn are kept in the signature only so
    rollout_func's return dict matches; they are NOT used to reshape the reward.
    """
    rewards = [float(r) for r in episode_reward]
    debug = " | ".join(
        f"r={r:+.3f} mean_n={mn:.2f} bad={bs}"
        for r, mn, bs in zip(rewards, mean_n_actions_per_turn, invalid_steps)
    )
    print(f"[reward] sparse outcome | {debug}", flush=True)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Max tokens per assistant turn (128 = smoke test, 512 = proper run).")
    parser.add_argument("--full-ft", action="store_true", default=False,
                        help="Full fine-tuning (no LoRA). Requires more VRAM — use on B200.")
    parser.add_argument("--max-depth", type=int, default=0,
                        help=(
                            "Keep only training items with depth <= max-depth. "
                            "0 = all items. Requires data/train/textcraft_train_with_depth.json "
                            "(generate with src/utils/label_depths.py)."
                        ))
    parser.add_argument("--run-name", type=str, default="trl_grpo_textcraft_smoke")
    parser.add_argument(
        "--max-rounds-schedule",
        type=str,
        default="",
        help=(
            "ScalingInter curriculum, e.g. '5:0,10:13,15:26,20:38' means: "
            "max_rounds=5 from step 0, =10 from step 13, =15 from step 26, "
            "=20 from step 38. Empty string keeps the fixed MAX_SIM_ROUNDS cap."
        ),
    )
    parser.add_argument("--use-vllm", action="store_true", default=False)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help=(
            "Path to a TRL checkpoint dir (e.g. saves/.../checkpoint-25) "
            "to resume training state from. Empty = fresh run."
        ),
    )
    args = parser.parse_args()

    if args.max_rounds_schedule:
        global MAX_ROUNDS_SCHEDULE
        MAX_ROUNDS_SCHEDULE = parse_max_rounds_schedule(args.max_rounds_schedule)
        print(f"[scaling-inter] schedule = {MAX_ROUNDS_SCHEDULE}", flush=True)

    check_server()
    rows = build_prompt_rows(max_items=args.max_items, max_depth=args.max_depth)
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = REPO_ROOT / "saves" / "trl_grpo" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        report_to=[],
        # bs=1 + grad_acc=8 avoids OOM on B200: forward pass logits = 8×seq×vocab
        # instead of 64×seq×vocab (bs=8 would OOM at 148 GB for 7664-token seqs).
        # Mathematically equivalent to bs=8 for GRPO (per-prompt advantage norm is unchanged).
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        # Explicit clip (default is also 1.0, but make intent clear after the
        # grad_norm=1765 spike at step 35 of v2 — see WORKLOG step50 anomaly).
        max_grad_norm=1.0,
        # KL regularization: paper uses kl_loss_coef=0.001 (low_var_kl type).
        # TRL beta is equivalent; default 0.0 means no KL penalty at all.
        beta=0.001,
        use_vllm=args.use_vllm,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        logging_steps=1,
        # Disable saving for smoke tests (max_steps <= 10) to avoid wasting 6 GB per run.
        # For real runs: save every epoch, keep 1 checkpoint (peak disk = 12 GB during write).
        save_strategy="no" if args.max_steps <= 10 else "steps",
        save_steps=max(1, args.max_steps // 12) if args.full_ft else 5,
        save_total_limit=1 if args.full_ft else 3,
        save_only_model=args.full_ft,
        eval_strategy="no",
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True},
    )

    peft_config = None if args.full_ft else LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.full_ft:
        print("[train] Full fine-tuning (no LoRA).", flush=True)

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=textcraft_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=textcraft_rollout_func,
    )

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)
    print("[smoke] TRL+GRPO training run finished")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
