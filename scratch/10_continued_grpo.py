"""Continued GRPO training from a LoRA checkpoint, with selectable reward.

Two T3 branches are launched with this same script:

  T3-A (verifiable env reward, baseline of T3):
    python -u scratch/10_continued_grpo.py \
      --init-from-lora saves/trl_grpo/trl_grpo_v4_scalinginter_step50/checkpoint-50 \
      --items-offset 256 --max-items 256 --max-steps 50 \
      --num-generations 4 --reward-mode env \
      --run-name trl_grpo_v5_cont_envreward_step50

  T3-B (SCPO self-consistency reward, no env signal in the reward — env is
        still rolled to drive the same multi-turn interaction loop, but its
        success/failure is ignored when shaping rewards):
    python -u scratch/10_continued_grpo.py \
      --init-from-lora saves/trl_grpo/trl_grpo_v4_scalinginter_step50/checkpoint-50 \
      --items-offset 256 --max-items 256 --max-steps 50 \
      --num-generations 4 --reward-mode scpo \
      --run-name trl_grpo_v5_cont_scporeward_step50

Why this script is a near-copy of scratch/07_trl_grpo_textcraft_smoke.py:
T2 may still be running scratch/07_* in a sibling process; importing from
that module while it is being edited or used elsewhere is fragile, so we
duplicate the small set of helpers we need. The rollout function is bitwise
identical; only main() and the SCPO reward differ.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from agentenv.envs import TextCraftEnvClient

REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_PATH = REPO_ROOT / "AgentEval" / "train" / "textcraft_train.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_SIM_ROUNDS = 20
ITEM_TAG_RE = re.compile(r"^<ITEM_IDX:(\d+)>$")

MAX_ROUNDS_SCHEDULE: list[tuple[int, int]] | None = None


# -----------------------------------------------------------------------------
# ScalingInter curriculum (same as 07_*)
# -----------------------------------------------------------------------------

def parse_max_rounds_schedule(spec: str) -> list[tuple[int, int]]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[tuple[int, int]] = []
    for p in parts:
        rounds_str, step_str = p.split(":")
        out.append((int(step_str), int(rounds_str)))
    out.sort(key=lambda x: x[0])
    return out


def current_max_rounds(trainer: Any) -> int:
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


# -----------------------------------------------------------------------------
# Helpers (item index encoding, action parsing) — copy of 07_*
# -----------------------------------------------------------------------------

def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int, items_offset: int = 0) -> list[dict[str, Any]]:
    """Build prompt rows from textcraft_train.json[items_offset : items_offset+max_items]."""
    with TRAIN_PATH.open() as f:
        rows = json.load(f)
    if items_offset > 0:
        rows = rows[items_offset:]
    if max_items > 0:
        rows = rows[:max_items]

    probe = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(rows), timeout=60)
    manual_human = probe.conversation_start[0]["value"]
    manual_ack = probe.conversation_start[1]["value"]

    out: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        idx = item_id_to_idx(item_id)
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
        a = " ".join(a.strip().split())
        if a:
            out.append(a)
    return out


def first_action_or_empty(text: str) -> str:
    acts = extract_actions(text)
    return acts[0] if acts else ""


# -----------------------------------------------------------------------------
# SCPO reward — action-signature self-consistency
# -----------------------------------------------------------------------------
#
# Idea (adapted from SCPO / self-consistency CoT): if no verifiable reward is
# available, sample N=4 trajectories from the same prompt, extract their action
# signature (the sequence of (action_type, target) pairs), and score each
# trajectory by its average weighted similarity to the other N-1 peers.
# Trajectories that agree with the majority get higher rewards — the modal
# trajectory wins. No env feedback needed beyond what the rollout already does.
#
# Action-type weighting reflects "importance" in TextCraft:
#   craft 2.0   -- crafting an item is the only way to score
#   get   1.0   -- acquiring a base material is necessary but cheaper
#   inv.  0.5   -- inspecting inventory has no state impact
#   inval 0.3   -- mis-formed actions are noisy peers
# Final-turn bonus ×1.5 because the LAST action is usually the one that scores
# in TextCraft, so we reward agreement on the endgame more than on the prefix.

ACTION_WEIGHTS = {"craft": 2.0, "get": 1.0, "inventory": 0.5, "invalid": 0.3}
FINAL_BONUS = 1.5


def parse_action_tuple(action_text: str) -> tuple[str, str | None]:
    """Classify an extracted Action: payload into (type, normalized target)."""
    s = action_text.strip().lower()
    s = re.sub(r"^action:\s*", "", s)
    if s == "inventory":
        return ("inventory", None)
    m = re.match(r"^craft\s+\d*\s*(.+?)\s+using\s+", s)
    if m:
        target = m.group(1).strip()
        target = re.sub(r"\s+", "_", target)
        return ("craft", target)
    m = re.match(r"^get\s+\d+\s+(.+)$", s)
    if m:
        target = m.group(1).strip()
        target = re.sub(r"\s+", "_", target)
        return ("get", target)
    return ("invalid", None)


def extract_signature(text: str) -> list[tuple[str, str | None]]:
    return [parse_action_tuple(a) for a in extract_actions(text)]


def weighted_similarity(
    sig_a: list[tuple[str, str | None]],
    sig_b: list[tuple[str, str | None]],
) -> float:
    """Position-aligned weighted overlap between two action signatures.

    - Both signatures are padded with a sentinel ('none', None) to length
      max(L_a, L_b). The sentinel contributes 0 weight when paired with
      another sentinel, but the normal type weight when paired with a real
      action (so longer trajectories pay a price for tail actions that the
      shorter peer never took).
    - Each non-sentinel position's weight is max(weight[type_a], weight[type_b]).
    - The position with index L-1 gets an extra FINAL_BONUS factor when both
      sides have a real action there (endgame agreement matters more).
    - A position contributes its weight to the matched-mass iff (type, target)
      match exactly on both sides.
    Returns matched_mass / total_mass in [0, 1].
    """
    L = max(len(sig_a), len(sig_b))
    if L == 0:
        return 0.0
    pad = ("none", None)
    a = list(sig_a) + [pad] * (L - len(sig_a))
    b = list(sig_b) + [pad] * (L - len(sig_b))

    total = 0.0
    matched = 0.0
    for k in range(L):
        ta, tb = a[k][0], b[k][0]
        if ta == "none" and tb == "none":
            continue
        w = max(ACTION_WEIGHTS.get(ta, 0.3), ACTION_WEIGHTS.get(tb, 0.3))
        if k == L - 1 and ta != "none" and tb != "none":
            w *= FINAL_BONUS
        total += w
        if a[k] == b[k] and ta != "none":
            matched += w
    return matched / total if total > 0 else 0.0


def _prompt_key(prompt: Any) -> str:
    """Stable hashable key for grouping completions by source prompt."""
    if isinstance(prompt, list):
        try:
            return json.dumps(prompt, sort_keys=True, default=str)
        except Exception:
            return str(prompt)
    return str(prompt)


# -----------------------------------------------------------------------------
# Rollout — copy of textcraft_rollout_func from 07_*
# -----------------------------------------------------------------------------

def textcraft_rollout_func(
    prompts: list[list[dict[str, str]]], trainer: GRPOTrainer
) -> dict[str, Any]:
    tokenizer = trainer.processing_class

    n = len(prompts)
    states = [list(p) for p in prompts]
    done = [False] * n
    final_rewards = [0.0] * n
    invalid_counts = [0] * n
    actions_per_turn_sum = [0] * n
    n_active_turns = [0] * n
    env_clients: list[TextCraftEnvClient] = []

    for i in range(n):
        marker = str(states[i][-1]["content"]).strip()
        m = ITEM_TAG_RE.match(marker)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}] last user message: {marker!r}")
        idx = int(m.group(1))

        env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(idx)
        obs = env.observe()
        states[i][-1]["content"] = obs
        env_clients.append(env)

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
            rendered = tokenizer.apply_chat_template(
                states[i], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            round_prompt_ids.append(prompt_ids)
            if round_idx == 0:
                prompt_ids_out[i] = prompt_ids

        round_completion_ids, round_logprobs = trainer._generate_single_turn(
            round_prompt_ids, images=None, multimodal_fields={}
        )

        for j, i in enumerate(active):
            ids = round_completion_ids[j]
            lps = round_logprobs[j] if round_logprobs is not None else [0.0] * len(ids)

            text = tokenizer.decode(ids, skip_special_tokens=True)
            states[i].append({"role": "assistant", "content": text})

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

    for i in range(n):
        if not completion_ids_out[i]:
            completion_ids_out[i] = [tokenizer.eos_token_id]
            logprobs_out[i] = [0.0]
        if not prompt_ids_out[i]:
            prompt_ids_out[i] = [tokenizer.eos_token_id]

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
        "episode_reward": final_rewards,
        "invalid_steps": invalid_counts,
        "mean_n_actions_per_turn": mean_actions_per_turn,
    }


# -----------------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------------

def textcraft_env_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    mean_n_actions_per_turn: list[float],
    **kwargs: Any,
) -> list[float]:
    """v3 reward (same as 07_*): verifiable env signal + light shaping."""
    rewards: list[float] = []
    debug_rows: list[str] = []
    for comp, base_rew, bad_steps, mean_n in zip(
        completions, episode_reward, invalid_steps, mean_n_actions_per_turn, strict=True
    ):
        r = float(base_rew)
        if 0.9 <= mean_n <= 1.1:
            shape = 0.02
        elif mean_n > 1.5 or mean_n < 0.5:
            shape = -0.05
        else:
            shape = 0.0
        r += shape
        r -= 0.01 * float(bad_steps)
        rewards.append(r)
        debug_rows.append(
            f"base={base_rew:+.3f} mean_n={mean_n:.2f} shape={shape:+.3f} "
            f"bad={bad_steps} -> r={r:+.3f}"
        )
    print("[reward-env] " + " | ".join(debug_rows), flush=True)
    return rewards


def scpo_action_similarity_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    mean_n_actions_per_turn: list[float],
    **kwargs: Any,
) -> list[float]:
    """SCPO-style self-consistency reward over peer trajectories.

    For each (prompt, completion), reward = mean weighted_similarity to the
    other N-1 completions of the *same* prompt. We deliberately ignore
    ``episode_reward`` so the reward signal contains no env supervision; this
    is what makes T3-B's comparison with T3-A meaningful (same prompts, same
    rollout machinery, different supervision).

    Note: ``invalid_steps`` and ``mean_n_actions_per_turn`` are accepted in
    the signature to match the rollout's reward_kwargs but not used here.
    """
    n = len(completions)
    if n < 2:
        return [0.0] * n

    sigs = [extract_signature(completion_to_text(c)) for c in completions]
    keys = [_prompt_key(p) for p in prompts]

    rewards = [0.0] * n
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)

    debug_rows: list[str] = []
    for k, indices in by_key.items():
        if len(indices) < 2:
            debug_rows.append(f"group_size=1 -> reward=0 (degenerate)")
            continue
        for i in indices:
            sims = []
            for j in indices:
                if i == j:
                    continue
                sims.append(weighted_similarity(sigs[i], sigs[j]))
            rewards[i] = sum(sims) / len(sims)
        group_mean = sum(rewards[i] for i in indices) / len(indices)
        group_max = max(rewards[i] for i in indices)
        group_min = min(rewards[i] for i in indices)
        debug_rows.append(
            f"group_size={len(indices)} mean_sim={group_mean:.3f} "
            f"min={group_min:.3f} max={group_max:.3f}"
        )

    # Light per-trajectory shaping to penalize purely invalid sequences so SCPO
    # doesn't reward a "modal failure" of N trajectories that all syntax-fail
    # the same way.
    for i in range(n):
        if all(t[0] == "invalid" for t in sigs[i]) and len(sigs[i]) > 0:
            rewards[i] = max(rewards[i] - 0.1, -0.1)

    print(f"[reward-scpo] {' | '.join(debug_rows)}", flush=True)
    return rewards


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--init-from-lora",
        type=str,
        default="",
        help=(
            "Path to a LoRA adapter directory to resume training from "
            "(e.g. saves/trl_grpo/trl_grpo_v4_scalinginter_step50/checkpoint-50). "
            "Empty string starts from base model with a fresh LoRA."
        ),
    )
    parser.add_argument(
        "--items-offset",
        type=int,
        default=0,
        help=(
            "Skip the first N items of textcraft_train.json before slicing "
            "max_items. T3 uses --items-offset 256 to train on items 256..511, "
            "leaving items 0..255 (used by v3/v4) as the in-distribution test."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=["env", "scpo"],
        default="env",
        help=(
            "env  = verifiable TextCraft success + v3 shaping (T3-A baseline). "
            "scpo = action-signature self-consistency, ignores env reward (T3-B)."
        ),
    )
    parser.add_argument("--max-rounds-schedule", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-steps", type=int, default=10)
    args = parser.parse_args()

    if args.max_rounds_schedule:
        global MAX_ROUNDS_SCHEDULE
        MAX_ROUNDS_SCHEDULE = parse_max_rounds_schedule(args.max_rounds_schedule)
        print(f"[scaling-inter] schedule = {MAX_ROUNDS_SCHEDULE}", flush=True)

    gen_batch = args.num_generations
    print(
        f"[config] reward_mode={args.reward_mode} "
        f"items_offset={args.items_offset} max_items={args.max_items} "
        f"num_generations={args.num_generations} "
        f"init_from_lora={args.init_from_lora or '(none, fresh LoRA)'}",
        flush=True,
    )

    check_server()
    rows = build_prompt_rows(max_items=args.max_items, items_offset=args.items_offset)
    print(f"[data] loaded {len(rows)} prompts (items {args.items_offset}..{args.items_offset + len(rows) - 1})", flush=True)
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = REPO_ROOT / "saves" / "trl_grpo" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.init_from_lora:
        print(f"[init] loading base model from {MODEL_PATH}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"[init] loading LoRA adapter from {args.init_from_lora}", flush=True)
        model = PeftModel.from_pretrained(
            base, args.init_from_lora, is_trainable=True
        )
        # Needed so grads flow through PEFT-wrapped base when gradient
        # checkpointing is enabled (PEFT inputs would otherwise be detached).
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        peft_cfg: LoraConfig | None = None
        model_init_kwargs: dict[str, Any] = {}
        model_arg: Any = model
    else:
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model_init_kwargs = {"torch_dtype": "bfloat16", "low_cpu_mem_usage": True}
        model_arg = str(MODEL_PATH)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        report_to=[],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        generation_batch_size=gen_batch,
        max_completion_length=128,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="no",
        gradient_checkpointing=True,
        model_init_kwargs=model_init_kwargs if model_init_kwargs else None,
    )

    reward_func = (
        scpo_action_similarity_reward
        if args.reward_mode == "scpo"
        else textcraft_env_reward
    )
    print(f"[reward] using {reward_func.__name__}", flush=True)

    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=reward_func,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
        rollout_func=textcraft_rollout_func,
    )

    trainer.train()
    print(f"[done] continued GRPO training finished -> {out_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    main()
