from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from agentenv.envs import TextCraftEnvClient

REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_PATH = REPO_ROOT / "AgentEval" / "train" / "textcraft_train.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_SIM_ROUNDS = 20
ITEM_TAG_RE = re.compile(r"^<ITEM_IDX:(\d+)>$")


def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int) -> list[dict[str, Any]]:
    with TRAIN_PATH.open() as f:
        rows = json.load(f)
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

    for round_idx in range(MAX_SIM_ROUNDS):
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

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        # Extra fields forwarded to reward function through reward_kwargs
        "episode_reward": final_rewards,
        "invalid_steps": invalid_counts,
    }


def textcraft_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    **kwargs: Any,
) -> list[float]:
    rewards: list[float] = []

    for comp, base_rew, bad_steps in zip(completions, episode_reward, invalid_steps, strict=True):
        text = completion_to_text(comp)
        r = float(base_rew)
        # Shape with syntax info from generated text + env invalid step count.
        n_actions = count_actions(text)
        if n_actions == 0:
            r -= 0.05
        elif n_actions == 1:
            r += 0.02
        else:
            r -= 0.05
        r -= 0.01 * float(bad_steps)

        rewards.append(r)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--run-name", type=str, default="trl_grpo_textcraft_smoke")
    args = parser.parse_args()

    check_server()
    rows = build_prompt_rows(max_items=args.max_items)
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        generation_batch_size=2,
        max_completion_length=128,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=3,
        eval_strategy="no",
        gradient_checkpointing=True,
        model_init_kwargs={"torch_dtype": "bfloat16", "low_cpu_mem_usage": True},
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=textcraft_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=textcraft_rollout_func,
    )

    trainer.train()
    print("[smoke] TRL+GRPO training run finished")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    main()
