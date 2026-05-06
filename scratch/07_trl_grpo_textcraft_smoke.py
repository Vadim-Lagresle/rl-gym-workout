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
        probe.reset(idx)
        obs = probe.observe()
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": manual_human},
            {"role": "assistant", "content": manual_ack},
            {"role": "user", "content": obs},
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


_REWARD_ENV: TextCraftEnvClient | None = None


def textcraft_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    item_id: list[str],
    item_idx: list[int],
    **kwargs: Any,
) -> list[float]:
    global _REWARD_ENV
    if _REWARD_ENV is None:
        _REWARD_ENV = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
    env = _REWARD_ENV
    rewards: list[float] = []

    for comp, idx in zip(completions, item_idx, strict=True):
        text = completion_to_text(comp)
        env.reset(int(idx))
        actions = extract_actions(text)

        # Base shaping: syntax quality of the full completion
        if len(actions) == 0:
            rewards.append(-0.10)
            continue

        # Multi-turn simulated execution: replay extracted actions sequentially.
        r = 0.0
        done = False
        invalid_steps = 0
        repeated = 0
        prev_action = None

        for act in actions[:MAX_SIM_ROUNDS]:
            if prev_action is not None and act == prev_action:
                repeated += 1
            prev_action = act

            step = env.step(f"Action: {act}")
            obs = step.state.lower()
            done = bool(step.done)
            r = max(r, float(step.reward))  # keep sparse success reward if reached

            # Process shaping from env feedback
            if "could not" in obs or "error:" in obs or "wrong item format" in obs:
                invalid_steps += 1
                r -= 0.02
            else:
                r += 0.01

            if done:
                break

        # Penalize loops/repetitions and overlong completions
        if repeated >= 3:
            r -= 0.05
        if len(actions) > MAX_SIM_ROUNDS:
            r -= 0.05

        # Small bonus if solved in few actions
        if done and r > 0:
            r += 0.05

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
        save_strategy="no",
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
    )

    trainer.train()
    print("[smoke] TRL+GRPO training run finished")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    main()
