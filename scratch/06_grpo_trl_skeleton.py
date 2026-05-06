"""Plan B skeleton: TRL + GRPO for TextCraft multi-turn training.

This is intentionally a scaffold (not runnable end-to-end yet).
Goal: encode the architecture and TODOs so we can quickly switch away from
`verl` after the single-GPU smoke crash.

What this file already contains:
- Dataset loading (AgentEval train split)
- Env client bootstrap (TextCraft server)
- Rollout function signature for multi-turn episodes
- Reward extraction contract for GRPO
- Integration points for TRL GRPOTrainer

What remains TODO:
- Wire model/tokenizer into GRPOTrainer with correct generation kwargs
- Feed batched prompts and N rollouts/group
- Backprop/update loop + periodic eval on test split
- Save checkpoints and metrics
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

# Existing working pieces from our eval script
from agentenv.envs import TextCraftEnvClient
from transformers import AutoTokenizer


REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_DATASET = REPO_ROOT / "AgentEval" / "train" / "textcraft_train.json"
TEST_DATASET = REPO_ROOT / "AgentEval" / "eval" / "textcraft_test.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"

# Placeholder import: uncomment when we confirm exact TRL version/API.
# from trl import GRPOConfig, GRPOTrainer


@dataclass
class EpisodeResult:
    item_id: str
    reward: float
    done: bool
    rounds: int
    transcript: list[dict[str, str]]


def load_item_ids(path: Path) -> list[str]:
    with path.open() as f:
        rows = json.load(f)
    return [r["item_id"] for r in rows]


def build_prompt_messages(client: TextCraftEnvClient) -> list[dict[str, str]]:
    """Build initial chat history exactly like eval.

    NOTE: we keep the same prompt protocol as the baseline to isolate the
    effect of RL updates.
    """
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        }
    ]
    # AgentGym's conversation_start has: manual + scripted assistant ack
    for m in client.conversation_start:
        role = "user" if m["from"] == "human" else "assistant"
        messages.append({"role": role, "content": m["value"]})
    messages.append({"role": "user", "content": client.observe()})
    return messages


def item_id_to_idx(item_id: str) -> int:
    # textcraft_143 -> 143
    return int(item_id.rsplit("_", 1)[1])


def rollout_one_episode(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    env: TextCraftEnvClient,
    item_id: str,
    max_rounds: int = 20,
) -> EpisodeResult:
    """Run one multi-turn episode and return trajectory + outcome reward.

    TODO(model.generate):
    - produce one assistant response per turn using chat template
    - keep temperature sampling for exploration in GRPO groups
    """
    idx = item_id_to_idx(item_id)
    env.reset(idx)
    messages = build_prompt_messages(env)

    # TODO: replace with actual model generation.
    # For now, this placeholder intentionally fails fast to avoid silent misuse.
    raise NotImplementedError(
        "rollout_one_episode is a skeleton. Plug TRL/transformers generation here."
    )


class TextCraftGRPORewardFn:
    """Reward function contract for GRPO.

    Input: generated trajectory (or final assistant output depending on TRL API)
    Output: scalar reward in [0, 1] for now (success/failure).

    TODO:
    - optional shaping: penalize format violations, repeated invalid actions,
      long loops, etc.
    """

    def __call__(self, episode: EpisodeResult) -> float:
        return float(episode.reward)


def build_grpo_trainer_skeleton() -> None:
    """Pseudo-wiring for future implementation.

    Steps to complete:
    1) Load train item_ids from AgentEval/train/textcraft_train.json
    2) For each prompt, sample N rollouts (group size) with current policy
    3) Score each rollout with TextCraft reward function
    4) Compute group-relative advantages (GRPO)
    5) Update policy
    6) Periodically eval on test split (100 fixed items)
    """

    train_ids = load_item_ids(TRAIN_DATASET)
    test_ids = load_item_ids(TEST_DATASET)

    print(f"[skeleton] train items: {len(train_ids)}")
    print(f"[skeleton] test items : {len(test_ids)}")

    # Keep parity with baseline tokenizer behavior.
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    # Reuse same env contract already validated.
    env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(train_ids), timeout=60)
    print(f"[skeleton] env_id={env.env_id}")

    # TODO: instantiate TRL GRPO config/trainer once API is confirmed.
    # cfg = GRPOConfig(...)
    # trainer = GRPOTrainer(model=..., args=cfg, ...)

    # TODO: custom training loop or trainer.train() depending on API.
    # for step in range(max_steps):
    #     sample batch prompts
    #     collect N rollouts per prompt via rollout_one_episode
    #     compute rewards + GRPO advantages
    #     trainer.step(...)

    print("[skeleton] ready: fill TODOs to run TRL+GRPO plan B.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    build_grpo_trainer_skeleton()
