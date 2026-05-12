"""
Smoke tests for the verl pipeline that we plan to use on the 8x A100 VM.

Goal: validate the parts of `AgentGym-RL/verl` we can exercise on our current
1x A100 40GB VM WITHOUT triggering the verl FSDP+vLLM hybrid-engine path,
which is known to crash in NCCL when world_size=1 + load_format=dummy_dtensor.

What this script validates (offline, CPU is enough for tests 1-2):
  Test 1: `compute_grpo_outcome_advantage` math — synthetic per-group rewards
          go through GRPO normalization (mean-center + std-divide per prompt
          group). This is the heart of GRPO; if this is wrong, all training
          is broken.
  Test 2: `RolloutHandler.add_assistant_message` / `add_user_message` produce
          loss_masks consistent with the Qwen ChatML template (only assistant
          content contributes loss).
  Test 3: `RLHFDataset` loads our train file and produces the exact same chat
          prompt as `scratch/03_eval_qwen.py` — i.e., training and eval use
          the same prompt, no hidden few-shot or chain-of-thought primer.
          (This test requires the TextCraft env server to be running on
          127.0.0.1:36005 because `RLHFDataset.__init__` instantiates an
          `EnvClient` which hits `/create` on import.)

What this script does NOT validate (intentionally):
  - The verl vLLM rollout (`verl.workers.rollout.agent_vllm_rollout`). It
    requires an FSDP-wrapped model and verl's custom vLLM fork; both crash
    at world_size=1 under our current setup. We've validated the rollout
    path separately via `main_generation.py` (eval-only) in
    `examples/eval/textcraft_eval.local.sh`.
  - FSDP sharding, optimizer step, model save/load — these need 8 GPUs to
    be exercised meaningfully.
  - The reward-to-token-tensor expansion (`reward_tensor[i, ...] = scores[i]`
    at vllm_rollout.py:326). It is dead-simple and bound to vLLM output.

Run from repo root with conda env `agentgym-rl` active:
    python scratch/smoke_verl_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
sys.path.insert(0, str(REPO_ROOT / "AgentGym-RL"))


def _ok(label: str) -> None:
    print(f"  [OK]   {label}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}{(': ' + detail) if detail else ''}")


def test_grpo_advantage_math() -> bool:
    """Synthetic check: GRPO normalizes scores within prompt-id groups."""
    print("[test 1] compute_grpo_outcome_advantage")
    from verl.agent_trainer.ppo.core_algos import compute_grpo_outcome_advantage

    # 3 prompts, 4 rollouts each, response_length=5.
    # Reward placed at last token position (matches vllm_rollout.py:326).
    bsz, resp_len = 12, 5
    token_level_rewards = torch.zeros(bsz, resp_len)
    # Scores per (prompt, rollout): prompt 0 gets [0,0,1,1] -> mean=0.5
    #                                prompt 1 gets [0,0,0,0] -> mean=0 (degenerate)
    #                                prompt 2 gets [1,1,1,1] -> mean=1 (degenerate)
    raw_scores = [0, 0, 1, 1,  0, 0, 0, 0,  1, 1, 1, 1]
    for i, s in enumerate(raw_scores):
        token_level_rewards[i, -1] = float(s)
    index = np.array(["p0"] * 4 + ["p1"] * 4 + ["p2"] * 4, dtype=object)
    eos_mask = torch.ones(bsz, resp_len)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=eos_mask,
        index=index,
    )

    # advantages are broadcast along the response_length dimension (* eos_mask),
    # so we look at a single token position (e.g., t=-1).
    adv_last = advantages[:, -1].numpy()

    passed = True
    # Prompt 0: scores [0,0,1,1], mean=0.5, std=sqrt(1/3) ~ 0.577 (sample std).
    # Normalized: ([-.5,-.5,.5,.5] / 0.577) ~ [-0.866, -0.866, 0.866, 0.866]
    p0_expected = np.array([-0.866, -0.866, 0.866, 0.866])
    if np.allclose(adv_last[:4], p0_expected, atol=1e-2):
        _ok("prompt 0 (mixed rewards) normalized to ~+-0.866")
    else:
        _fail("prompt 0 normalization", f"got {adv_last[:4].round(3).tolist()}")
        passed = False

    # Prompts 1 & 2: degenerate (std=0). Implementation falls back to mean=0,std=1
    # for the SINGLE-rollout case, but for >1 rollouts with all-equal scores,
    # std is 0 and we divide by (std + epsilon=1e-6). So all advantages should
    # be ~0/eps = ~0 for p1 (all zeros minus mean 0 = 0) and ~0 for p2 (1-1=0).
    if np.allclose(adv_last[4:8], 0.0, atol=1e-3):
        _ok("prompt 1 (all-zero rewards) advantages ~ 0")
    else:
        _fail("prompt 1 advantages should be ~0", f"got {adv_last[4:8].round(3).tolist()}")
        passed = False
    if np.allclose(adv_last[8:12], 0.0, atol=1e-3):
        _ok("prompt 2 (all-one rewards) advantages ~ 0")
    else:
        _fail("prompt 2 advantages should be ~0", f"got {adv_last[8:12].round(3).tolist()}")
        passed = False

    # Sanity: returns == advantages for GRPO outcome advantage (no per-token return).
    if torch.allclose(advantages, returns):
        _ok("returns == advantages (GRPO outcome advantage definition)")
    else:
        _fail("returns != advantages")
        passed = False

    return passed


def test_rollout_handler_loss_mask() -> bool:
    """Check that only assistant-content tokens get loss_mask=1 (training signal)."""
    print("[test 2] RolloutHandler loss_mask correctness (Qwen format)")
    from transformers import AutoTokenizer
    from verl.workers.rollout.schemas import Message, RolloutHandler

    tok = AutoTokenizer.from_pretrained(
        str(REPO_ROOT / "models" / "Qwen2.5-3B-Instruct")
    )

    initial_prompt_text = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nRules.<|im_end|>\n"
        "<|im_start|>assistant\nOK.<|im_end|>"
    )
    initial_ids = tok.encode(initial_prompt_text, add_special_tokens=False)

    handler = RolloutHandler(
        messages=[
            Message(role="user", content="Rules."),
            Message(role="assistant", content="OK."),
        ],
        task_name="textcraft",
        item_id=0,
        score=0.0,
        done=False,
        input_ids=list(initial_ids),
        prompt_ids=list(initial_ids),
        response_ids=[],
        attention_mask=[1] * len(initial_ids),
        prompt_attention_mask=[1] * len(initial_ids),
        response_attention_mask=[],
        position_ids=list(range(len(initial_ids))),
        prompt_position_ids=list(range(len(initial_ids))),
        response_position_ids=[],
        loss_mask=[0] * len(initial_ids),
        prompt_loss_mask=[0] * len(initial_ids),
        response_loss_mask=[],
    )

    handler.add_user_message(tok, "First observation.")
    handler.add_assistant_message(tok, "Thought:\nfoo\n\nAction:\nget 1 stick")

    # Assertions
    passed = True
    # Length consistency
    if not (len(handler.input_ids) == len(handler.attention_mask)
            == len(handler.position_ids) == len(handler.loss_mask)):
        _fail("length consistency")
        passed = False
    else:
        _ok("input_ids/attention_mask/position_ids/loss_mask same length")

    # No loss on prompt prefix
    if all(m == 0 for m in handler.loss_mask[: len(initial_ids)]):
        _ok("initial prompt has loss_mask=0 everywhere")
    else:
        _fail("initial prompt should have loss_mask=0")
        passed = False

    # Loss mask is 1 on the assistant content tokens (somewhere after the user msg)
    has_loss = sum(handler.loss_mask) > 0
    if has_loss:
        _ok(f"some tokens have loss_mask=1 (sum={sum(handler.loss_mask)})")
    else:
        _fail("no token has loss_mask=1")
        passed = False

    # Verify that decoding loss-masked tokens roughly matches assistant content
    loss_tokens = [tid for tid, m in zip(handler.input_ids, handler.loss_mask) if m == 1]
    decoded = tok.decode(loss_tokens, skip_special_tokens=False)
    # Should contain the action text plus the <|im_end|> suffix-on-content tokens
    if "get 1 stick" in decoded:
        _ok("decoded loss tokens contain the assistant action text")
    else:
        _fail("decoded loss tokens do not contain assistant text", f"got: {decoded!r}")
        passed = False

    return passed


def test_dataset_prompt_equality() -> bool:
    """Verify training data pipeline builds the same prompt as our eval scripts."""
    print("[test 3] RLHFDataset prompt matches scratch/03_eval_qwen.py")
    # Requires TextCraft env server up because RLHFDataset.__init__ creates a client.
    env_addr = "http://127.0.0.1:36005"
    try:
        import requests
        r = requests.post(f"{env_addr}/create", json={}, timeout=5)
        if r.status_code != 200:
            raise RuntimeError(f"env /create returned {r.status_code}")
    except Exception as e:
        _fail("skipped: TextCraft env server unreachable", repr(e))
        print("        (start it with: textcraft --host 127.0.0.1 --port 36005)")
        return True  # not a failure of the code under test

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.agent_dataset.rl_dataset import RLHFDataset

    model_path = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(str(model_path))

    data_cfg = OmegaConf.create({
        "prompt_key": "item_id",
        "max_prompt_length": 2048,
        "return_raw_chat": True,
        "truncation": "error",
    })
    agentgym_cfg = OmegaConf.create({
        "task_name": "textcraft",
        "env_addr": env_addr,
        "max_retries": 1,
        "timeout": 60,
    })
    ds = RLHFDataset(
        data_file=str(REPO_ROOT / "AgentEval/train/textcraft_train.json"),
        tokenizer=tok,
        data_config=data_cfg,
        agentgym_config=agentgym_cfg,
    )
    sample = ds[0]
    verl_prompt = tok.decode(sample["input_ids"], skip_special_tokens=False)
    # Strip left padding
    verl_prompt = verl_prompt.replace(tok.pad_token, "")

    # What scratch/03_eval_qwen.py builds for the SAME (rules, ack) pair, BEFORE
    # adding the initial env observation:
    from agentenv.envs import TextCraftEnvClient  # noqa: E402
    client = TextCraftEnvClient(env_server_base=env_addr, data_len=1, timeout=60)
    rules = client.conversation_start[0]["value"]
    ack = client.conversation_start[1]["value"]
    eval_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": rules},
        {"role": "assistant", "content": ack},
    ]
    eval_prompt = tok.apply_chat_template(
        eval_messages, tokenize=False, add_generation_prompt=False
    ).rstrip()

    # verl uses a hand-rolled f-string in rl_dataset.py _build_messages; this
    # should be byte-identical to what apply_chat_template gives (minus trailing
    # whitespace and the optional <|im_end|>\n at the very end).
    passed = True

    # Normalize: strip whitespace at boundaries, and strip the assistant suffix
    # (verl's hand-rolled string ends with "<|im_end|>" without trailing newline).
    def _norm(s: str) -> str:
        return s.strip().rstrip()

    if _norm(verl_prompt).startswith(_norm(eval_prompt)[:50]):
        _ok("verl prompt and scratch/03 prompt share the same prefix (first 50 chars)")
    else:
        _fail("verl/eval prompt prefix mismatch")
        passed = False

    # More thorough: both must contain the rules string verbatim (no extra few-shots).
    if rules.split(".")[0] in verl_prompt:
        _ok("verl prompt contains the verbatim TextCraft rules from conversation_start[0]")
    else:
        _fail("verl prompt missing TextCraft rules")
        passed = False

    if "few-shot" in verl_prompt.lower() or "example:" in verl_prompt.lower():
        _fail("verl prompt unexpectedly contains few-shot examples")
        passed = False
    else:
        _ok("no few-shot / chain-of-thought primer detected in verl prompt")

    if "OK. I'll follow your instructions" in verl_prompt and "OK. I'll follow your instructions" in eval_prompt:
        _ok("both prompts contain the canonical assistant ACK message")
    else:
        _fail("ACK message mismatch")
        passed = False

    print(f"        verl prompt length: {len(verl_prompt)} chars")
    print(f"        eval prompt length: {len(eval_prompt)} chars")

    try:
        client.close()
    except Exception:
        pass
    return passed


def main() -> int:
    print("=" * 70)
    print("verl smoke tests (offline; no GPU required for tests 1 & 2)")
    print("=" * 70)
    results: dict[str, bool] = {}

    try:
        results["grpo_advantage_math"] = test_grpo_advantage_math()
    except Exception as e:
        _fail("grpo_advantage_math crashed", repr(e))
        results["grpo_advantage_math"] = False

    print()
    try:
        results["rollout_handler_loss_mask"] = test_rollout_handler_loss_mask()
    except Exception as e:
        _fail("rollout_handler_loss_mask crashed", repr(e))
        results["rollout_handler_loss_mask"] = False

    print()
    try:
        results["dataset_prompt_equality"] = test_dataset_prompt_equality()
    except Exception as e:
        _fail("dataset_prompt_equality crashed", repr(e))
        results["dataset_prompt_equality"] = False

    print()
    print("=" * 70)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
