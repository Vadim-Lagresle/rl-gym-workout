"""
Évaluation de Qwen2.5-3B-Instruct + LoRA sur TextCraft.

Variante de `scratch/03_eval_qwen.py` qui charge un adaptateur LoRA produit par
notre training TRL+GRPO (`scratch/07_trl_grpo_textcraft_smoke.py`). Logs sortis
dans un répertoire dédié pour ne pas écraser ceux du baseline non-LoRA.

Pré-requis :
  1. Serveur TextCraft sur 127.0.0.1:36005.
  2. Conda env `agentgym-rl` activé.
  3. Un checkpoint LoRA peft valide (contient `adapter_config.json` +
     `adapter_model.safetensors`).

Lancement (exemples) :
    conda activate agentgym-rl
    LORA_PATH=saves/trl_grpo/trl_grpo_rolloutfunc_v2_step10/checkpoint-10 \\
        python scratch/08_eval_qwen_lora.py
    LORA_PATH=... MAX_ITEMS=30 python scratch/08_eval_qwen_lora.py
    LORA_PATH=... EVAL_TAG=step10 python scratch/08_eval_qwen_lora.py
    LORA_PATH=... FORCE_REDO=1 python scratch/08_eval_qwen_lora.py

Logs cibles : `scratch/eval_logs_<EVAL_TAG>/<item_id>.json`. Si EVAL_TAG n'est
pas fourni, on dérive un tag du nom du checkpoint LoRA (par ex. `checkpoint-10`).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentenv.envs import TextCraftEnvClient
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"

ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ROUNDS = 30
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))
FORCE_REDO = bool(int(os.environ.get("FORCE_REDO", "0")))

# LoRA-specific config — required.
_lora_path_str = os.environ.get("LORA_PATH")
if not _lora_path_str:
    raise SystemExit(
        "[eval-lora] LORA_PATH env var is required, "
        "e.g. saves/trl_grpo/trl_grpo_rolloutfunc_v2_step10/checkpoint-10"
    )
LORA_PATH = (REPO_ROOT / _lora_path_str).resolve() if not Path(_lora_path_str).is_absolute() else Path(_lora_path_str)
if not (LORA_PATH / "adapter_config.json").exists():
    raise SystemExit(f"[eval-lora] {LORA_PATH} does not look like a peft LoRA dir (missing adapter_config.json)")

EVAL_TAG = os.environ.get("EVAL_TAG") or LORA_PATH.name
LOG_DIR = REPO_ROOT / "runs" / f"eval_logs_{EVAL_TAG}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# vLLM LoRARequest: (lora_name, lora_int_id, lora_local_path).
LORA_REQUEST = LoRARequest("textcraft_lora", 1, str(LORA_PATH))


@dataclass
class EpisodeResult:
    item_id: str
    item_idx: int
    reward: float
    done: bool
    rounds: int
    duration_s: float
    transcript: list[dict] = field(default_factory=list)


def build_initial_messages(client: TextCraftEnvClient) -> list[dict]:
    rules_msg = client.conversation_start[0]["value"]
    ack_msg = client.conversation_start[1]["value"]
    initial_obs = client.observe()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rules_msg},
        {"role": "assistant", "content": ack_msg},
        {"role": "user", "content": initial_obs},
    ]


def item_id_to_idx(item_id: str) -> int:
    m = re.search(r"(\d+)$", item_id)
    if not m:
        raise ValueError(f"Cannot parse item_id: {item_id!r}")
    return int(m.group(1))


def run_episode(
    llm: LLM,
    tokenizer,
    client: TextCraftEnvClient,
    sampling: SamplingParams,
    item_id: str,
    max_rounds: int = MAX_ROUNDS,
) -> EpisodeResult:
    item_idx = item_id_to_idx(item_id)
    client.reset(item_idx)
    messages = build_initial_messages(client)
    t0 = time.time()
    reward = 0.0
    done = False
    rounds = 0

    for round_idx in range(max_rounds):
        rounds = round_idx + 1
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = llm.generate(
            [prompt], sampling, use_tqdm=False, lora_request=LORA_REQUEST
        )
        assistant_text = outputs[0].outputs[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        step_out = client.step(assistant_text)
        reward = float(step_out.reward)
        done = bool(step_out.done)
        messages.append({"role": "user", "content": step_out.state})

        if done:
            break

    return EpisodeResult(
        item_id=item_id,
        item_idx=item_idx,
        reward=reward,
        done=done,
        rounds=rounds,
        duration_s=time.time() - t0,
        transcript=messages,
    )


def write_episode_log(result: EpisodeResult) -> None:
    fp = LOG_DIR / f"{result.item_id}.json"
    payload = {
        "item_id": result.item_id,
        "item_idx": result.item_idx,
        "reward": result.reward,
        "done": result.done,
        "rounds": result.rounds,
        "duration_s": round(result.duration_s, 2),
        "lora_path": str(LORA_PATH),
        "transcript": result.transcript,
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def load_existing_log(item_id: str) -> EpisodeResult | None:
    fp = LOG_DIR / f"{item_id}.json"
    if not fp.exists():
        return None
    try:
        with fp.open() as f:
            d = json.load(f)
        required = {"item_id", "item_idx", "reward", "done", "rounds", "duration_s"}
        if not required.issubset(d):
            return None
        return EpisodeResult(
            item_id=d["item_id"],
            item_idx=d["item_idx"],
            reward=float(d["reward"]),
            done=bool(d["done"]),
            rounds=int(d["rounds"]),
            duration_s=float(d["duration_s"]),
            transcript=d.get("transcript", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def main() -> None:
    print(f"[eval-lora] LORA_PATH = {LORA_PATH}")
    print(f"[eval-lora] EVAL_TAG  = {EVAL_TAG}")
    print(f"[eval-lora] LOG_DIR   = {LOG_DIR}")

    with DATASET_PATH.open() as f:
        items = json.load(f)
    if MAX_ITEMS > 0:
        items = items[:MAX_ITEMS]

    cached: list[EpisodeResult] = []
    todo: list[dict] = []
    for it in items:
        existing = None if FORCE_REDO else load_existing_log(it["item_id"])
        if existing is not None:
            cached.append(existing)
        else:
            todo.append(it)

    print(
        f"[eval-lora] {len(items)} total items. "
        f"{len(cached)} cached (skipped), {len(todo)} to run."
    )

    new_results: list[EpisodeResult] = []
    if todo:
        print("[eval-lora] Loading vLLM with LoRA support (this takes ~10-30s)...")
        # NOTE: enforce_eager=False is intentional. With LoRA enabled, vLLM's
        # Punica kernels go through a Python dispatch that is ~100x slower in
        # eager mode. Letting vLLM compile CUDA graphs masks this overhead.
        llm = LLM(
            model=str(MODEL_PATH),
            enforce_eager=False,
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            dtype="bfloat16",
            load_format="safetensors",
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank=16,
            max_loras=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512)

        client = TextCraftEnvClient(
            env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60
        )
        print(f"[eval-lora] Connected to env server (env_id={client.env_id}).")

        overall_t0 = time.time()
        for i, item in enumerate(todo):
            item_id = item["item_id"]
            try:
                r = run_episode(llm, tokenizer, client, sampling, item_id)
            except Exception as e:
                print(f"[eval-lora][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
                continue
            write_episode_log(r)
            status = "DONE" if r.done else "TIMEOUT"
            print(
                f"[eval-lora][{i+1}/{len(todo)}] {item_id} reward={r.reward} "
                f"rounds={r.rounds} dur={r.duration_s:.1f}s [{status}]"
            )
            new_results.append(r)
        print(f"\n[eval-lora] Wall time on new rollouts: {time.time() - overall_t0:.1f}s")

    all_results = cached + new_results
    if not all_results:
        print("[eval-lora] No results to summarize.")
        return
    avg_reward = sum(r.reward for r in all_results) / len(all_results)
    pass_rate = sum(1 for r in all_results if r.reward > 0) / len(all_results)
    print("=" * 50)
    print(f"GLOBAL STATS over {len(all_results)} items")
    print(f"  ({len(cached)} cached + {len(new_results)} fresh)")
    print(f"  LoRA: {LORA_PATH.name}")
    print("=" * 50)
    print(f"Avg@1   = {avg_reward:.4f}")
    print(f"Pass@1  = {pass_rate:.4f}  ({sum(1 for r in all_results if r.reward > 0)}/{len(all_results)})")
    print(f"Mean rounds = {sum(r.rounds for r in all_results) / len(all_results):.1f}")
    print(f"Mean duration = {sum(r.duration_s for r in all_results) / len(all_results):.1f}s")
    print(f"Logs: {LOG_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
