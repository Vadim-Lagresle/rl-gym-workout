"""
⚠️  FALLBACK UNIQUEMENT — préférer src/eval/eval_vllm.py (×2.9 plus rapide, KV cache).
vLLM est désormais opérationnel sur ce serveur (0.9.1 manylinux1 + 2 patches, malgré
glibc 2.28). Ne garder ce script que comme repli si vLLM casse. Voir CLAUDE.md.

Évalue un checkpoint full fine-tuning produit par src/train/train_grpo.py sur TextCraft.

Charge le checkpoint avec transformers (AutoModelForCausalLM) — pas vLLM. Plus lent
(pas de KV cache entre les rounds), à n'utiliser que si eval_vllm.py est indisponible.

Joue 100 épisodes multi-tours, enregistre les logs dans runs/<run-name>/eval_logs/,
calcule Pass@1 à la fin. Compatible eval_baseline.py (même format de logs).

Usage :
    source ~/envs/agentgym-rl/bin/activate
    python src/eval/eval_fullft.py \\
        --checkpoint saves/trl_grpo/exp7_b200_fullft/checkpoint-564 \\
        --run-name exp7_b200_fullft

    # Smoke test (7 items)
    python src/eval/eval_fullft.py \\
        --checkpoint saves/trl_grpo/exp7_b200_fullft/checkpoint-564 \\
        --run-name exp7_b200_fullft --max-items 7

    # Forcer le recalcul
    python src/eval/eval_fullft.py ... --force-redo
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from agentenv.envs import TextCraftEnvClient
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ROUNDS = 30
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class EpisodeResult:
    item_id: str
    item_idx: int
    reward: float
    done: bool
    rounds: int
    duration_s: float
    transcript: list[dict] = field(default_factory=list)


def item_id_to_idx(item_id: str) -> int:
    m = re.search(r"(\d+)$", item_id)
    if not m:
        raise ValueError(f"Cannot parse item_id: {item_id!r}")
    return int(m.group(1))


def build_initial_messages(client: TextCraftEnvClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    rules_msg = client.conversation_start[0]["value"]
    ack_msg = client.conversation_start[1]["value"]
    initial_obs = client.observe()
    # If system_prompt is empty (e.g. Gemma-3 which rejects system role),
    # inject it at the start of the first user message instead.
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rules_msg},
            {"role": "assistant", "content": ack_msg},
            {"role": "user", "content": initial_obs},
        ]
    else:
        return [
            {"role": "user", "content": rules_msg},
            {"role": "assistant", "content": ack_msg},
            {"role": "user", "content": initial_obs},
        ]


def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 512,
                   enable_thinking: bool = True) -> str:
    # Qwen3/Qwen3.5 démarrent en "thinking mode" et émettent <think>...</think> :
    # illisible pour le parser TextCraft et ça consomme tout le budget de tokens.
    # enable_thinking=False force des actions directes. Guard: les vieux templates
    # ignorent le kwarg (TypeError) → on retombe sur l'appel sans le kwarg.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_episode(
    model,
    tokenizer,
    client: TextCraftEnvClient,
    item_id: str,
    max_rounds: int = MAX_ROUNDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enable_thinking: bool = True,
) -> EpisodeResult:
    item_idx = item_id_to_idx(item_id)
    client.reset(item_idx)
    messages = build_initial_messages(client, system_prompt=system_prompt)
    t0 = time.time()
    reward = 0.0
    done = False
    rounds = 0

    for round_idx in range(max_rounds):
        rounds = round_idx + 1
        assistant_text = generate_reply(model, tokenizer, messages, enable_thinking=enable_thinking)
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


def write_episode_log(result: EpisodeResult, log_dir: Path) -> None:
    fp = log_dir / f"{result.item_id}.json"
    payload = {
        "item_id": result.item_id,
        "item_idx": result.item_idx,
        "reward": result.reward,
        "done": result.done,
        "rounds": result.rounds,
        "duration_s": round(result.duration_s, 2),
        "transcript": result.transcript,
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def load_existing_log(item_id: str, log_dir: Path) -> EpisodeResult | None:
    fp = log_dir / f"{item_id}.json"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to full-FT checkpoint dir")
    parser.add_argument("--run-name", required=True, help="Run name for log dir (runs/<run-name>/eval_logs/)")
    parser.add_argument("--max-items", type=int, default=0, help="0 = all 100 items")
    parser.add_argument("--force-redo", action="store_true", help="Ignore cached logs")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt. Pass '' for models without system role (e.g. Gemma-3).")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Force non-thinking mode (Qwen3/3.5): enable_thinking=False dans le chat template.")
    args = parser.parse_args()

    # --checkpoint accepte un chemin local OU un id HF Hub (ex: Qwen/Qwen3.5-4B).
    local_path = Path(args.checkpoint)
    if not local_path.is_absolute():
        local_path = REPO_ROOT / args.checkpoint
    if local_path.exists():
        model_ref = str(local_path)
    else:
        model_ref = args.checkpoint  # traité comme id HF Hub
        print(f"[eval] '{args.checkpoint}' introuvable en local → traité comme un id HF Hub.")

    log_dir = REPO_ROOT / "runs" / args.run_name / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with DATASET_PATH.open() as f:
        items = json.load(f)
    if args.max_items > 0:
        items = items[:args.max_items]

    cached, todo = [], []
    for it in items:
        existing = None if args.force_redo else load_existing_log(it["item_id"], log_dir)
        if existing is not None:
            cached.append(existing)
        else:
            todo.append(it)

    print(f"[eval] {len(items)} items — {len(cached)} cached, {len(todo)} to run.")

    new_results: list[EpisodeResult] = []
    if todo:
        print(f"[eval] Loading model from {model_ref} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("[eval] Model loaded.")

        client = TextCraftEnvClient(
            env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60
        )
        print(f"[eval] Connected to TextCraft server (env_id={client.env_id}).")

        overall_t0 = time.time()
        for i, item in enumerate(todo):
            item_id = item["item_id"]
            try:
                r = run_episode(model, tokenizer, client, item_id, system_prompt=args.system_prompt,
                                enable_thinking=not args.no_thinking)
            except Exception as e:
                print(f"[eval][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
                continue
            write_episode_log(r, log_dir)
            status = "DONE" if r.done else "TIMEOUT"
            print(
                f"[eval][{i+1}/{len(todo)}] {item_id} reward={r.reward:.0f} "
                f"rounds={r.rounds} dur={r.duration_s:.1f}s [{status}]"
            )
            new_results.append(r)
        print(f"\n[eval] Wall time: {time.time() - overall_t0:.1f}s")

    all_results = cached + new_results
    if not all_results:
        print("[eval] No results.")
        return

    n_pass = sum(1 for r in all_results if r.reward > 0)
    print("=" * 50)
    print(f"RESULTS — {len(all_results)} items ({len(cached)} cached + {len(new_results)} fresh)")
    print(f"Pass@1  = {n_pass}/{len(all_results)}  ({100*n_pass/len(all_results):.1f}%)")
    print(f"Mean rounds    = {sum(r.rounds for r in all_results)/len(all_results):.1f}")
    print(f"Mean duration  = {sum(r.duration_s for r in all_results)/len(all_results):.1f}s")
    print(f"Logs: {log_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
