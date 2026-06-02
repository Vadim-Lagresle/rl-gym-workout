"""
Évalue un checkpoint TextCraft via un serveur vLLM (OpenAI-compatible API).

Avantage vs eval_fullft.py : vLLM maintient un KV cache entre les rounds d'un
épisode. Au round 20, seuls les ~270 nouveaux tokens sont traités (au lieu des
~8000 du contexte complet). Speedup estimé : ×7-10 (35 min → ~5 min pour 100 items).

Pré-requis :
    # Terminal 1 — lancer le serveur vLLM (charge le modèle une fois)
    bash src/utils/start_vllm_server.sh saves/trl_grpo/exp7_b200_fullft/checkpoint-564

    # Terminal 2 — eval
    source ~/envs/agentgym-rl/bin/activate
    python src/eval/eval_vllm.py \\
        --run-name exp7_vllm \\
        --model saves/trl_grpo/exp7_b200_fullft/checkpoint-564

    # Smoke test (10 items)
    python src/eval/eval_vllm.py --run-name smoke_vllm --model models/Qwen2.5-3B-Instruct --max-items 10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from agentenv.envs import TextCraftEnvClient


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"
VLLM_SERVER_URL = "http://127.0.0.1:8001"
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


def check_vllm_server() -> None:
    try:
        r = requests.get(f"{VLLM_SERVER_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"Serveur vLLM non disponible sur {VLLM_SERVER_URL}.\n"
            f"Lance-le avec : bash src/utils/start_vllm_server.sh <checkpoint>\n"
            f"Erreur : {e}"
        )


def generate_reply_vllm(model_name: str, messages: list[dict],
                         vllm_url: str = VLLM_SERVER_URL,
                         max_tokens: int = 512, temperature: float = 1.0) -> str:
    """Appel HTTP au serveur vLLM (OpenAI chat/completions)."""
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
    }
    r = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def build_initial_messages(client: TextCraftEnvClient,
                            system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    rules_msg = client.conversation_start[0]["value"]
    ack_msg = client.conversation_start[1]["value"]
    initial_obs = client.observe()
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += [
        {"role": "user", "content": rules_msg},
        {"role": "assistant", "content": ack_msg},
        {"role": "user", "content": initial_obs},
    ]
    return msgs


def run_episode(model_name: str, client: TextCraftEnvClient, item_id: str,
                system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                max_rounds: int = MAX_ROUNDS,
                vllm_url: str = VLLM_SERVER_URL) -> EpisodeResult:
    item_idx = item_id_to_idx(item_id)
    client.reset(item_idx)
    messages = build_initial_messages(client, system_prompt=system_prompt)
    t0 = time.time()
    reward = 0.0
    done = False
    rounds = 0

    for round_idx in range(max_rounds):
        rounds = round_idx + 1
        # vLLM reuses KV cache for the shared prefix between consecutive rounds.
        # Each round only computes new tokens (observation + generation prompt).
        assistant_text = generate_reply_vllm(model_name, messages, vllm_url=vllm_url)
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
    parser.add_argument("--model", required=True,
                        help="Chemin ou nom du modèle chargé dans le serveur vLLM")
    parser.add_argument("--run-name", required=True,
                        help="Nom du run (logs dans runs/<run-name>/eval_logs/)")
    parser.add_argument("--max-items", type=int, default=0, help="0 = tous les 100 items")
    parser.add_argument("--force-redo", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--vllm-url", type=str, default=VLLM_SERVER_URL,
                        help="URL du serveur vLLM (défaut: http://localhost:8001)")
    args = parser.parse_args()

    vllm_url = args.vllm_url
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model_name = str(model_path)

    try:
        requests.get(f"{vllm_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"Serveur vLLM non disponible sur {vllm_url}.\n"
                         f"Lance : bash src/utils/start_vllm_server.sh <checkpoint>\nErreur: {e}")
    print(f"[eval_vllm] Serveur vLLM OK sur {vllm_url}")

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

    print(f"[eval_vllm] {len(items)} items — {len(cached)} cached, {len(todo)} à évaluer.")

    client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60)

    new_results: list[EpisodeResult] = []
    overall_t0 = time.time()
    for i, item in enumerate(todo):
        item_id = item["item_id"]
        try:
            r = run_episode(model_name, client, item_id, system_prompt=args.system_prompt,
                            vllm_url=vllm_url)
        except Exception as e:
            print(f"[eval_vllm][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
            continue
        write_episode_log(r, log_dir)
        status = "DONE" if r.done else "TIMEOUT"
        print(f"[eval_vllm][{i+1}/{len(todo)}] {item_id} reward={r.reward:.0f} "
              f"rounds={r.rounds} dur={r.duration_s:.1f}s [{status}]")
        new_results.append(r)

    all_results = cached + new_results
    if not all_results:
        print("[eval_vllm] Aucun résultat.")
        return

    n_pass = sum(1 for r in all_results if r.reward > 0)
    print(f"\n[eval_vllm] Wall time: {time.time() - overall_t0:.1f}s")
    print("=" * 50)
    print(f"RESULTS — {len(all_results)} items ({len(cached)} cached + {len(new_results)} fresh)")
    print(f"Pass@1  = {n_pass}/{len(all_results)}  ({100*n_pass/len(all_results):.1f}%)")
    print(f"Mean rounds   = {sum(r.rounds for r in all_results)/len(all_results):.1f}")
    print(f"Mean duration = {sum(r.duration_s for r in all_results)/len(all_results):.1f}s")
    print(f"Logs: {log_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
