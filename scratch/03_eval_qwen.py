"""
Évaluation manuelle de Qwen2.5-3B-Instruct sur TextCraft, sans verl/Ray.

Pourquoi ce script existe : `examples/eval/textcraft_eval.local.sh` invoquait
`verl.agent_trainer.main_generation` qui passe par un fork custom de vLLM
(`verl.third_party.vllm`) avec un `load_format=dummy_dtensor`. Ce flow est conçu
pour évaluer un checkpoint qui sort d'un training PPO multi-GPU avec FSDP : il
init NCCL puis attend qu'on patche les vrais poids depuis un DTensor FSDP. Pour
notre cas (modèle HF brut, 1 A100), l'init NCCL crash en SIGSEGV silencieux.

Ce script reproduit donc la boucle de rollout multi-tour directement avec :
  - vLLM "standard" (l'install pip dans `agentgym-rl`)
  - `agentenv.envs.TextCraftEnvClient` (le wrapper HTTP, intact)
  - `tokenizer.apply_chat_template` pour formater les conversations Qwen

Pré-requis :
  1. Serveur TextCraft sur 127.0.0.1:36005 (cf. WORKLOG.md, démarrage hors sandbox).
  2. Conda env `agentgym-rl` activé.

Lancement :
    conda activate agentgym-rl
    python scratch/03_eval_qwen.py             # full eval (100 items, resume si déjà fait)
    MAX_ITEMS=7 python scratch/03_eval_qwen.py # smoke test
    FORCE_REDO=1 python scratch/03_eval_qwen.py # ignore les logs précédents

Mode resume (par défaut) : si un item a déjà un log valide dans
`scratch/eval_logs/<item_id>.json`, on le charge au lieu de refaire le
rollout. Pratique pour reprendre après une interruption (préemption Spot,
SIGTERM volontaire, etc.) ou pour ne réévaluer qu'un sous-ensemble. Mettre
`FORCE_REDO=1` pour tout recalculer.
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


REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
DATASET_PATH = REPO_ROOT / "AgentEval" / "eval" / "textcraft_test.json"
LOG_DIR = REPO_ROOT / "scratch" / "eval_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ROUNDS = 30
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))  # 0 = tout
FORCE_REDO = bool(int(os.environ.get("FORCE_REDO", "0")))

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


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
    """Compose [system, user(rules), assistant(ack), user(initial_obs)]."""
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
    """'textcraft_42' -> 42."""
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
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
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
        "transcript": result.transcript,
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def load_existing_log(item_id: str) -> EpisodeResult | None:
    """Charge un log précédent si présent et valide ; sinon None (=> à rerun)."""
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
        f"[eval] {len(items)} total items. "
        f"{len(cached)} cached (skipped), {len(todo)} to run."
    )

    new_results: list[EpisodeResult] = []
    if todo:
        print("[eval] Loading vLLM (this takes ~10-30s)...")
        llm = LLM(
            model=str(MODEL_PATH),
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            dtype="bfloat16",
            load_format="safetensors",
            tensor_parallel_size=1,
            enable_prefix_caching=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512)

        client = TextCraftEnvClient(
            env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60
        )
        print(f"[eval] Connected to env server (env_id={client.env_id}).")

        overall_t0 = time.time()
        for i, item in enumerate(todo):
            item_id = item["item_id"]
            try:
                r = run_episode(llm, tokenizer, client, sampling, item_id)
            except Exception as e:
                print(f"[eval][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
                continue
            write_episode_log(r)
            status = "DONE" if r.done else "TIMEOUT"
            print(
                f"[eval][{i+1}/{len(todo)}] {item_id} reward={r.reward} "
                f"rounds={r.rounds} dur={r.duration_s:.1f}s [{status}]"
            )
            new_results.append(r)
        print(f"\n[eval] Wall time on new rollouts: {time.time() - overall_t0:.1f}s")

    all_results = cached + new_results
    if not all_results:
        print("[eval] No results to summarize.")
        return
    avg_reward = sum(r.reward for r in all_results) / len(all_results)
    pass_rate = sum(1 for r in all_results if r.reward > 0) / len(all_results)
    print("=" * 50)
    print(f"GLOBAL STATS over {len(all_results)} items")
    print(f"  ({len(cached)} cached + {len(new_results)} fresh)")
    print("=" * 50)
    print(f"Avg@1   = {avg_reward:.4f}")
    print(f"Pass@1  = {pass_rate:.4f}  ({sum(1 for r in all_results if r.reward > 0)}/{len(all_results)})")
    print(f"Mean rounds = {sum(r.rounds for r in all_results) / len(all_results):.1f}")
    print(f"Mean duration = {sum(r.duration_s for r in all_results) / len(all_results):.1f}s")
    print(f"Logs: {LOG_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
