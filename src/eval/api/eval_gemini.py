"""
Évalue Gemini (Flash ou Pro) sur TextCraft via l'API Google Gemini.

Même architecture que eval_baseline.py : épisodes multi-tours avec TextCraftEnvClient,
logs JSON identiques, mode resume. Remplace vLLM par des appels à l'API Gemini.

Pré-requis :
  1. Serveur TextCraft sur 127.0.0.1:36005.
  2. Variable d'env GEMINI_API_KEY définie.
  3. pip install google-genai

Usage :
    GEMINI_API_KEY=xxx python src/eval/eval_gemini.py
    GEMINI_API_KEY=xxx MODEL_NAME=gemini-3.5-flash MAX_ITEMS=10 python src/eval/eval_gemini.py
    GEMINI_API_KEY=xxx FORCE_REDO=1 python src/eval/eval_gemini.py

Variables d'environnement :
    GEMINI_API_KEY  — clé API Google (obligatoire)
    MODEL_NAME      — modèle Gemini à évaluer (défaut : gemini-3.5-flash)
    MAX_ITEMS       — tronque le dataset pour smoke test (0 = tous les 100 items)
    FORCE_REDO      — ignore les logs existants si 1
    EVAL_LOG_DIR    — dossier de sortie des logs (défaut : runs/exp_gemini_baseline/eval_logs)
    RPM_LIMIT       — requests/minute max pour respecter les quotas API (défaut : 15)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentenv.envs import TextCraftEnvClient

try:
    import httpx
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError("Installer google-genai et httpx : pip install google-genai httpx")


REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.5-flash")
LOG_DIR = Path(os.environ.get("EVAL_LOG_DIR", str(
    REPO_ROOT / "runs" / f"exp_gemini_{MODEL_NAME.replace('.', '_').replace('-', '_')}" / "eval_logs"
)))
LOG_DIR.mkdir(parents=True, exist_ok=True)

ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ROUNDS = 30
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))
FORCE_REDO = bool(int(os.environ.get("FORCE_REDO", "0")))
RPM_LIMIT = int(os.environ.get("RPM_LIMIT", "15"))

# Délai minimum entre appels API pour rester sous RPM_LIMIT
MIN_CALL_INTERVAL = 60.0 / RPM_LIMIT  # secondes

SYSTEM_PROMPT = (
    "You are a helpful assistant playing a crafting game. "
    "At each turn, read the observation carefully and respond with exactly one action "
    "using the format: Action: <action>. "
    "Think step by step before acting."
)


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


def call_gemini_with_retry(
    client: "genai.Client",
    contents: list,
    max_retries: int = 8,
) -> str:
    """Appelle l'API Gemini avec retry exponentiel sur les erreurs de rate limit."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.0,
                    max_output_tokens=512,
                ),
            )
            return response.text or ""
        except Exception as e:
            err = str(e).lower()
            is_retryable = (
                "429" in err or "503" in err or "quota" in err
                or "rate" in err or "unavailable" in err or "timeout" in err
            )
            if is_retryable and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                print(f"  [retry] attente {wait:.1f}s (tentative {attempt+1})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def run_episode(
    client: "genai.Client",
    env_client: TextCraftEnvClient,
    item_id: str,
    last_call_time: list[float],
    max_rounds: int = MAX_ROUNDS,
) -> EpisodeResult:
    item_idx = item_id_to_idx(item_id)
    env_client.reset(item_idx)

    rules_msg = env_client.conversation_start[0]["value"]
    ack_msg = env_client.conversation_start[1]["value"]
    initial_obs = env_client.observe()

    # Historique de conversation au format Gemini
    # Note : le system prompt est passé séparément dans GenerateContentConfig
    contents: list[genai_types.Content] = [
        genai_types.Content(role="user",  parts=[genai_types.Part(text=rules_msg)]),
        genai_types.Content(role="model", parts=[genai_types.Part(text=ack_msg)]),
        genai_types.Content(role="user",  parts=[genai_types.Part(text=initial_obs)]),
    ]

    # Transcript humain-lisible (même format que eval_baseline.py)
    transcript: list[dict] = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": rules_msg},
        {"role": "assistant", "content": ack_msg},
        {"role": "user",      "content": initial_obs},
    ]

    t0 = time.time()
    reward = 0.0
    done = False
    rounds = 0

    for round_idx in range(max_rounds):
        rounds = round_idx + 1

        # Respecter le rate limit
        elapsed = time.time() - last_call_time[0]
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)

        assistant_text = call_gemini_with_retry(client, contents)
        last_call_time[0] = time.time()

        contents.append(genai_types.Content(
            role="model", parts=[genai_types.Part(text=assistant_text)]
        ))
        transcript.append({"role": "assistant", "content": assistant_text})

        step_out = env_client.step(assistant_text)
        reward = float(step_out.reward)
        done = bool(step_out.done)

        contents.append(genai_types.Content(
            role="user", parts=[genai_types.Part(text=step_out.state)]
        ))
        transcript.append({"role": "user", "content": step_out.state})

        if done:
            break

    return EpisodeResult(
        item_id=item_id,
        item_idx=item_idx,
        reward=reward,
        done=done,
        rounds=rounds,
        duration_s=time.time() - t0,
        transcript=transcript,
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
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY non définie. "
            "Exporter la clé : export GEMINI_API_KEY=your_key"
        )

    gemini_client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            httpx_client=httpx.Client(timeout=90.0),
        ),
    )
    print(f"[eval] Modèle : {MODEL_NAME}")
    print(f"[eval] Rate limit : {RPM_LIMIT} RPM  ({MIN_CALL_INTERVAL:.1f}s min entre appels)")
    print(f"[eval] Logs → {LOG_DIR}/")

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
        f"[eval] {len(items)} items total. "
        f"{len(cached)} cached (skipped), {len(todo)} à lancer."
    )

    new_results: list[EpisodeResult] = []
    if todo:
        env_client = TextCraftEnvClient(
            env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60
        )
        print(f"[eval] Connecté au serveur TextCraft (env_id={env_client.env_id}).")

        last_call_time = [0.0]  # liste pour être mutable dans run_episode
        overall_t0 = time.time()

        for i, item in enumerate(todo):
            item_id = item["item_id"]
            try:
                r = run_episode(gemini_client, env_client, item_id, last_call_time)
            except Exception as e:
                print(f"[eval][{i+1}/{len(todo)}] {item_id} CRASH: {e}")
                continue
            write_episode_log(r)
            status = "DONE" if r.done else "TIMEOUT"
            print(
                f"[eval][{i+1}/{len(todo)}] {item_id}  reward={r.reward}  "
                f"rounds={r.rounds}  dur={r.duration_s:.1f}s  [{status}]"
            )
            new_results.append(r)

        print(f"\n[eval] Temps total (nouveaux rollouts) : {time.time() - overall_t0:.1f}s")

    all_results = cached + new_results
    if not all_results:
        print("[eval] Aucun résultat.")
        return

    successes = sum(1 for r in all_results if r.reward > 0)
    print("=" * 50)
    print(f"RÉSULTATS  {MODEL_NAME}  —  {len(all_results)} items")
    print(f"  ({len(cached)} cached + {len(new_results)} frais)")
    print("=" * 50)
    print(f"Pass@1      = {successes / len(all_results):.4f}  ({successes}/{len(all_results)})")
    print(f"Avg reward  = {sum(r.reward for r in all_results) / len(all_results):.4f}")
    print(f"Mean rounds = {sum(r.rounds for r in all_results) / len(all_results):.1f}")
    print(f"Mean dur/ep = {sum(r.duration_s for r in all_results) / len(all_results):.1f}s")
    print(f"Logs : {LOG_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()