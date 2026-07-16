"""Tronc commun des évaluations TextCraft.

Contient l'unique implémentation de la boucle d'épisode multi-tour d'ÉVALUATION
(`run_episode`, paramétrée par une fonction de génération — serveur vLLM, HF,
API externe…) et toute la plomberie partagée : format des logs JSON, cache de
reprise, dataset, depths, estimateur pass@k.

Avant le refacto 2026-07-16, cette boucle et cette plomberie étaient copiées
dans 6 scripts (eval_vllm, eval_fullft, eval_baseline, eval_lora, eval_gemini,
eval_openai_compat) — voir archive/eval/README.md.

NB training : la boucle d'ENTRAÎNEMENT (token-level, env_mask) est distincte et
vit dans src/train/rollout.py — ne pas confondre les deux.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Callable

from agentenv.envs import TextCraftEnvClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
DEPTH_MAP_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"
RUNS_DIR = REPO_ROOT / "runs"
ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ROUNDS = 30  # protocole d'éval (le training utilise 20, cf. src/train/schedules.py)
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Signature d'un backend de génération : messages (format chat OpenAI) → texte.
GenerateFn = Callable[[list[dict]], str]


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


def load_items(max_items: int = 0) -> list[dict]:
    """Items du test set (100), tronqués à max_items si > 0."""
    with DATASET_PATH.open() as f:
        items = json.load(f)
    return items[:max_items] if max_items > 0 else items


def load_depth_map(path: Path = DEPTH_MAP_PATH) -> dict[str, int]:
    """Charge le mapping {item_id: depth} du test set TextCraft."""
    with path.open() as f:
        return {item_id: int(depth) for item_id, depth in json.load(f).items()}


def build_initial_messages(client: TextCraftEnvClient,
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                           fewshot_block: str | None = None) -> list[dict]:
    """system? + règles du jeu (+ exemples few-shot) + ack précodé + observation initiale.

    Si system_prompt est vide (ex. Gemma-3 qui rejette le rôle system), on
    démarre directement sur le message user. fewshot_block (exp18) : bloc
    d'exemples résolus injecté À LA FIN du message de règles — la structure
    des tours reste identique au zero-shot (seul le 1er message user grossit)."""
    rules_msg = client.conversation_start[0]["value"]
    ack_msg = client.conversation_start[1]["value"]
    initial_obs = client.observe()
    if fewshot_block:
        rules_msg = rules_msg + "\n\n" + fewshot_block
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += [
        {"role": "user", "content": rules_msg},
        {"role": "assistant", "content": ack_msg},
        {"role": "user", "content": initial_obs},
    ]
    return msgs


def run_episode(generate_fn: GenerateFn, client: TextCraftEnvClient, item_id: str,
                system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                max_rounds: int = MAX_ROUNDS,
                fewshot_block: str | None = None) -> EpisodeResult:
    """LA boucle d'épisode d'éval : reset → (générer → step env → observer)*.

    generate_fn est le seul point de variation entre backends (serveur vLLM avec
    KV cache, HF generate, API externe) — voir llm_chat.ChatGenerator."""
    item_idx = item_id_to_idx(item_id)
    client.reset(item_idx)
    messages = build_initial_messages(client, system_prompt=system_prompt,
                                      fewshot_block=fewshot_block)
    t0 = time.time()
    reward = 0.0
    done = False
    rounds = 0

    for round_idx in range(max_rounds):
        rounds = round_idx + 1
        assistant_text = generate_fn(messages)
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
    """Cache de reprise : recharge un épisode déjà joué (sauf --force-redo)."""
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


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimateur non biaisé pass@k (Chen et al. 2021, HumanEval) : probabilité qu'un
    sous-ensemble aléatoire de k tirages parmi n contienne >=1 réussite, sachant c
    réussites. Plus stable que de prendre "les k premiers"."""
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
