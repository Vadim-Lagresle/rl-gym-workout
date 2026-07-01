"""
Phase 2 exp16 — replay des plans single-turn dans TextCraft (pass@1 oracle).

Pour chaque plan (phase 1, lecture seule dans plans/) :
  1. LLM extrait une séquence d'actions structurées (JSON)
  2. Conversion en commandes TextCraft (get / craft / inventory)
  3. Replay déterministe dans l'env (sans LLM entre les steps)

Les logs phase 1 (plans/) ne sont jamais modifiés.
Sortie : runs/<run-name>/replay_logs/

Pré-requis :
  - Serveur TextCraft sur 127.0.0.1:36005
  - Serveur vLLM sur 127.0.0.1:8001

Usage :
    bash src/utils/start_vllm_server.sh models/Qwen2.5-3B-Instruct

    python src/eval/replay_single_turn_plans.py \\
        --model models/Qwen2.5-3B-Instruct \\
        --run-name exp16_single_turn_reasoning

    python src/eval/replay_single_turn_plans.py \\
        --model models/Qwen2.5-3B-Instruct --max-items 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentenv.envs import TextCraftEnvClient

from llm_chat import ChatGenerator, VLLM_SERVER_URL


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANS_DIR_DEFAULT = REPO_ROOT / "runs" / "exp16_single_turn_reasoning" / "plans"
DEPTH_MAP_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"
MAX_ACTIONS = 50

EXTRACT_SYSTEM = (
    "You convert crafting plans into executable TextCraft action lists. "
    "Output valid JSON only, no markdown."
)

EXTRACT_USER_TEMPLATE = """Convert this crafting plan into an ordered list of TextCraft actions.

Allowed crafting recipes:
{recipes}

Goal: {goal}

Plan:
{plan}

Output JSON with this exact schema:
{{
  "actions": [
    {{"type": "get", "count": 9, "item": "lapis lazuli"}},
    {{"type": "craft", "count": 1, "item": "lapis block", "using": [{{"count": 9, "item": "lapis lazuli"}}]}}
  ]
}}

Rules:
- type is one of: get, craft, inventory
- get: fetch base (non-craftable) items from the environment
- craft: item = target; using = list of ingredients with counts
- use ONLY items and recipes from the allowed list above
- item names must match the recipe wording (e.g. "lapis lazuli", not ids)
- include every step needed to reach the goal, in order
- output JSON only"""


@dataclass
class ReplayResult:
    item_id: str
    item_idx: int
    depth: int | None
    reward: float
    done: bool
    n_actions: int
    n_steps_executed: int
    extracted_actions: list[dict]
    textcraft_commands: list[str]
    step_log: list[dict] = field(default_factory=list)
    extraction_raw: str = ""
    duration_s: float = 0.0
    error: str | None = None


def load_depth_map(path: Path = DEPTH_MAP_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {item_id: int(depth) for item_id, depth in json.load(f).items()}


def extract_actions(
    llm: ChatGenerator,
    recipes: str,
    goal: str,
    plan: str,
) -> tuple[list[dict], str]:
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": EXTRACT_USER_TEMPLATE.format(
                recipes=recipes, goal=goal, plan=plan
            ),
        },
    ]
    raw = llm.generate(messages, max_tokens=1024, temperature=0.0)
    data = parse_json_from_llm(raw)
    actions = data.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"invalid actions list in JSON: {data!r}")
    return actions, raw


def parse_json_from_llm(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def action_to_textcraft(cmd: dict) -> str:
    typ = cmd.get("type", "").lower()
    if typ == "inventory":
        return "inventory"
    if typ == "get":
        count = int(cmd["count"])
        item = str(cmd["item"]).strip()
        return f"get {count} {item}"
    if typ == "craft":
        count = int(cmd.get("count", 1))
        item = str(cmd["item"]).strip()
        using = cmd.get("using") or cmd.get("ingredients") or []
        parts = []
        for ing in using:
            parts.append(f"{int(ing['count'])} {str(ing['item']).strip()}")
        if not parts:
            raise ValueError(f"craft action missing ingredients: {cmd}")
        return f"craft {count} {item} using {', '.join(parts)}"
    raise ValueError(f"unknown action type: {typ!r}")


def replay_commands(
    client: TextCraftEnvClient,
    item_idx: int,
    commands: list[str],
) -> tuple[float, bool, list[dict]]:
    client.reset(item_idx)
    step_log: list[dict] = []
    reward = 0.0
    done = False

    for i, cmd in enumerate(commands[:MAX_ACTIONS]):
        # TextCraftEnvClient extrait le texte après "Action:"
        out = client.step(f"Action: {cmd}")
        reward = float(out.reward)
        done = bool(out.done)
        step_log.append(
            {
                "step": i + 1,
                "command": cmd,
                "observation": out.state,
                "reward": reward,
                "done": done,
            }
        )
        if done:
            break

    return reward, done, step_log


def load_plan(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_existing_replay(item_id: str, log_dir: Path) -> ReplayResult | None:
    fp = log_dir / f"{item_id}.json"
    if not fp.exists():
        return None
    try:
        with fp.open() as f:
            d = json.load(f)
        required = {"item_id", "item_idx", "reward", "done"}
        if not required.issubset(d):
            return None
        return ReplayResult(
            item_id=d["item_id"],
            item_idx=int(d["item_idx"]),
            depth=d.get("depth"),
            reward=float(d["reward"]),
            done=bool(d["done"]),
            n_actions=int(d.get("n_actions", 0)),
            n_steps_executed=int(d.get("n_steps_executed", 0)),
            extracted_actions=d.get("extracted_actions", []),
            textcraft_commands=d.get("textcraft_commands", []),
            step_log=d.get("step_log", []),
            extraction_raw=d.get("extraction_raw", ""),
            duration_s=float(d.get("duration_s", 0)),
            error=d.get("error"),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def write_replay_log(result: ReplayResult, log_dir: Path, plan_path: Path) -> None:
    fp = log_dir / f"{result.item_id}.json"
    payload = {
        "item_id": result.item_id,
        "item_idx": result.item_idx,
        "depth": result.depth,
        "reward": result.reward,
        "done": result.done,
        "n_actions": result.n_actions,
        "n_steps_executed": result.n_steps_executed,
        "extracted_actions": result.extracted_actions,
        "textcraft_commands": result.textcraft_commands,
        "step_log": result.step_log,
        "extraction_raw": result.extraction_raw,
        "duration_s": round(result.duration_s, 2),
        "error": result.error,
        "plan_path": str(plan_path),
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def replay_one_plan(
    llm: ChatGenerator,
    client: TextCraftEnvClient,
    plan_path: Path,
    depth: int | None,
) -> ReplayResult:
    plan_data = load_plan(plan_path)
    item_id = plan_data["item_id"]
    item_idx = int(plan_data["item_idx"])
    recipes = plan_data["recipes"]
    goal = plan_data["goal"]
    plan = plan_data["plan"]
    t0 = time.time()

    try:
        extracted, raw = extract_actions(llm, recipes, goal, plan)
        commands = [action_to_textcraft(a) for a in extracted]
        reward, done, step_log = replay_commands(client, item_idx, commands)
        return ReplayResult(
            item_id=item_id,
            item_idx=item_idx,
            depth=depth,
            reward=reward,
            done=done,
            n_actions=len(commands),
            n_steps_executed=len(step_log),
            extracted_actions=extracted,
            textcraft_commands=commands,
            step_log=step_log,
            extraction_raw=raw,
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return ReplayResult(
            item_id=item_id,
            item_idx=item_idx,
            depth=depth,
            reward=0.0,
            done=False,
            n_actions=0,
            n_steps_executed=0,
            extracted_actions=[],
            textcraft_commands=[],
            extraction_raw="",
            duration_s=time.time() - t0,
            error=str(e),
        )


def report_results(results: list[ReplayResult], depth_map: dict[str, int]) -> None:
    n = len(results)
    if n == 0:
        print("[replay] Aucun résultat.")
        return

    solved = [r for r in results if r.reward >= 1.0]
    print("\n" + "=" * 52)
    print(f"PASS@1 ORACLE — {len(solved)}/{n} ({100 * len(solved) / n:.1f}%)")
    print("=" * 52)

    by_depth: dict[int, list[ReplayResult]] = {}
    for r in results:
        d = r.depth if r.depth is not None else depth_map.get(r.item_id, -1)
        by_depth.setdefault(d, []).append(r)

    print(f"{'depth':>5} | {'pass':>8} | {'total':>5}")
    print("-" * 28)
    for d in sorted(by_depth):
        rs = by_depth[d]
        p = sum(1 for r in rs if r.reward >= 1.0)
        print(f"{d:>5} | {p:>3}/{len(rs):<4} | {100 * p / len(rs):>5.1f}%")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nExtraction/replay errors: {len(errors)}/{n}")
    print("=" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exp16 plans in TextCraft (phase 2).")
    parser.add_argument("--model", required=True, help="Chemin local ou id HF Hub du modèle")
    parser.add_argument("--run-name", default="exp16_single_turn_reasoning")
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=None,
        help="Dossier plans phase 1 (défaut: runs/<run-name>/plans/)",
    )
    parser.add_argument("--max-items", type=int, default=0, help="0 = tous les plans")
    parser.add_argument("--force-redo", action="store_true")
    parser.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    parser.add_argument("--vllm-url", type=str, default=VLLM_SERVER_URL)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Qwen3/3.5 : enable_thinking=False")
    args = parser.parse_args()

    plans_dir = args.plans_dir
    if plans_dir is None:
        plans_dir = REPO_ROOT / "runs" / args.run_name / "plans"
    elif not plans_dir.is_absolute():
        plans_dir = REPO_ROOT / plans_dir
    if not plans_dir.is_dir():
        raise SystemExit(f"Plans introuvables : {plans_dir}")

    llm = ChatGenerator(
        args.model,
        backend=args.backend,
        vllm_url=args.vllm_url,
        no_thinking=args.no_thinking,
    )

    log_dir = REPO_ROOT / "runs" / args.run_name / "replay_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan_files = sorted(plans_dir.glob("textcraft_*.json"), key=lambda p: p.name)
    if args.max_items > 0:
        plan_files = plan_files[: args.max_items]

    depth_map = load_depth_map()

    cached, todo = [], []
    for pf in plan_files:
        plan_data = load_plan(pf)
        item_id = plan_data["item_id"]
        existing = None if args.force_redo else load_existing_replay(item_id, log_dir)
        if existing is not None:
            cached.append(existing)
        else:
            todo.append(pf)

    print(
        f"[replay] {len(plan_files)} plans — {len(cached)} cached, {len(todo)} à rejouer."
    )
    print(f"[replay] plans (read-only): {plans_dir}")
    print(f"[replay] logs: {log_dir}")

    client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)

    new_results: list[ReplayResult] = []
    t0 = time.time()
    for i, pf in enumerate(todo):
        plan_data = load_plan(pf)
        item_id = plan_data["item_id"]
        depth = plan_data.get("depth") or depth_map.get(item_id)
        result = replay_one_plan(llm, client, pf, depth=depth)
        write_replay_log(result, log_dir, pf)
        status = "OK" if result.reward >= 1.0 else ("ERR" if result.error else "FAIL")
        depth_str = f"d{depth}" if depth is not None else "d?"
        print(
            f"[replay][{i+1}/{len(todo)}] {item_id} {depth_str} "
            f"reward={result.reward:.0f} steps={result.n_steps_executed} "
            f"[{status}]"
            + (f" err={result.error[:60]}" if result.error else "")
        )
        new_results.append(result)

    all_results = cached + new_results
    print(f"\n[replay] Wall time: {time.time() - t0:.1f}s")
    report_results(all_results, depth_map)
    print(f"Logs: {log_dir}/")


if __name__ == "__main__":
    main()
