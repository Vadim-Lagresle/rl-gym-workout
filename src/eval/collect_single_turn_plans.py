"""
Phase 1 — collecte de plans single-turn pour TextCraft (exp16).

Pour chaque item du test set :
  1. reset TextCraft → observation (recettes + goal)
  2. prompt unique au LLM : planifier la séquence de craft complète
  3. sauvegarde de la réponse en texte libre (pas de JSON côté modèle)

La phase 2 (à coder ensuite) convertira ces plans en actions TextCraft
et mesurera pass@1.

Pré-requis :
  - Serveur TextCraft sur 127.0.0.1:36005
  - Serveur vLLM sur 127.0.0.1:8001

Usage :
    bash src/utils/start_vllm_server.sh models/Qwen2.5-3B-Instruct

    python src/eval/collect_single_turn_plans.py \\
        --model models/Qwen2.5-3B-Instruct \\
        --run-name exp16_single_turn_reasoning

    # Smoke test (5 items)
    python src/eval/collect_single_turn_plans.py \\
        --model models/Qwen2.5-3B-Instruct \\
        --run-name exp16_single_turn_reasoning \\
        --max-items 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from agentenv.envs import TextCraftEnvClient

from llm_chat import ChatGenerator, VLLM_SERVER_URL


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
DEPTH_MAP_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    "skilled at logical planning and step-by-step reasoning."
)

PLANNING_USER_TEMPLATE = """You are solving a Minecraft crafting puzzle.

Here are the crafting recipes you are allowed to use:
{recipes}

Your goal: {goal}

Before executing anything in the game, work out the complete strategy to reach the goal.

Explain your reasoning, then give the exact ordered sequence of steps needed — from obtaining raw materials through every intermediate craft until the final item is produced.

Write in plain English as a numbered list. You do NOT need to use any special command syntax (no JSON, no "Action:" format). Focus on the logic: what to gather, what quantities, and what to combine at each step."""


@dataclass
class PlanResult:
    item_id: str
    item_idx: int
    depth: int | None
    observation: str
    recipes: str
    goal: str
    plan: str
    duration_s: float
    messages: list[dict]


def item_id_to_idx(item_id: str) -> int:
    m = re.search(r"(\d+)$", item_id)
    if not m:
        raise ValueError(f"Cannot parse item_id: {item_id!r}")
    return int(m.group(1))


def parse_textcraft_observation(obs: str) -> tuple[str, str]:
    """Extrait recettes et goal depuis l'observation initiale TextCraft."""
    obs = obs.strip()
    if "Goal:" not in obs:
        return obs, ""

    before, goal_part = obs.rsplit("Goal:", 1)
    goal = goal_part.strip().rstrip(".")
    recipes = before.strip()
    if recipes.startswith("Crafting commands:"):
        recipes = recipes[len("Crafting commands:"):].strip()
    return recipes, goal


def load_depth_map(path: Path = DEPTH_MAP_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {item_id: int(depth) for item_id, depth in json.load(f).items()}


def build_planning_messages(
    recipes: str,
    goal: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    user_content = PLANNING_USER_TEMPLATE.format(recipes=recipes, goal=goal)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def collect_plan(
    llm: ChatGenerator,
    client: TextCraftEnvClient,
    item_id: str,
    depth: int | None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> PlanResult:
    item_idx = item_id_to_idx(item_id)
    client.reset(item_idx)
    observation = client.observe()
    recipes, goal = parse_textcraft_observation(observation)
    messages = build_planning_messages(recipes, goal, system_prompt=system_prompt)

    t0 = time.time()
    plan = llm.generate(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    duration_s = time.time() - t0

    messages.append({"role": "assistant", "content": plan})
    return PlanResult(
        item_id=item_id,
        item_idx=item_idx,
        depth=depth,
        observation=observation,
        recipes=recipes,
        goal=goal,
        plan=plan,
        duration_s=duration_s,
        messages=messages,
    )


def write_plan_log(result: PlanResult, log_dir: Path, model: str, run_name: str) -> None:
    fp = log_dir / f"{result.item_id}.json"
    payload = {
        "item_id": result.item_id,
        "item_idx": result.item_idx,
        "depth": result.depth,
        "observation": result.observation,
        "recipes": result.recipes,
        "goal": result.goal,
        "plan": result.plan,
        "duration_s": round(result.duration_s, 2),
        "model": model,
        "run_name": run_name,
        "messages": result.messages,
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def load_existing_plan(item_id: str, log_dir: Path) -> PlanResult | None:
    fp = log_dir / f"{item_id}.json"
    if not fp.exists():
        return None
    try:
        with fp.open() as f:
            d = json.load(f)
        required = {"item_id", "item_idx", "plan", "goal", "recipes", "observation"}
        if not required.issubset(d) or not d.get("plan", "").strip():
            return None
        return PlanResult(
            item_id=d["item_id"],
            item_idx=int(d["item_idx"]),
            depth=d.get("depth"),
            observation=d["observation"],
            recipes=d["recipes"],
            goal=d["goal"],
            plan=d["plan"],
            duration_s=float(d.get("duration_s", 0)),
            messages=d.get("messages", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collecte des plans single-turn TextCraft (phase 1 exp16)."
    )
    parser.add_argument("--model", required=True,
                        help="Chemin local ou id HF Hub du modèle")
    parser.add_argument("--run-name", default="exp16_single_turn_reasoning",
                        help="Nom du run (plans dans runs/<run-name>/plans/)")
    parser.add_argument("--max-items", type=int, default=0, help="0 = tous les 100 items")
    parser.add_argument("--force-redo", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    parser.add_argument("--vllm-url", type=str, default=VLLM_SERVER_URL)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Qwen3/3.5 : enable_thinking=False")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_ref = args.model
    llm = ChatGenerator(
        model_ref,
        backend=args.backend,
        vllm_url=args.vllm_url,
        no_thinking=args.no_thinking,
    )
    model_name = llm.model_ref

    log_dir = REPO_ROOT / "runs" / args.run_name / "plans"
    log_dir.mkdir(parents=True, exist_ok=True)

    with DATASET_PATH.open() as f:
        items = json.load(f)
    if args.max_items > 0:
        items = items[: args.max_items]

    depth_map = load_depth_map()

    cached, todo = [], []
    for it in items:
        existing = None if args.force_redo else load_existing_plan(it["item_id"], log_dir)
        if existing is not None:
            cached.append(existing)
        else:
            todo.append(it)

    print(f"[collect_plans] {len(items)} items — {len(cached)} cached, {len(todo)} à collecter.")

    client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(items), timeout=60)

    new_results: list[PlanResult] = []
    overall_t0 = time.time()
    for i, item in enumerate(todo):
        item_id = item["item_id"]
        depth = depth_map.get(item_id)
        try:
            r = collect_plan(
                llm,
                client,
                item_id,
                depth=depth,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"[collect_plans][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
            continue
        write_plan_log(r, log_dir, model=model_name, run_name=args.run_name)
        depth_str = f"d{r.depth}" if r.depth is not None else "d?"
        print(
            f"[collect_plans][{i+1}/{len(todo)}] {item_id} {depth_str} "
            f"goal={r.goal!r} plan_len={len(r.plan)} dur={r.duration_s:.1f}s"
        )
        new_results.append(r)

    all_results = cached + new_results
    if not all_results:
        print("[collect_plans] Aucun résultat.")
        return

    print(f"\n[collect_plans] Wall time: {time.time() - overall_t0:.1f}s")
    print("=" * 50)
    print(f"COLLECTED — {len(all_results)} plans ({len(cached)} cached + {len(new_results)} fresh)")
    print(f"Mean plan length = {sum(len(r.plan) for r in all_results) / len(all_results):.0f} chars")
    print(f"Mean duration    = {sum(r.duration_s for r in all_results) / len(all_results):.1f}s")
    print(f"Plans: {log_dir}/")
    print("=" * 50)
    print("Phase 2 (à coder) : convertir ces plans en actions TextCraft et mesurer pass@1.")


if __name__ == "__main__":
    main()
