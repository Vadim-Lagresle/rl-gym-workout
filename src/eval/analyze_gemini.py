#!/usr/bin/env python3
"""
Analyse les logs d'évaluation Gemini et produit un rapport par depth.

Usage :
    python src/eval/analyze_gemini.py
    python src/eval/analyze_gemini.py --log-dir runs/exp_gemini_gemini_2_5_flash/eval_logs
    python src/eval/analyze_gemini.py --compare runs/exp1_baseline/eval_logs
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"


def get_item_depths() -> dict[str, int]:
    """Retourne {item_id: depth} pour tous les items du test set."""
    import subprocess, os

    script = """
import sys, os
sys.path.insert(0, "external/AgentGym/agentenv-textcraft")
os.chdir("external/AgentGym/agentenv-textcraft")
from agentenv_textcraft.crafting_tree import CraftingTree
import json

ct = CraftingTree(minecraft_dir="agentenv_textcraft")
item_depth_list = list(ct.item_recipes_min_depth(1))
sorted_list = sorted(item_depth_list, key=lambda x: x[1])

with open("/home/v.lagresle/rl-gym-workout/data/eval/textcraft_test.json") as f:
    items = json.load(f)

result = {}
for item in items:
    idx = int(item["item_id"].split("_")[1])
    name, depth = sorted_list[idx % len(sorted_list)]
    result[item["item_id"]] = {"depth": depth, "name": name}
print(json.dumps(result))
"""
    out = subprocess.check_output(
        ["conda", "run", "-n", "agentenv-textcraft", "python", "-c", script],
        cwd=str(REPO_ROOT),
    )
    raw = json.loads(out)
    return {k: v["depth"] for k, v in raw.items()}, {k: v["name"] for k, v in raw.items()}


def load_logs(log_dir: Path) -> list[dict]:
    results = []
    for fp in sorted(log_dir.glob("textcraft_*.json")):
        with fp.open() as f:
            results.append(json.load(f))
    return results


def print_table(label: str, results: list[dict], depth_map: dict[str, int], name_map: dict[str, str]) -> None:
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        d = depth_map.get(r["item_id"], -1)
        by_depth[d].append(r)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    header = f"{'Depth':<8} {'Items':>6} {'Solved':>8} {'Pass@1':>8} {'Avg rounds':>12} {'Avg dur(s)':>12}"
    print(header)
    print("-" * 60)

    total_solved = 0
    for depth in sorted(by_depth.keys()):
        items = by_depth[depth]
        solved = sum(1 for r in items if float(r.get("reward", 0)) > 0)
        total_solved += solved
        avg_rounds = sum(r.get("rounds", 0) for r in items) / len(items)
        avg_dur = sum(r.get("duration_s", 0) for r in items) / len(items)
        label_d = f"depth {depth}" if depth > 0 else "unknown"
        print(f"{label_d:<8} {len(items):>6} {solved:>8} {solved/len(items):>8.1%} {avg_rounds:>12.1f} {avg_dur:>12.1f}")

    print("-" * 60)
    n = len(results)
    avg_rounds_all = sum(r.get("rounds", 0) for r in results) / n if n else 0
    avg_dur_all = sum(r.get("duration_s", 0) for r in results) / n if n else 0
    print(f"{'TOTAL':<8} {n:>6} {total_solved:>8} {total_solved/n:>8.1%} {avg_rounds_all:>12.1f} {avg_dur_all:>12.1f}")
    print(f"{'='*60}\n")

    # Failures at depth ≥ 3
    hard_failures = [
        r for r in results
        if depth_map.get(r["item_id"], 0) >= 3 and float(r.get("reward", 0)) == 0
    ]
    if hard_failures:
        print(f"  Echecs depth ≥ 3 :")
        for r in hard_failures:
            d = depth_map.get(r["item_id"], "?")
            name = name_map.get(r["item_id"], "?")
            print(f"    {r['item_id']:20s}  depth={d}  rounds={r.get('rounds','?'):>2}  {name}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default=str(REPO_ROOT / "runs" / "exp_gemini_gemini_2_5_flash" / "eval_logs"),
    )
    parser.add_argument("--compare", default="", help="Second log dir to compare (e.g. baseline)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading depth info from TextCraft environment...", end=" ", flush=True)
    depth_map, name_map = get_item_depths()
    print("OK")

    results = load_logs(log_dir)
    if not results:
        print(f"No logs found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print_table(f"Gemini 2.5 Flash — {log_dir.parent.name}  ({len(results)} items)", results, depth_map, name_map)

    if args.compare:
        compare_dir = Path(args.compare)
        compare_results = load_logs(compare_dir)
        if compare_results:
            print_table(f"Comparaison : {compare_dir.parent.name}  ({len(compare_results)} items)", compare_results, depth_map, name_map)

            # Side-by-side delta
            print("  Delta Gemini vs comparaison (par depth) :")
            by_depth_main = defaultdict(list)
            by_depth_cmp = defaultdict(list)
            for r in results:
                by_depth_main[depth_map.get(r["item_id"], -1)].append(r)
            for r in compare_results:
                by_depth_cmp[depth_map.get(r["item_id"], -1)].append(r)

            for depth in sorted(set(by_depth_main) | set(by_depth_cmp)):
                m = by_depth_main.get(depth, [])
                c = by_depth_cmp.get(depth, [])
                if not m or not c:
                    continue
                p_m = sum(1 for r in m if float(r.get("reward", 0)) > 0) / len(m)
                p_c = sum(1 for r in c if float(r.get("reward", 0)) > 0) / len(c)
                delta = p_m - p_c
                sign = "+" if delta >= 0 else ""
                print(f"    depth {depth}: Gemini={p_m:.1%}  baseline={p_c:.1%}  delta={sign}{delta:.1%}")
            print()


if __name__ == "__main__":
    main()