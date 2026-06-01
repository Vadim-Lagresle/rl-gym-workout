"""
Génère data/eval/textcraft_test_with_depth.json : un dict {item_id: depth}
pour les 100 items du test set TextCraft.

Même logique que label_depths.py mais appliquée au test set.

Pré-requis :
    source ~/envs/agentenv-textcraft/bin/activate
    python src/utils/label_depths_test.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))
TEST_PATH  = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
OUT_PATH   = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"


def find_agentenv_textcraft_dir() -> Path:
    spec = importlib.util.find_spec("agentenv_textcraft")
    if spec is None or spec.origin is None:
        raise ImportError(
            "agentenv_textcraft not found. "
            "Run: source ~/envs/agentenv-textcraft/bin/activate"
        )
    return Path(spec.origin).parent


def main() -> None:
    pkg_dir = find_agentenv_textcraft_dir()
    sys.path.insert(0, str(pkg_dir.parent))

    from agentenv_textcraft.crafting_tree import CraftingTree

    tree = CraftingTree(minecraft_dir=str(pkg_dir) + "/")
    sorted_items = sorted(
        list(tree.item_recipes_min_depth(1)),
        key=lambda x: x[1],
    )
    idx_to_depth: dict[int, int] = {
        idx: int(depth) for idx, (_, depth) in enumerate(sorted_items)
    }

    with TEST_PATH.open() as f:
        test_items = json.load(f)

    depth_map: dict[str, int] = {}
    for item in test_items:
        item_id = item["item_id"]
        idx = int(item_id.rsplit("_", 1)[1])
        if idx not in idx_to_depth:
            print(f"  WARNING: {item_id} (idx={idx}) not found in crafting tree")
            depth_map[item_id] = -1
        else:
            depth_map[item_id] = idx_to_depth[idx]

    with OUT_PATH.open("w") as f:
        json.dump(depth_map, f, indent=2)
    print(f"Saved {len(depth_map)} entries → {OUT_PATH}")

    dist = Counter(depth_map.values())
    print("\nDistribution par depth :")
    for d in sorted(dist):
        print(f"  depth {d} : {dist[d]} items")


if __name__ == "__main__":
    main()
