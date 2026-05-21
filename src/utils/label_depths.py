"""
Génère data/train/textcraft_train_with_depth.json : un dict {item_id: depth}
pour les 374 items du training set TextCraft.

Utilise la même logique que TextCraftEnv.reset() : items triés par profondeur
de recette minimale via CraftingTree.item_recipes_min_depth(1).

Pré-requis :
    conda activate agentenv-textcraft
    python src/utils/label_depths.py

Sortie :
    data/train/textcraft_train_with_depth.json   (dict {item_id: depth})
    (affiche aussi la distribution par depth)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/v.lagresle/rl-gym-workout")
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"
OUT_PATH   = REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"


def find_agentenv_textcraft_dir() -> Path:
    spec = importlib.util.find_spec("agentenv_textcraft")
    if spec is None or spec.origin is None:
        raise ImportError(
            "agentenv_textcraft not found. "
            "Run this script with: conda activate agentenv-textcraft"
        )
    return Path(spec.origin).parent


def main() -> None:
    pkg_dir = find_agentenv_textcraft_dir()
    sys.path.insert(0, str(pkg_dir.parent))

    from agentenv_textcraft.crafting_tree import CraftingTree

    # CraftingTree expects a path that contains recipes/ subdir
    tree = CraftingTree(minecraft_dir=str(pkg_dir) + "/")

    # Same sort as TextCraftEnv.reset(): primary key = depth
    sorted_items = sorted(
        list(tree.item_recipes_min_depth(1)),
        key=lambda x: x[1],
    )

    # Build idx → depth map (idx = data_idx used by env.reset)
    idx_to_depth: dict[int, int] = {
        idx: int(depth) for idx, (_, depth) in enumerate(sorted_items)
    }

    with TRAIN_PATH.open() as f:
        train_items = json.load(f)

    depth_map: dict[str, int] = {}
    for item in train_items:
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

    # Distribution
    from collections import Counter
    dist = Counter(depth_map.values())
    print("\nDistribution par depth :")
    for d in sorted(dist):
        print(f"  depth {d} : {dist[d]} items")


if __name__ == "__main__":
    main()