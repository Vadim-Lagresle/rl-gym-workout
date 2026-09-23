"""Labels every TextCraft task with its depth.

In plain words: the depth of a recipe is the number of crafting levels between raw
materials and the goal (1 to 4). This script computes it with the environment's own
crafting tree and writes the {task: depth} files used for stratified results and depth
curricula. Needs the TextCraft environment package (its own virtualenv).

Notes (FR) — Génère les mappings {item_id: depth} des datasets TextCraft :
    data/train/textcraft_train_with_depth.json   (374 items)
    data/eval/textcraft_test_with_depth.json     (100 items)

Utilise la même logique que TextCraftEnv.reset() : items triés par profondeur
de recette minimale via CraftingTree.item_recipes_min_depth(1).

Pré-requis (le package agentenv_textcraft n'existe que dans cet env, et son
import instancie un CraftingTree avec le chemin RELATIF 'agentenv_textcraft/recipes/'
→ lancer OBLIGATOIREMENT depuis le dossier du package, comme le serveur) :
    source ~/envs/agentenv-textcraft/bin/activate
    cd external/AgentGym/agentenv-textcraft
    python /chemin/absolu/src/utils/label_depths.py                 # les deux splits
    python /chemin/absolu/src/utils/label_depths.py --split test    # test seul

(Fusion de l'ancien label_depths_test.py — archive/utils/ — 2026-07-16.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))

SPLITS = {
    "train": (REPO_ROOT / "data" / "train" / "textcraft_train.json",
              REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"),
    "test":  (REPO_ROOT / "data" / "eval" / "textcraft_test.json",
              REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"),
}


def find_agentenv_textcraft_dir() -> Path:
    spec = importlib.util.find_spec("agentenv_textcraft")
    if spec is None or spec.origin is None:
        raise ImportError(
            "agentenv_textcraft not found. "
            "Run: source ~/envs/agentenv-textcraft/bin/activate"
        )
    return Path(spec.origin).parent


def build_idx_to_depth() -> dict[int, int]:
    """idx → depth, avec le tri exact de TextCraftEnv.reset() (clé = depth)."""
    pkg_dir = find_agentenv_textcraft_dir()
    sys.path.insert(0, str(pkg_dir.parent))
    from agentenv_textcraft.crafting_tree import CraftingTree

    # CraftingTree attend un chemin contenant le sous-dossier recipes/
    tree = CraftingTree(minecraft_dir=str(pkg_dir) + "/")
    sorted_items = sorted(list(tree.item_recipes_min_depth(1)), key=lambda x: x[1])
    return {idx: int(depth) for idx, (_, depth) in enumerate(sorted_items)}


def label_split(split: str, idx_to_depth: dict[int, int]) -> None:
    src_path, out_path = SPLITS[split]
    with src_path.open() as f:
        items = json.load(f)

    depth_map: dict[str, int] = {}
    for item in items:
        item_id = item["item_id"]
        idx = int(item_id.rsplit("_", 1)[1])
        if idx not in idx_to_depth:
            print(f"  WARNING: {item_id} (idx={idx}) not found in crafting tree")
            depth_map[item_id] = -1
        else:
            depth_map[item_id] = idx_to_depth[idx]

    with out_path.open("w") as f:
        json.dump(depth_map, f, indent=2)
    print(f"[{split}] Saved {len(depth_map)} entries → {out_path}")

    dist = Counter(depth_map.values())
    print(f"[{split}] Distribution par depth :")
    for d in sorted(dist):
        print(f"  depth {d} : {dist[d]} items")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()

    idx_to_depth = build_idx_to_depth()
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        label_split(split, idx_to_depth)


if __name__ == "__main__":
    main()
