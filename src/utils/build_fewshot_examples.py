"""Builds the solved examples used in few-shot prompts.

In plain words: takes the 70 TextCraft recipes that are in neither the training nor the
test set, derives their optimal solution from the crafting tree, checks it in the
environment, and writes them as ready-to-insert dialogue turns. No test recipe is ever
shown to the model.

Notes (FR) — Construit les exemples few-shot pour l'éval de prompting (exp18).

Source des exemples : les items de l'univers TextCraft ABSENTS du train ET du
test (70 items : 18 depth-2, 45 depth-3, 7 depth-4 — aucun depth-1 disponible),
donc zéro contamination. Pour chaque item held-out :

  1. la solution optimale (suite get/craft) est dérivée de CraftingTree en
     suivant la même logique min-depth que l'env (recette minimisant
     max(depth(inputs))+1 ; tags résolus vers un item concret gettable de
     préférence, sinon min-depth) ;
  2. la solution est VALIDÉE en la rejouant dans un TextCraftEnv réel
     (chaque action doit réussir, reward final = 1) — un exemple qui ne
     se rejoue pas n'est pas écrit ;
  3. l'exemple est rendu en bloc texte : recettes pertinentes (sans
     distracteurs), goal, un Thought court, puis les paires Action/Observation
     réelles du replay.

Ordre des exemples (déterministe) : profondeurs entrelacées selon le cycle
d2,d3,d2,d3,d4 (les k premiers exemples forment donc toujours un mix équilibré,
et les préfixes sont emboîtés : exemples(k=3) ⊂ exemples(k=10) ⊂ …) ; à
profondeur égale, solutions courtes d'abord.

Sortie : data/eval/textcraft_fewshot_examples.json
  [{"idx", "item", "depth", "n_actions", "n_recipes", "block"}, ...]

À lancer depuis external/AgentGym/agentenv-textcraft/ (env agentenv-textcraft,
même contrainte de cwd que label_depths.py) :
    source ~/envs/agentenv-textcraft/bin/activate
    cd external/AgentGym/agentenv-textcraft
    python /home/criteo/rl-gym-workout/src/utils/build_fewshot_examples.py
"""

from __future__ import annotations

import json
import os
from math import ceil
from pathlib import Path

from agentenv_textcraft.crafting_tree import CraftingTree
from agentenv_textcraft.environment import TextCraftEnv
from agentenv_textcraft.utils import item_id_to_str

REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/criteo/rl-gym-workout"))
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"
TEST_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
OUT_PATH = REPO_ROOT / "data" / "eval" / "textcraft_fewshot_examples.json"

DEPTH_CYCLE = [2, 3, 2, 3, 4]  # ordre d'entrelacement des profondeurs


# ---------------------------------------------------------------------------
# Solveur : item → suite d'actions get/craft (logique min-depth de l'env)
# ---------------------------------------------------------------------------

def min_depth_recipe(tree: CraftingTree, item: str):
    """La recette que get_min_depth_recipes considère comme la plus simple."""
    return min(
        tree.itemid_recipes[item],
        key=lambda r: max(tree.get_min_depth(i.item_tag.name) + 1 for i in r.input_items),
    )


def resolve_tag(tree: CraftingTree, tag: str) -> str:
    """Tag → item concret : gettable (non craftable) de préférence, sinon min-depth."""
    cands = sorted(tree.get_items_with_tags(tag))
    non_craftable = [c for c in cands if not tree.is_craftable(c)]
    if non_craftable:
        return non_craftable[0]
    return min(cands, key=lambda c: (tree.get_min_depth(c), c))


def solve(tree: CraftingTree, item: str, need: int,
          actions: list[str], used_recipes: list) -> None:
    """Ajoute à `actions` la suite get/craft produisant >= need exemplaires d'item."""
    recipe = min_depth_recipe(tree, item)
    times = ceil(need / recipe.output_item.count)
    concrete_inputs: list[tuple[str, int]] = []
    for inp in recipe.input_items:
        concrete = inp.item_tag.item_id or resolve_tag(tree, inp.item_tag.tag)
        total = inp.count * times
        if tree.is_craftable(concrete):
            solve(tree, concrete, total, actions, used_recipes)
        else:
            actions.append(f"get {total} {item_id_to_str(concrete)}")
        concrete_inputs.append((concrete, inp.count))
    if recipe not in used_recipes:
        used_recipes.append(recipe)
    craft_cmd = (
        f"craft {recipe.output_item.count} {item_id_to_str(item)} using "
        + ", ".join(f"{c} {item_id_to_str(n)}" for n, c in concrete_inputs)
    )
    actions.extend([craft_cmd] * times)


# ---------------------------------------------------------------------------
# Validation par replay dans l'env réel + rendu du bloc
# ---------------------------------------------------------------------------

def replay(tree: CraftingTree, idx: int, actions: list[str]) -> list[str] | None:
    """Rejoue les actions dans TextCraftEnv (reset(data_idx=idx)). Retourne les
    observations si tout réussit ET reward final = 1, sinon None."""
    env = TextCraftEnv(tree, None, None)
    env.reset(data_idx=idx)
    observations: list[str] = []
    reward = 0
    for act in actions:
        obs, reward, terminated, _, _ = env.step(act)
        low = obs.lower()
        if "could not" in low or "wrong item format" in low:
            print(f"    ÉCHEC replay idx={idx} sur {act!r} → {obs!r}")
            return None
        observations.append(obs)
        if terminated:
            break
    if reward != 1:
        print(f"    ÉCHEC replay idx={idx} : reward final {reward} != 1")
        return None
    if len(observations) != len(actions):
        print(f"    ÉCHEC replay idx={idx} : terminé avant la dernière action")
        return None
    return observations


def render_block(goal_item: str, used_recipes: list, actions: list[str],
                 observations: list[str]) -> str:
    goal_str = item_id_to_str(goal_item)
    lines = ["Crafting commands:"]
    lines += [r.recipe_str for r in used_recipes]
    lines.append("")
    lines.append(f"Goal: craft {goal_str}.")
    lines.append(
        "Thought: I check which ingredients are base items I can get, craft the "
        "intermediate items first, then craft the goal."
    )
    for act, obs in zip(actions, observations):
        lines.append(f"Action: {act}")
        lines.append(f"Observation: {obs}")
    return "\n".join(lines)


def main() -> None:
    tree = CraftingTree(minecraft_dir="agentenv_textcraft/")
    sorted_items = sorted(list(tree.item_recipes_min_depth(1)), key=lambda x: x[1])

    with TRAIN_PATH.open() as f:
        train_idx = {int(r["item_id"].rsplit("_", 1)[1]) for r in json.load(f)}
    with TEST_PATH.open() as f:
        test_idx = {int(r["item_id"].rsplit("_", 1)[1]) for r in json.load(f)}
    used_idx = train_idx | test_idx

    held = [(i, item, int(d)) for i, (item, d) in enumerate(sorted_items) if i not in used_idx]
    print(f"[fewshot] {len(held)} items hors train+test")

    examples = []
    for idx, item, depth in held:
        actions: list[str] = []
        used_recipes: list = []
        try:
            solve(tree, item, 1, actions, used_recipes)
        except Exception as e:
            print(f"    ÉCHEC solveur idx={idx} ({item}) : {e!r}")
            continue
        observations = replay(tree, idx, actions)
        if observations is None:
            continue
        examples.append({
            "idx": idx,
            "item": item,
            "depth": depth,
            "n_actions": len(actions),
            "n_recipes": len(used_recipes),
            # Champs structurés : permettent l'injection en format DIALOGUE
            # (vrais tours user/assistant) — cf. eval_textcraft --fewshot-format.
            "goal_str": item_id_to_str(item),
            "commands": [r.recipe_str for r in used_recipes],
            "thought": ("I check which ingredients are base items I can get, craft "
                        "the intermediate items first, then craft the goal."),
            "steps": [{"action": a, "observation": o}
                      for a, o in zip(actions, observations)],
            # Rendu monolithique (format « bloc », conservé pour l'ablation exp18)
            "block": render_block(item, used_recipes, actions, observations),
        })
    print(f"[fewshot] {len(examples)}/{len(held)} exemples validés par replay (reward=1)")

    # Ordre : cycle de profondeurs d2,d3,d2,d3,d4 ; à depth égale, court d'abord.
    buckets = {d: sorted([e for e in examples if e["depth"] == d], key=lambda e: e["n_actions"])
               for d in (2, 3, 4)}
    ordered = []
    c = 0
    while any(buckets.values()):
        d = DEPTH_CYCLE[c % len(DEPTH_CYCLE)]
        c += 1
        if buckets[d]:
            ordered.append(buckets[d].pop(0))
        elif not buckets[d]:
            # bucket vide : prend dans le plus rempli pour ne pas boucler à vide
            fallback = max(buckets, key=lambda k: len(buckets[k]))
            if buckets[fallback]:
                ordered.append(buckets[fallback].pop(0))

    with OUT_PATH.open("w") as f:
        json.dump(ordered, f, indent=1, ensure_ascii=False)
    from collections import Counter
    print(f"[fewshot] écrit {len(ordered)} exemples → {OUT_PATH}")
    print(f"[fewshot] ordre des 10 premiers (depth) : {[e['depth'] for e in ordered[:10]]}")
    print(f"[fewshot] distribution : {dict(Counter(e['depth'] for e in ordered))}")
    print(f"[fewshot] taille moyenne bloc : "
          f"{sum(len(e['block']) for e in ordered) // len(ordered)} caractères")
    print("\n───── Exemple 1 ─────\n" + ordered[0]["block"])


if __name__ == "__main__":
    main()
