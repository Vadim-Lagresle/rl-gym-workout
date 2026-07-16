"""Dataset d'entraînement et prompts TextCraft.

Charge data/train/textcraft_train.json, filtre optionnellement par depth
(curriculum), et construit les prompts chat avec le marqueur <ITEM_IDX:n>
que la rollout_func remplace par l'observation initiale de l'environnement.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from agentenv.envs import TextCraftEnvClient

# Paths
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Marqueur caché inséré comme dernier message user du prompt : il encode l'index
# de l'item pour la rollout_func (qui le remplace par l'observation env réelle).
ITEM_TAG_RE = re.compile(r"^<ITEM_IDX:(\d+)>$")


def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    """Check that the TextCraft server is running and reachable before any model interaction."""
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int, max_depth: int = 0, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                      depth_in: set[int] | None = None) -> list[dict[str, Any]]:
    """Build the list of prompt rows for rollout_func, optionally filtering by depth and max_items.

    Deux modes de filtrage par depth (exclusifs) :
      - max_depth > 0   : garde les items de depth <= max_depth (curriculum cumulatif).
      - depth_in donné  : garde uniquement les items dont depth ∈ depth_in (curriculum par
                          stage, ex. {1} ou {3, 4}). Utilisé par run_curriculum_staged.sh."""

    with TRAIN_PATH.open() as f:
        rows = json.load(f)

    if max_depth > 0 or depth_in:
        # Filtre par depth. Requiert le mapping pré-calculé
        # data/train/textcraft_train_with_depth.json (src/utils/label_depths.py).
        depth_file = REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"
        if not depth_file.exists():
            raise FileNotFoundError(
                f"--max-depth/--depth-exact requires {depth_file}. "
                "Generate it with: <agentenv-textcraft python> src/utils/label_depths.py"
            )
        with depth_file.open() as f:
            depth_map: dict[str, int] = json.load(f)
        if depth_in:
            rows = [r for r in rows if depth_map.get(r["item_id"], 99) in depth_in]
            print(f"[curriculum] depth_exact={sorted(depth_in)} → {len(rows)} items retained.", flush=True)
        else:
            rows = [r for r in rows if depth_map.get(r["item_id"], 99) <= max_depth]
            print(f"[curriculum] max_depth={max_depth} → {len(rows)} items retained.", flush=True)

    if max_items > 0:
        rows = rows[:max_items]

    # Amorce de conversation (human + ack) récupérée du serveur TextCraft.
    # La première réponse du LLM est précodée dans conversation_start[1].
    probe = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(rows), timeout=60)
    manual_human = probe.conversation_start[0]["value"]
    manual_ack = probe.conversation_start[1]["value"]

    # Assemblage des prompts. Si system_prompt est vide (ex. Gemma-3 qui refuse
    # le rôle system), on démarre directement sur le message user.
    out: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        idx = item_id_to_idx(item_id)
        base = [{"role": "system", "content": system_prompt}] if system_prompt else []
        prompt = base + [
            {"role": "user", "content": manual_human},
            {"role": "assistant", "content": manual_ack},
            {"role": "user", "content": f"<ITEM_IDX:{idx}>"},  # marqueur caché pour rollout_func
        ]
        out.append({"prompt": prompt, "item_id": item_id, "item_idx": idx})

    try:
        probe.close()
    except Exception:
        pass
    return out
