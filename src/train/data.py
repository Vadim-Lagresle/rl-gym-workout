"""The training tasks and the prompts the model starts each episode with.

In plain words: loads the list of training recipes (data/train/textcraft_train.json by
default, or another file with --train-file), optionally keeps only some depths, and
builds for each task the chat prompt: the game rules sent by the TextCraft server,
optional solved examples (few-shot), and a hidden marker that the episode loop replaces
by the actual task. Also holds the shared paths (repository root, base model, server
address).

Notes (FR) — Dataset d'entraînement et prompts TextCraft.

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
# Le modèle vit sur /tmp depuis 2026-07-29 (home 35 Go trop petit ; /tmp volatil
# mais retéléchargeable en ~2 min via setup/ensure_qwen_tmp.sh).
DEFAULT_MODEL_PATH = Path("/tmp/models/Qwen2.5-3B-Instruct")
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"


def depth_file_for(train_path: Path) -> Path:
    """Fichier de depths associé à un fichier de train : <stem>_with_depth.json à côté."""
    return train_path.with_name(train_path.stem + "_with_depth.json")
ENV_SERVER_URL = "http://127.0.0.1:36005"

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Marqueur caché inséré dans le dernier message user du prompt : il encode l'index
# de l'item pour la rollout_func (qui le remplace par l'observation env réelle).
# Non ancré : avec le few-shot (exp19), le marqueur cohabite dans le même message
# avec l'observation finale du dernier exemple.
ITEM_TAG_RE = re.compile(r"<ITEM_IDX:(\d+)>")


def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    """Check that the TextCraft server is running and reachable before any model interaction."""
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int, max_depth: int = 0, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                      depth_in: set[int] | None = None,
                      fewshot_block: str | None = None,
                      fewshot_messages: list[dict] | None = None,
                      train_path: Path | None = None) -> list[dict[str, Any]]:
    """Build the list of prompt rows for rollout_func, optionally filtering by depth and max_items.

    Deux modes de filtrage par depth (exclusifs) :
      - max_depth > 0   : garde les items de depth <= max_depth (curriculum cumulatif).
      - depth_in donné  : garde uniquement les items dont depth ∈ depth_in (curriculum par
                          stage, ex. {1} ou {3, 4}). Utilisé par run_curriculum_staged.sh.

    Few-shot (exp19) : fewshot_block est ajouté à la fin du message de règles,
    fewshot_messages (tours user/assistant, cf. textcraft_common.load_fewshot) est
    inséré entre l'ack et le marqueur ; l'observation finale du dernier exemple est
    fusionnée avec le message porteur du marqueur <ITEM_IDX:n> (pas de tours user
    consécutifs). Les tokens des exemples sont dans prompt_ids → jamais de gradient."""

    train_path = Path(train_path) if train_path else TRAIN_PATH
    with train_path.open() as f:
        rows = json.load(f)
    if train_path != TRAIN_PATH:
        print(f"[data] fichier de train : {train_path} ({len(rows)} items)", flush=True)

    if max_depth > 0 or depth_in:
        # Filtre par depth. Requiert le mapping pré-calculé
        # data/train/textcraft_train_with_depth.json (src/utils/label_depths.py).
        depth_file = depth_file_for(train_path)
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
    rules_msg = manual_human + ("\n\n" + fewshot_block if fewshot_block else "")
    example_turns: list[dict] = []
    carry = None
    if fewshot_messages:
        example_turns = list(fewshot_messages)
        if example_turns and example_turns[-1]["role"] == "user":
            carry = example_turns.pop()["content"]

    out: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        idx = item_id_to_idx(item_id)
        marker = (f"{carry}\n\nNow solve this new task:\n\n<ITEM_IDX:{idx}>"
                  if carry else f"<ITEM_IDX:{idx}>")  # marqueur caché pour rollout_func
        base = [{"role": "system", "content": system_prompt}] if system_prompt else []
        prompt = base + [
            {"role": "user", "content": rules_msg},
            {"role": "assistant", "content": manual_ack},
            *example_turns,
            {"role": "user", "content": marker},
        ]
        out.append({"prompt": prompt, "item_id": item_id, "item_idx": idx})

    try:
        probe.close()
    except Exception:
        pass
    return out
