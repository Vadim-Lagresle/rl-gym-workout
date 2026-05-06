"""Visualisation lisible d'un log d'eval avec annotation des "vrais" auteurs.

Le format chat de Qwen impose 3 rôles : `system`, `user`, `assistant`. Mais
dans notre setup, "user" est ambigu : c'est tantôt nous (la consigne au
démarrage), tantôt le serveur TextCraft (les observations et messages
d'erreur). Ce viewer annote chaque message avec son auteur réel.

Légende des auteurs :
- 📜 setup    = nous (script `03_eval_qwen.py`), définit le rôle système
- 📚 manuel   = AgentGym (`TextCraftEnvClient.conversation_start`), donne
                les règles du jeu et l'ack scripté
- 🎮 env      = le serveur TextCraft (réponses aux actions, observations)
- 🤖 Qwen     = le modèle (les vraies réponses générées)

Lancement :
    python scratch/05_view_log.py textcraft_17        # 1 log
    python scratch/05_view_log.py textcraft_17 --raw  # transcript brut sans annotation
    python scratch/05_view_log.py 17                  # raccourci sur l'index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "eval_logs"

ACK_TEXT = "OK. I'll follow your instructions and try my best to solve the task."


def classify(role: str, content: str, idx: int) -> str:
    """Retourne l'auteur réel d'un message du transcript."""
    if role == "system":
        return "📜 setup"
    if role == "assistant":
        if idx == 2 and content.strip() == ACK_TEXT:
            return "📚 manuel (ack scripté)"
        return "🤖 Qwen"
    if role == "user":
        if idx == 1:
            return "📚 manuel (règles du jeu)"
        return "🎮 env"
    return f"? {role}"


def render(log: dict, raw: bool = False) -> None:
    print("=" * 80)
    print(f"Item: {log['item_id']}  |  reward={log['reward']}  |  rounds={log['rounds']}  |  duration={log['duration_s']}s")
    print("=" * 80)
    for i, m in enumerate(log["transcript"]):
        author = m["role"] if raw else classify(m["role"], m["content"], i)
        print(f"\n--- [{i:2d}] {author} ---")
        print(m["content"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("item", help="item_id (e.g. textcraft_17) or just the index (e.g. 17)")
    parser.add_argument("--raw", action="store_true", help="transcript brut, sans annotation des auteurs")
    args = parser.parse_args()

    name = args.item if args.item.startswith("textcraft_") else f"textcraft_{args.item}"
    fp = LOG_DIR / f"{name}.json"
    if not fp.exists():
        raise FileNotFoundError(f"Log not found: {fp}")
    with fp.open() as f:
        log = json.load(f)
    render(log, raw=args.raw)


if __name__ == "__main__":
    main()
