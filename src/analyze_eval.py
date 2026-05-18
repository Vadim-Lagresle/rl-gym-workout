"""Analyse qualitative des 100 logs d'eval Qwen-3B sur TextCraft.

Catégorise les échecs et sort des stats globales pour comprendre
*pourquoi* le modèle échoue, afin d'orienter le design RL futur.

Lancement (depuis env conda agentgym-rl, ou n'importe quel python3) :
    python scratch/04_analyze_eval.py
    python scratch/04_analyze_eval.py --examples  # affiche 2-3 transcripts
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "runs" / "exp1_baseline" / "eval_logs"

ACTION_RE = re.compile(r"^Action:\s*(.+?)\s*$", re.MULTILINE)
INVALID_PATTERNS = (
    "invalid",       # générique
    "could not",     # "could not parse..."
    "no valid",      # "no valid action found"
    "format",        # "format error"
    "not a valid",   # "X is not a valid action"
    "must be",       # contraintes du format
)


def parse_log(fp: Path) -> dict:
    with fp.open() as f:
        return json.load(f)


def count_actions_in_message(content: str) -> int:
    """Combien de lignes 'Action: ...' dans la réponse LLM."""
    return len(ACTION_RE.findall(content))


def is_env_complaint(env_msg: str) -> bool:
    """Détecte une réponse env qui signale une violation de format / action invalide."""
    low = env_msg.lower()
    return any(p in low for p in INVALID_PATTERNS)


def actions_extracted(content: str) -> list[str]:
    return [m.strip().lower() for m in ACTION_RE.findall(content)]


def analyze(d: dict) -> dict:
    """Retourne un dict de features par épisode."""
    transcript = d["transcript"]
    assistant_msgs = [m["content"] for m in transcript if m["role"] == "assistant"]
    env_msgs_after_assistant = []
    for i, m in enumerate(transcript):
        if m["role"] == "assistant" and i + 1 < len(transcript):
            env_msgs_after_assistant.append(transcript[i + 1]["content"])

    n_assistant = len(assistant_msgs)
    multi_action_msgs = sum(1 for m in assistant_msgs if count_actions_in_message(m) > 1)
    no_action_msgs = sum(1 for m in assistant_msgs if count_actions_in_message(m) == 0)
    env_complaints = sum(1 for env in env_msgs_after_assistant if is_env_complaint(env))

    # Détection de loops : action exactement identique répétée >= 3 fois
    all_actions: list[str] = []
    for m in assistant_msgs:
        acts = actions_extracted(m)
        if acts:
            all_actions.append(acts[0])  # 1ʳᵉ action de chaque réponse
    action_counts = Counter(all_actions)
    most_common = action_counts.most_common(1)
    looped_action = most_common[0] if most_common else (None, 0)
    is_loop = looped_action[1] >= 3 and (looped_action[1] / max(n_assistant, 1)) > 0.4

    avg_assistant_len = (
        sum(len(m) for m in assistant_msgs) / n_assistant if n_assistant else 0
    )

    return {
        "item_id": d["item_id"],
        "reward": d["reward"],
        "rounds": d["rounds"],
        "n_assistant": n_assistant,
        "multi_action_msgs": multi_action_msgs,
        "no_action_msgs": no_action_msgs,
        "env_complaints": env_complaints,
        "is_loop": is_loop,
        "looped_action": looped_action[0],
        "looped_count": looped_action[1],
        "avg_assistant_len": round(avg_assistant_len, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", action="store_true", help="affiche 2-3 transcripts")
    args = parser.parse_args()

    files = sorted(LOG_DIR.glob("textcraft_*.json"))
    rows = [analyze(parse_log(fp)) for fp in files]
    succ = [r for r in rows if r["reward"] > 0]
    fail = [r for r in rows if r["reward"] == 0]

    print("=" * 70)
    print(f"Analyse de {len(rows)} épisodes ({len(succ)} succès, {len(fail)} échecs)")
    print("=" * 70)

    def stats(label: str, group: list[dict]) -> None:
        if not group:
            return
        n = len(group)
        avg_rounds = sum(r["rounds"] for r in group) / n
        avg_multi = sum(r["multi_action_msgs"] for r in group) / n
        avg_noaction = sum(r["no_action_msgs"] for r in group) / n
        avg_complaints = sum(r["env_complaints"] for r in group) / n
        avg_len = sum(r["avg_assistant_len"] for r in group) / n
        n_loops = sum(1 for r in group if r["is_loop"])
        print(f"\n[{label}]  n={n}")
        print(f"  mean rounds                : {avg_rounds:.1f}")
        print(f"  mean multi-action msgs     : {avg_multi:.2f}  (≥2 'Action:' dans 1 réponse → env rejette)")
        print(f"  mean no-action msgs        : {avg_noaction:.2f}  (réponse sans 'Action:' du tout)")
        print(f"  mean env complaints/format : {avg_complaints:.2f}")
        print(f"  mean assistant msg length  : {avg_len:.0f} chars")
        print(f"  episodes with action loop  : {n_loops} ({100 * n_loops / n:.0f}%)  (même 1ʳᵉ action ≥40% du temps, ≥3 fois)")

    stats("SUCCESS", succ)
    stats("FAILURE (timeout)", fail)

    print("\n" + "=" * 70)
    print("Catégories d'échec (overlap possible)")
    print("=" * 70)
    bucket = {
        "format_violations_heavy": [r for r in fail if r["multi_action_msgs"] >= 5],
        "any_format_violation": [r for r in fail if r["multi_action_msgs"] >= 1],
        "no_action_at_all": [r for r in fail if r["no_action_msgs"] >= 5],
        "action_loop": [r for r in fail if r["is_loop"]],
        "env_complaints_heavy": [r for r in fail if r["env_complaints"] >= 5],
    }
    for label, lst in bucket.items():
        print(f"  {label:30s}: {len(lst)}/{len(fail)} ({100 * len(lst) / max(len(fail), 1):.0f}%)")

    if args.examples:
        print("\n" + "=" * 70)
        print("EXEMPLES CONCRETS")
        print("=" * 70)
        succ_short = sorted(succ, key=lambda r: r["rounds"])[:1]
        fail_format = next((r for r in fail if r["multi_action_msgs"] >= 5), None)
        fail_loop = next((r for r in fail if r["is_loop"]), None)
        for kind, r in [("✅ SUCCESS court", succ_short[0] if succ_short else None),
                        ("❌ FAILURE — multi-actions", fail_format),
                        ("❌ FAILURE — loop", fail_loop)]:
            if r is None:
                continue
            print(f"\n--- {kind} : {r['item_id']} ({r['rounds']} rounds, multi={r['multi_action_msgs']}, "
                  f"loop={r['looped_action']!r} x{r['looped_count']}) ---")
            d = parse_log(LOG_DIR / f"{r['item_id']}.json")
            for i, m in enumerate(d["transcript"][:8]):
                content = m["content"][:280].replace("\n", "\n    ")
                print(f"  [{m['role']:9s}] {content}")
            if len(d["transcript"]) > 8:
                print(f"  ... [{len(d['transcript']) - 8} more messages] ...")
                last = d["transcript"][-1]
                content = last["content"][:280].replace("\n", "\n    ")
                print(f"  [{last['role']:9s}] {content}")


if __name__ == "__main__":
    main()
