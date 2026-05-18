#!/usr/bin/env python3
"""
Agrège les logs `textcraft_*.json` produits par `scratch/03_eval_qwen.py`
dans un répertoire donné (ex. `scratch/eval_logs_4gpu_global_step_50/`).

Écrit :
  - scores_table.csv — une ligne par item (success, reward, rounds, …)
  - scores_summary.json — totaux et taux de réussite

Usage :
  python scratch/summarize_textcraft_eval_dir.py scratch/eval_logs_4gpu_global_step_50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "log_dir",
        type=Path,
        help="Répertoire contenant textcraft_*.json",
    )
    args = ap.parse_args()
    log_dir: Path = args.log_dir.resolve()
    if not log_dir.is_dir():
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for p in sorted(log_dir.glob("textcraft_*.json"), key=lambda x: int(x.stem.split("_")[-1])):
        d = json.loads(p.read_text())
        r = float(d["reward"])
        succ = 1 if r > 0 else 0
        rows.append(
            {
                "item_id": d["item_id"],
                "item_idx": d["item_idx"],
                "success": succ,
                "reward": r,
                "done": int(bool(d["done"])),
                "rounds": d["rounds"],
                "duration_s": d["duration_s"],
            }
        )

    if not rows:
        print(f"No textcraft_*.json under {log_dir}", file=sys.stderr)
        return 1

    n = len(rows)
    tot_succ = sum(r["success"] for r in rows)
    mean_r = sum(r["reward"] for r in rows) / n
    mean_rounds = sum(r["rounds"] for r in rows) / n

    out_csv = log_dir / "scores_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "log_dir": str(log_dir),
        "episodes": n,
        "successes": tot_succ,
        "failures": n - tot_succ,
        "success_rate_pct": round(100.0 * tot_succ / n, 2),
        "mean_reward": round(mean_r, 4),
        "mean_rounds": round(mean_rounds, 2),
        "metric_note": (
            "Avec 1 rollout par item, le taux de réussite = successes/episodes. "
            "Le script d’eval l’affiche aussi sous le nom « Pass@1 » (k essais "
            "indépendants par problème, ici k=1). Ce n’est pas la limite de tours "
            "LLM↔env (voir MAX_ROUNDS dans 03_eval_qwen.py)."
        ),
    }
    out_sum = log_dir / "scores_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {out_csv} ({n} rows)")
    print(f"Wrote {out_sum}")
    print(
        f"successes={tot_succ}/{n}  success_rate={summary['success_rate_pct']}%  "
        f"mean_rounds={summary['mean_rounds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
