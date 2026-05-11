"""Compare Pass@1 of four TextCraft eval runs on the same 100 items.

Variants:
  - baseline    -> scratch/eval_logs
  - LoRA step10 -> scratch/eval_logs_step10
  - LoRA step50 -> scratch/eval_logs_step50  (v2)
  - LoRA step50 -> scratch/eval_logs_step50_v3 (v3, with reward shaping fix)

Usage:
    python scratch/09_compare_4way.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_BASE = REPO_ROOT / "scratch"

VARIANTS = {
    "baseline":  LOG_BASE / "eval_logs",
    "step10_v2": LOG_BASE / "eval_logs_step10",
    "step50_v2": LOG_BASE / "eval_logs_step50",
    "step50_v3": LOG_BASE / "eval_logs_step50_v3",
}


def load_results(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not d.exists():
        return out
    for fp in d.glob("textcraft_*.json"):
        try:
            with fp.open() as f:
                payload = json.load(f)
            item_id = payload.get("item_id") or fp.stem
            out[item_id] = {
                "reward":    float(payload.get("reward", 0.0)),
                "rounds":    int(payload.get("rounds", 0)),
                "duration":  float(payload.get("duration_s", 0.0)),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out


def passed(rec: dict | None) -> bool | None:
    if rec is None:
        return None
    return rec["reward"] > 0


def main() -> None:
    results: dict[str, dict[str, dict]] = {k: load_results(v) for k, v in VARIANTS.items()}
    for k, recs in results.items():
        print(f"[load] {k:10s}: {len(recs)} items from {VARIANTS[k]}")

    # Intersection across all four variants.
    common = set(results["baseline"].keys())
    for k in VARIANTS:
        common &= set(results[k].keys())
    common = sorted(common)
    print(f"\n[common] {len(common)} item_ids present in all 4 variants")

    if not common:
        print("[common] empty intersection — skipping per-item table")
        return

    # Pass@1 table (restricted to common set so the comparison is fair).
    print("\n" + "=" * 60)
    print(f"PASS@1 ON COMMON SET (n={len(common)})")
    print("=" * 60)
    print(f"{'variant':12s} | {'pass':>5s} | {'rate':>6s} | mean rounds")
    print("-" * 60)
    for k in VARIANTS:
        recs = results[k]
        n_pass = sum(1 for it in common if passed(recs[it]))
        rate = n_pass / len(common)
        mean_rounds = sum(recs[it]["rounds"] for it in common) / len(common)
        print(f"{k:12s} | {n_pass:>5d} | {rate*100:>5.1f}% | {mean_rounds:.1f}")

    # 4-way confusion (B, S10, S50_v2, S50_v3).
    print("\n" + "=" * 60)
    print("4-WAY CONFUSION (B, S10, S50_v2, S50_v3)")
    print("=" * 60)
    buckets: dict[tuple, list[str]] = {}
    for it in common:
        key = tuple(passed(results[k][it]) for k in ["baseline", "step10_v2", "step50_v2", "step50_v3"])
        buckets.setdefault(key, []).append(it)

    print(f"{'(B, S10, v2, v3)':24s} | count | items (truncated)")
    print("-" * 60)
    for key in sorted(buckets, key=lambda k: (-sum(k), k)):
        flag = "(" + ", ".join("T" if v else "F" for v in key) + ")"
        items = buckets[key]
        sample = ", ".join(items[:4]) + ("..." if len(items) > 4 else "")
        print(f"{flag:24s} | {len(items):5d} | {sample}")

    # Direct deltas vs v2 step50.
    base = results["baseline"]
    v2   = results["step50_v2"]
    v3   = results["step50_v3"]
    only_v3 = [it for it in common if passed(v3[it]) and not passed(v2[it])]
    only_v2 = [it for it in common if passed(v2[it]) and not passed(v3[it])]
    both    = [it for it in common if passed(v3[it]) and passed(v2[it])]
    base_pass = [it for it in common if passed(base[it])]
    v3_pass   = [it for it in common if passed(v3[it])]

    print("\n" + "=" * 60)
    print("DELTAS")
    print("=" * 60)
    print(f"  Resolved by v3 only (not v2)        : {len(only_v3)}  ({only_v3})")
    print(f"  Resolved by v2 only (not v3)        : {len(only_v2)}  ({only_v2})")
    print(f"  Resolved by both v2 and v3          : {len(both)}")
    print(f"  Resolved by baseline                : {len(base_pass)}")
    print(f"  Resolved by v3                      : {len(v3_pass)}")
    print(f"  Net shift v3 vs v2                  : {len(only_v3) - len(only_v2):+d}")
    print(f"  Net shift v3 vs baseline            : {len(v3_pass) - len(base_pass):+d}")


if __name__ == "__main__":
    main()
