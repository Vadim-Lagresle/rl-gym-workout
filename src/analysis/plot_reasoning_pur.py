#!/usr/bin/env python
"""Reasoning pur (single-turn) sur TextCraft — pass@1 oracle par depth.

Protocole exp16 : le LLM produit UN plan en texte libre (single-turn), un parseur
extrait les actions, puis on les rejoue dans TextCraft (pass@1 oracle, replay
deterministe). Pas de dimension pass@k -> comparaison en barres par depth.

Sources (config.yaml des runs) :
  - Qwen2.5-3B-Instruct : runs/exp16_single_turn_reasoning  -> 20/100
  - Qwen3.5-4B          : runs/exp16_qwen35_4b              -> 54/100

Sortie : docs/dashboard/09_reasoning_pur_par_depth.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "dashboard" / "09_reasoning_pur_par_depth.png"

# (solved, total) par depth, repris des config.yaml exp16.
DEPTHS = ["depth 1", "depth 2", "depth 3", "depth 4"]
MODELS = {
    "Qwen2.5-3B (20/100)": {
        "data": [(13, 31), (7, 41), (0, 25), (0, 3)],
        "color": "#4C72B0",
    },
    "Qwen3.5-4B (54/100)": {
        "data": [(24, 31), (22, 41), (8, 25), (0, 3)],
        "color": "#C44E52",
    },
}


def main() -> None:
    x = np.arange(len(DEPTHS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (name, spec) in enumerate(MODELS.items()):
        pct = [100.0 * s / t if t else 0.0 for s, t in spec["data"]]
        offset = (i - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, pct, width, label=name, color=spec["color"])
        for bar, (s, t) in zip(bars, spec["data"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{s}/{t}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={t})" for d, (_, t) in zip(DEPTHS, MODELS['Qwen2.5-3B (20/100)']['data'])])
    ax.set_ylabel("Pass@1 oracle (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Reasoning pur (single-turn) sur TextCraft — pass@1 oracle par depth\n"
                 "plan texte libre -> extraction d'actions -> replay deterministe",
                 fontsize=11)
    ax.legend(title="Modele", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.18, "depth 4 = mur universel (0/3) ; le modele plus recent gagne surtout aux depth 2-3.",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#555")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[plot] ecrit {OUT} ({OUT.stat().st_size} octets)")


if __name__ == "__main__":
    main()
