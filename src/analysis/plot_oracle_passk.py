"""
Génère les 3 figures oracle pass@k superposées pour Qwen2.5-3B baseline,
meilleur checkpoint RL (exp7.3) et Qwen3.5-4B.

Usage :
    python src/analysis/plot_oracle_passk.py runs/7_oracle/oracle_qwen35_4b --model-label "Qwen3.5-4B"

Sorties (dans le même dossier que passes.jsonl) :
    passk_curves.png       — courbes pass@k par depth (k=1..N)
    passk_histograms.png   — barres pass@1/10/18 par depth
    distribution_curves.png — distribution réussites/item (brut + KDE)
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"

# Modèles de référence à superposer automatiquement (oracle_passk.json précomputés)
REFERENCE_MODELS = [
    {
        "run_dir": REPO_ROOT / "runs" / "7_oracle" / "oracle_baseline",
        "label": "Qwen2.5-3B (base)",
        "color": "gray",
        "linestyle": "--",
    },
    {
        "run_dir": REPO_ROOT / "runs" / "7_oracle" / "oracle_best32",
        "label": "Qwen2.5-3B RL (best32)",
        "color": "tomato",
        "linestyle": "-.",
    },
]


# ── estimateur non biaisé pass@k (Chen et al. 2021) ──────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ── lecture données ───────────────────────────────────────────────────────────

def load_passes(jsonl_path: Path) -> tuple[dict[str, int], list[str], int]:
    """Retourne (successes_per_item, items, N_passes) depuis passes.jsonl."""
    lines = jsonl_path.read_text().strip().splitlines()
    N = len(lines)
    totals: dict[str, int] = defaultdict(int)
    items: list[str] = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        if i == 0:
            items = row["items"]
        for item_id, s in zip(row["items"], row["successes"]):
            totals[item_id] += s
    return dict(totals), items, N


def load_oracle_json(json_path: Path) -> tuple[dict[str, dict[int, float]], int]:
    """Charge oracle_passk.json précomputé.
    Retourne (passk_by_depth, N) où passk_by_depth[depth][k] = float."""
    d = json.loads(json_path.read_text())
    N = d["n_samples"]
    by_depth = {
        int(depth): {int(k): v for k, v in kv.items()}
        for depth, kv in d["pass_at_k_by_depth"].items()
    }
    return by_depth, N


# ── calcul pass@k ─────────────────────────────────────────────────────────────

def compute_passk_for_group(item_ids: list[str], totals: dict[str, int], N: int) -> dict[int, float]:
    """pass@k pour un groupe d'items, k=1..N."""
    return {
        k: float(np.mean([pass_at_k(N, totals[it], k) for it in item_ids]))
        for k in range(1, N + 1)
    }


# ── figure 1 : courbes pass@k par depth ──────────────────────────────────────

def plot_passk_curves(
    series: list[dict],   # [{"label", "color", "linestyle", "by_depth": {d: {k: float}}, "N": int}]
    depths: list[int],
    out_path: Path,
) -> None:
    ncols = 2
    nrows = (len(depths) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    for i, d in enumerate(depths):
        ax = axes[i]
        for s in series:
            if d not in s["by_depth"]:
                continue
            passk = s["by_depth"][d]
            ks = sorted(passk.keys())
            vals = [100 * passk[k] for k in ks]
            ax.plot(ks, vals, marker="o", markersize=4, linewidth=2,
                    color=s["color"], linestyle=s["linestyle"], label=s["label"])
        ax.set_title(f"depth {d}", fontsize=13)
        ax.set_xlabel("k (essais)")
        ax.set_ylabel("pass@k (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for j in range(len(depths), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Oracle pass@k par depth — TextCraft 100 items", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── figure 2 : histogrammes pass@1/10/18 par depth ───────────────────────────

def plot_passk_histograms(
    series: list[dict],
    depths: list[int],
    show_ks: list[int],
    out_path: Path,
) -> None:
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(len(depths))
    depth_labels = [f"d{d}" for d in depths]

    fig, axes = plt.subplots(1, len(show_ks), figsize=(5 * len(show_ks), 5))
    if len(show_ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, show_ks):
        for si, s in enumerate(series):
            vals = []
            for d in depths:
                passk = s["by_depth"].get(d, {})
                # clamp k au max disponible pour ce modèle
                k_use = min(k, s["N"])
                vals.append(100 * passk.get(k_use, 0.0))
            offset = (si - (n_series - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, color=s["color"],
                          label=s["label"], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"pass@{k}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(depth_labels)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% items résolus")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("pass@k par depth — TextCraft 100 items", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── figure 3 : distribution réussites/item + KDE ─────────────────────────────

def plot_distribution_curves(
    series: list[dict],   # chaque entrée a aussi "totals": {item_id: int}, "items": list
    depth_groups: dict[int, list[str]],
    all_items: list[str],
    out_path: Path,
) -> None:
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    from collections import Counter

    depths = sorted(depth_groups.keys())
    group_keys = ["Global"] + [f"depth {d}" for d in depths]
    group_items = [all_items] + [depth_groups[d] for d in depths]

    ncols = 3
    nrows = (len(group_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    N_max = max(s["N"] for s in series)

    for i, (title, items) in enumerate(zip(group_keys, group_items)):
        ax = axes[i]
        for s in series:
            totals = s["totals"]
            N = s["N"]
            counts = [totals.get(it, 0) for it in items]
            # normalise les counts à N_max pour comparer sur même axe
            counts_norm = [c / N * N_max for c in counts]

            freq = Counter([round(c) for c in counts_norm])
            xs = sorted(freq.keys())
            ys = [freq[x] for x in xs]
            ax.scatter(xs, ys, color=s["color"], alpha=0.5, s=30)

            if has_scipy and len(set(counts_norm)) > 2:
                x_kde = np.linspace(0, N_max, 300)
                try:
                    kde = gaussian_kde(counts_norm, bw_method=0.4)
                    y_kde = kde(x_kde) * len(counts_norm)
                    ax.plot(x_kde, y_kde, color=s["color"], linewidth=2,
                            linestyle=s["linestyle"], label=f"{s['label']} (KDE)")
                except Exception:
                    ax.plot([], [], color=s["color"], linewidth=2,
                            linestyle=s["linestyle"], label=s["label"])
            else:
                ax.plot([], [], color=s["color"], linewidth=2,
                        linestyle=s["linestyle"], label=s["label"])

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"nb réussites (normalisé sur {N_max})")
        ax.set_ylabel("nombre d'items")
        ax.set_xlim(-0.5, N_max + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(len(group_keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distribution réussites/item (normalisée sur {N_max}) — points=brut, courbe=KDE",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Dossier contenant passes.jsonl")
    parser.add_argument("--model-label", default="Qwen3.5-4B", help="Label du modèle principal")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    jsonl_path = run_dir / "passes.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"passes.jsonl introuvable dans {run_dir}")

    depth_map: dict[str, int] = json.loads(DEPTH_PATH.read_text())

    # ── charger le modèle principal (passes.jsonl) ──
    print(f"[plot_oracle_passk] lecture {jsonl_path} ...")
    totals_main, all_items, N_main = load_passes(jsonl_path)
    print(f"  {N_main} passes, {len(all_items)} items")

    depth_groups: dict[int, list[str]] = defaultdict(list)
    for it in all_items:
        depth_groups[depth_map.get(it, 0)].append(it)
    depths = sorted(depth_groups.keys())
    for d in depths:
        print(f"  depth {d}: {len(depth_groups[d])} items")

    # pass@k par depth pour le modèle principal
    by_depth_main = {
        d: compute_passk_for_group(depth_groups[d], totals_main, N_main)
        for d in depths
    }

    main_series = {
        "label": args.model_label,
        "color": "steelblue",
        "linestyle": "-",
        "by_depth": by_depth_main,
        "N": N_main,
        "totals": totals_main,
        "items": all_items,
    }

    # ── charger les modèles de référence (oracle_passk.json) ──
    all_series = []
    for ref in REFERENCE_MODELS:
        json_path = ref["run_dir"] / "oracle_passk.json"
        if not json_path.exists():
            print(f"  [skip] {json_path} introuvable")
            continue
        by_depth_ref, N_ref = load_oracle_json(json_path)
        # ne garder que les depths présents dans le modèle principal
        by_depth_ref = {d: by_depth_ref[d] for d in depths if d in by_depth_ref}
        # reconstruire totals depuis successes_per_item pour la distribution
        d_json = json.loads(json_path.read_text())
        totals_ref = {k: v for k, v in d_json["successes_per_item"].items()}
        all_series.append({
            "label": ref["label"],
            "color": ref["color"],
            "linestyle": ref["linestyle"],
            "by_depth": by_depth_ref,
            "N": N_ref,
            "totals": totals_ref,
            "items": all_items,
        })
        print(f"  chargé {ref['label']} (N={N_ref})")

    all_series.append(main_series)

    # k communs pour les histogrammes (k=1, k=10, k=min_N)
    N_min = min(s["N"] for s in all_series)
    show_ks = sorted({1, 10, N_min})

    print(f"\n[plot_oracle_passk] génération (séries={[s['label'] for s in all_series]}, show_ks={show_ks}) ...")

    plot_passk_curves(all_series, depths, run_dir / "passk_curves.png")
    plot_passk_histograms(all_series, depths, show_ks, run_dir / "passk_histograms.png")
    plot_distribution_curves(all_series, depth_groups, all_items, run_dir / "distribution_curves.png")

    print("\n[plot_oracle_passk] done.")


if __name__ == "__main__":
    main()
