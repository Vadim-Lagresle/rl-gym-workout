"""Comparatif multi-modèles des evals TextCraft — tableaux complets.

Charge les eval_logs de plusieurs runs, calcule toutes les métriques comportementales
(pass@1, tours moy±std, erreurs par type, diversité, stagnation, récupération) globalement
et par depth, et imprime des tableaux. Base du futur dashboard (PNG matplotlib).

Usage : python src/analysis/compare_dashboard.py
"""
from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path
from statistics import pstdev

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "analysis"))
from analyze_eval import (  # noqa: E402
    analyze_episode, load_depth_map, ERROR_PATTERNS, ERROR_DESCRIPTIONS, _Tee,
)

# (label, run dir) — ordonnés par capacité croissante
RUNS = [
    ("Qwen2.5-0.5B", "exp10_qwen0.5b_baseline"),
    ("Qwen2.5-3B", "exp1_baseline"),
    ("Qwen3.5-4B", "exp15_qwen35_4b"),
    ("Gemini F3.5", "exp_gemini_gemini_3_5_flash"),
]
ETYPES = [e[0] for e in ERROR_PATTERNS]
DEPTHS = [1, 2, 3, 4]


def load() -> dict:
    depth = load_depth_map(REPO)
    data = {}
    for label, run in RUNS:
        paths = glob.glob(str(REPO / "runs" / run / "eval_logs" / "*.json"))
        data[label] = [analyze_episode(Path(p)) for p in paths]
    return data, depth


def recov_rate(res: list[dict]) -> float:
    att = sum(len(v) for x in res for v in x["recoveries"].values())
    rec = sum(sum(v) for x in res for v in x["recoveries"].values())
    return 100 * rec / att if att else 0.0


def hrow(label, vals, fmt="{}", w=13):
    print(f"{label:<20}" + "".join(f"{fmt.format(v):>{w}}" for v in vals))


# Couleurs cohérentes par modèle sur tous les graphes (palette lisible)
MODEL_COLORS = {
    "Qwen2.5-0.5B": "#bdbdbd",
    "Qwen2.5-3B": "#4e79a7",
    "Qwen3.5-4B": "#59a14f",
    "Gemini F3.5": "#e15759",
}

# Couleur dédiée et bien distincte pour chaque type d'erreur (graphe empilé #3)
ERROR_COLORS = {
    "format_error":   "#1f77b4",  # bleu franc
    "recipe_wrong":   "#2ca02c",  # vert
    "missing_items":  "#9467bd",  # violet
    "item_not_found": "#e377c2",  # rose
    "wrong_format":   "#bcbd22",  # olive
    "multi_action":   "#ff7f0e",  # orange vif
    "other_error":    "#7f7f7f",  # gris (résiduel)
    "generic_fail":   "#8c564b",  # brun (résiduel)
}


def make_plots(data: dict, depth: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless : sauvegarde PNG, pas d'affichage
    import matplotlib.pyplot as plt
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [l for l, _ in RUNS]
    colors = [MODEL_COLORS[l] for l in labels]
    x = np.arange(5)  # depth 1-4 + Global
    width = 0.2

    # ── 1) Pass@1 par depth + GLOBAL (barres groupées) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, l in enumerate(labels):
        vals = []
        for d in DEPTHS:
            r = [z for z in data[l] if depth.get(z["item_id"]) == d]
            vals.append(100 * sum(1 for z in r if z["reward"] > 0) / len(r) if r else 0)
        vals.append(100 * sum(1 for z in data[l] if z["reward"] > 0) / len(data[l]))  # global
        bars = ax.bar(x + (j - 1.5) * width, vals, width, label=l, color=colors[j])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["depth 1", "depth 2", "depth 3", "depth 4", "GLOBAL"])
    ax.axvline(3.5, color="gray", ls=":", lw=1)  # sépare depths / global
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("Pass@1 par profondeur de craft + global — TextCraft (100 items)")
    ax.set_ylim(0, 109)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "1_pass1_par_depth.png", dpi=130)
    plt.close(fig)

    # ── 2) Tours moyens vs depth (courbes + bande ±écart-type) ───────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for l in labels:
        means, stds = [], []
        for d in DEPTHS:
            rounds = [z["rounds"] for z in data[l] if depth.get(z["item_id"]) == d]
            from statistics import pstdev as _ps
            means.append(sum(rounds) / len(rounds) if rounds else 0)
            stds.append(_ps(rounds) if len(rounds) > 1 else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(DEPTHS, means, "-o", label=l, color=MODEL_COLORS[l], lw=2)
        ax.fill_between(DEPTHS, means - stds, means + stds, color=MODEL_COLORS[l], alpha=0.13)
    ax.axhline(30, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(1, 30.3, "plafond 30 tours", fontsize=8, color="black", alpha=0.7)
    ax.set_xticks(DEPTHS)
    ax.set_xlabel("Profondeur de craft (depth)")
    ax.set_ylabel("Tours moyens (± écart-type)")
    ax.set_title("Nombre de tours vs profondeur — bande = écart-type")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "2_tours_vs_depth.png", dpi=130)
    plt.close(fig)

    # ── 3) Types d'erreur (barres empilées, par épisode) ────────────────────
    # On n'affiche que les types réellement présents (other_error/generic_fail
    # sont à 0 depuis le raffinage de la taxonomie).
    active = [et for et in ETYPES
              if sum(z["error_counts"].get(et, 0) for l in labels for z in data[l]) > 0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottoms = np.zeros(len(labels))
    for et in active:
        vals = np.array([sum(z["error_counts"].get(et, 0) for z in data[l]) / len(data[l]) for l in labels])
        ax.bar(labels, vals, bottom=bottoms, label=et, color=ERROR_COLORS.get(et, "#333333"),
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    ax.set_ylabel("Erreurs par épisode (empilées par type)")
    ax.set_title("Répartition des types d'erreur, par modèle")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "3_types_erreur.png", dpi=130)
    plt.close(fig)

    # ── 4) Erreurs/épisode vs depth (barres groupées) ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xd = np.arange(len(DEPTHS))
    for j, l in enumerate(labels):
        vals = []
        for d in DEPTHS:
            r = [z for z in data[l] if depth.get(z["item_id"]) == d]
            vals.append(sum(z["n_errors"] for z in r) / len(r) if r else 0)
        ax.bar(xd + (j - 1.5) * width, vals, width, label=l, color=MODEL_COLORS[l])
    ax.set_xticks(xd)
    ax.set_xticklabels([f"depth {d}" for d in DEPTHS])
    ax.set_ylabel("Erreurs par épisode")
    ax.set_title("Erreurs par épisode vs profondeur de craft")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "4_erreurs_vs_depth.png", dpi=130)
    plt.close(fig)

    # ── 5) Récupération après erreur (% action changée, ≠ succès) ───────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    tot_att = {et: sum(len(z["recoveries"].get(et, [])) for l in labels for z in data[l]) for et in ETYPES}
    sel = [et for et in ETYPES if tot_att[et] >= 20]  # types avec assez de tentatives
    xe = np.arange(len(sel))
    for j, l in enumerate(labels):
        vals = []
        for et in sel:
            att = [b for z in data[l] for b in z["recoveries"].get(et, [])]
            vals.append(100 * sum(att) / len(att) if att else 0)
        ax.bar(xe + (j - 1.5) * width, vals, width, label=l, color=MODEL_COLORS[l])
    ax.set_xticks(xe)
    ax.set_xticklabels(sel, rotation=20, ha="right")
    ax.set_ylabel("Action changée au tour suivant (%)")
    ax.set_title("Récupération après erreur — % de CHANGEMENT d'action (≠ succès) — types n≥20")
    ax.set_ylim(0, 109)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "5_recuperation.png", dpi=130)
    plt.close(fig)

    # ── 6) Types d'erreur PAR DEPTH, PAR MODÈLE (grille 2×2, empilé) ─────────
    active = [et for et in ETYPES
              if sum(z["error_counts"].get(et, 0) for l in labels for z in data[l]) > 0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))  # y libre : magnitudes très différentes
    for ax, l in zip(axes.flat, labels):
        bottoms = np.zeros(len(DEPTHS))
        for et in active:
            vals = []
            for d in DEPTHS:
                r = [z for z in data[l] if depth.get(z["item_id"]) == d]
                vals.append(sum(z["error_counts"].get(et, 0) for z in r) / len(r) if r else 0)
            vals = np.array(vals)
            ax.bar([f"depth {d}" for d in DEPTHS], vals, bottom=bottoms, label=et,
                   color=ERROR_COLORS.get(et, "#333333"), edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_title(l, fontsize=11)
        ax.set_ylabel("erreurs / épisode")
        ax.grid(axis="y", alpha=0.3)
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Types d'erreur par profondeur de craft, par modèle", y=0.995, fontsize=13)
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=len(active), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outdir / "6_types_erreur_par_depth.png", dpi=130)
    plt.close(fig)

    print(f"\n[dashboard] 6 PNG sauvés dans {outdir}/")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true", help="imprime seulement les tableaux, pas de PNG")
    args = ap.parse_args()

    data, depth = load()
    labels = [l for l, _ in RUNS]

    out = REPO / "docs" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out / "tables.txt")  # duplique l'affichage des tableaux vers tables.txt

    print("=" * 72)
    print("① TABLEAU GLOBAL (100 items, test set TextCraft, 30 tours max)")
    print("=" * 72)
    print(f"{'Métrique':<20}" + "".join(f"{l:>13}" for l in labels))
    hrow("Pass@1 /100", [sum(1 for x in data[l] if x["reward"] > 0) for l in labels], "{}")
    hrow("Tours moyens", [sum(x["rounds"] for x in data[l]) / len(data[l]) for l in labels], "{:.1f}")
    hrow("Tours écart-type", [pstdev([x["rounds"] for x in data[l]]) for l in labels], "{:.1f}")
    hrow("Erreurs/épisode", [sum(x["n_errors"] for x in data[l]) / len(data[l]) for l in labels], "{:.1f}")
    hrow("Diversité actions", [sum(x["action_diversity"] for x in data[l]) / len(data[l]) for l in labels], "{:.2f}")
    hrow("Stagnation max moy", [sum(x["max_stagnation"] for x in data[l]) / len(data[l]) for l in labels], "{:.1f}")
    hrow("Récup. (chg action)", [recov_rate(data[l]) for l in labels], "{:.0f}%")

    print("\n" + "=" * 72)
    print("② ERREURS PAR TYPE — total absolu (et moyenne par épisode)")
    print("=" * 72)
    print(f"{'Type erreur':<16}" + "".join(f"{l:>14}" for l in labels))
    for et in ETYPES:
        cells = []
        for l in labels:
            tot = sum(x["error_counts"].get(et, 0) for x in data[l])
            cells.append(f"{tot} ({tot/len(data[l]):.1f})")
        print(f"{et:<16}" + "".join(f"{c:>14}" for c in cells))
    tot_cells = [f"{sum(x['n_errors'] for x in data[l])}" for l in labels]
    print(f"{'TOTAL':<16}" + "".join(f"{c:>14}" for c in tot_cells))

    print("\n" + "=" * 72)
    print("③ PASS@1 PAR DEPTH")
    print("=" * 72)
    print(f"{'Depth':<8}" + "".join(f"{l:>13}" for l in labels))
    for d in DEPTHS:
        cells = []
        for l in labels:
            r = [x for x in data[l] if depth.get(x["item_id"]) == d]
            cells.append(f"{sum(1 for x in r if x['reward']>0)}/{len(r)}")
        print(f"depth {d:<2}" + "".join(f"{c:>13}" for c in cells))

    print("\n" + "=" * 72)
    print("④ TOURS MOYENS ± ÉCART-TYPE, PAR DEPTH")
    print("=" * 72)
    print(f"{'Depth':<8}" + "".join(f"{l:>17}" for l in labels))
    for d in DEPTHS:
        cells = []
        for l in labels:
            rounds = [x["rounds"] for x in data[l] if depth.get(x["item_id"]) == d]
            cells.append(f"{sum(rounds)/len(rounds):.1f} ± {pstdev(rounds):.1f}" if rounds else "--")
        print(f"depth {d:<2}" + "".join(f"{c:>17}" for c in cells))

    print("\n" + "=" * 72)
    print("⑤ ERREURS/ÉPISODE PAR DEPTH")
    print("=" * 72)
    print(f"{'Depth':<8}" + "".join(f"{l:>13}" for l in labels))
    for d in DEPTHS:
        cells = []
        for l in labels:
            r = [x for x in data[l] if depth.get(x["item_id"]) == d]
            cells.append(f"{sum(x['n_errors'] for x in r)/len(r):.1f}" if r else "--")
        print(f"depth {d:<2}" + "".join(f"{c:>13}" for c in cells))

    print("\n" + "=" * 72)
    print("⑥ RÉCUPÉRATION PAR TYPE D'ERREUR — % action changée (n tentatives)")
    print("=" * 72)
    print(f"{'Type erreur':<16}" + "".join(f"{l:>16}" for l in labels))
    for et in ETYPES:
        cells = []
        for l in labels:
            attempts = [b for x in data[l] for b in x["recoveries"].get(et, [])]
            cells.append(f"{100*sum(attempts)/len(attempts):.0f}% ({len(attempts)})" if attempts else "-- (0)")
        print(f"{et:<16}" + "".join(f"{c:>16}" for c in cells))

    print("\n" + "=" * 72)
    print("⑦ SIGNIFICATION DES TYPES D'ERREUR")
    print("=" * 72)
    for et in ETYPES:
        print(f"  {et:<14} : {ERROR_DESCRIPTIONS.get(et, '')}")

    tee.close()  # fin de la capture : tables.txt contient les tableaux ①–⑦
    print(f"[dashboard] tableaux écrits dans {out / 'tables.txt'}")

    if not args.no_plots:
        make_plots(data, depth, out)


if __name__ == "__main__":
    main()
