"""Longueur d'épisode, tours par épisode et tokens par tour avant le collapse (15/09/2026).
Données : collapse_lengths.json (extract_lengths.py). Moyennes par demi-époque.
exp36/exp40 = G=16 sans curriculum (collapse ép. 10 / 15) ; exp41/exp25 = G=8 (stables)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("docs/rapport/figures/collapse_lengths.json"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {  # couleurs validées (validate_palette.js, light) ; exp25 en tirets = référence historique
    "exp40_g16_8tasks_anchor8": ("#eb6834", "-",  "exp40 · G=16, 8 tâches, ancre 368 pas"),
    "exp36_n16":                ("#a3405c", "-",  "exp36 · G=16, 4 tâches, ancre 372 pas"),
    "exp41_g8_8tasks_anchor8":  ("#2a78d6", "-",  "exp41 · G=8, 8 tâches, ancre 368 pas"),
    "exp25_r8_anchor4ep":       ("#0e9f8a", (0, (4, 2)), "exp25 · G=8, 8 tâches, ancre 184 pas"),
}
COLLAPSE = {"exp36_n16": 10.0, "exp40_g16_8tasks_anchor8": 15.1}  # 1re explosion soutenue (log)
XMAX, BIN = 20, 0.5
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), constrained_layout=True)
for k_run, (run, (col, ls, label)) in enumerate(STYLE.items()):
    r = np.array(d[run]["rows"]); ep, ln, tr = r[:, 1], r[:, 2], r[:, 3]
    edges = np.arange(0, XMAX + BIN, BIN); x, y_len, y_tr = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ep >= a) & (ep < b)
        if m.sum() >= 3:
            x.append((a + b) / 2); y_len.append(ln[m].mean()); y_tr.append(tr[m].mean())
    x, y_len, y_tr = map(np.array, (x, y_len, y_tr))
    for ax, y in zip(axes, (y_len, y_tr, y_len / y_tr)):
        ax.plot(x, y, color=col, ls=ls, lw=1.4, label=label)
        # ancres : petits traits en haut ; collapse : point plein
        for s in d[run]["anchors"]:
            e = s / d[run]["spe"]
            if e <= XMAX: ax.plot([e], [1.0 + 0.025 * k_run], marker="|", ms=5, color=col,
                                  transform=ax.get_xaxis_transform(), clip_on=False)
        if run in COLLAPSE:
            i = np.argmin(np.abs(x - COLLAPSE[run])); ax.plot(x[i], y[i], "o", ms=5, color=col)
axes[0].set_yscale("log"); axes[0].set_ylim(150, 12000); axes[0].set_ylabel("tokens par épisode (moy.)")
axes[0].set_title("(a) longueur d'épisode", loc="left", pad=16)
axes[1].set_ylim(0, 31); axes[1].set_ylabel("tours par épisode (moy.)"); axes[1].set_title("(b) tours (cap 30)", loc="left", pad=16)
axes[2].set_yscale("log"); axes[2].set_ylim(7, 1000); axes[2].set_ylabel("tokens par tour (≈ len / tours)")
axes[2].set_title("(c) tokens par tour", loc="left", pad=16)
for ax in axes:
    ax.set_xlim(0, XMAX); ax.set_xlabel("époque"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
fig.text(0.5, -0.17, "traits au-dessus des panneaux : ré-ancrages (une rangée par run) · point plein : première explosion de KL",
         ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_collapse_lengths"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
