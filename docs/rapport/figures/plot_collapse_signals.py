"""Figure 4.2.5 : signaux qui précèdent un collapse. Entropie (a) et KL (b), moyennes/médianes glissantes
sur une époque, seize premières époques, pour trois familles : runs stables, collapses à référence fixe,
collapses à G=16 sans curriculum. Source : logs TRL extraits dans collapse_signals.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
style = {"stable": (BLUE, "-", "stables (6 runs)"), "fixe": (ORANGE, (0, (4, 2)), "collapse, référence fixe (6 runs)"),
         "g16": (AQUA, "-", "collapse, $G=16$ sans curriculum (3 runs)")}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
seen = set()
for name, r in d.items():
    col, ls, lab = style[r["group"]]
    x = np.array(r["x"]); ent = np.array(r["ent"], dtype=float); kl = np.array(r["kl"], dtype=float)
    label = lab if r["group"] not in seen else None; seen.add(r["group"])
    axes[0].plot(x, ent, color=col, ls=ls, lw=1.1, alpha=0.9, label=label)
    axes[1].plot(x, np.clip(kl, 1e-4, 1e3), color=col, ls=ls, lw=1.1, alpha=0.9)
axes[0].set_ylim(0, 1.6); axes[0].set_ylabel("entropie (moy. 1 époque)"); axes[0].set_title("(a) entropie de la politique", loc="left")
axes[1].set_yscale("log"); axes[1].set_ylim(3e-4, 1e3); axes[1].set_ylabel("KL vers la référence (méd. 1 ép.)"); axes[1].set_title("(b) divergence KL", loc="left")
for ax in axes:
    ax.set_xlim(0, 16); ax.set_xlabel("époques"); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(loc="lower left", frameon=False, fontsize=7)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("ok")
