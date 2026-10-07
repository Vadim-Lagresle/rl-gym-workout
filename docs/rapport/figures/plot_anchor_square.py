"""Figure 4.2.4 : carré référence fixe / mobile × β. (a) pass@1 test et (b) KL sur les 16 premières
époques pour les cinq runs ; (c) pass@1 test des deux runs à référence mobile sur toute leur durée.
Source : logs TRL, extraits dans anchor_square.json. Couleur = β ; tirets = référence fixe ; plein = mobile."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
DASH = (0, (4, 2))
series = [  # clé, label, couleur, style
    ("mobile_b01",  "mobile, $\\beta=10^{-2}$ ($r=8$)",  BLUE,   "-"),
    ("fixe_b01",    "fixe, $\\beta=10^{-2}$ ($r=8$)",    BLUE,   DASH),
    ("mobile_b001", "mobile, $\\beta=10^{-3}$ ($r=8$)",  ORANGE, "-"),
    ("fixe_b001",   "fixe, $\\beta=10^{-3}$ ($r=16$)",   ORANGE, DASH),
    ("fixe_b0001",  "fixe, $\\beta=10^{-4}$ ($r=8$)",    AQUA,   DASH),
]

def rolling_median(tr, width=1.0, xmax=16.0):
    a = np.array(tr); x, y = a[:, 0], a[:, 1]
    xs = np.arange(0.5, min(x.max(), xmax), 0.1); ys = []
    for c in xs:
        m = (x >= c - width / 2) & (x < c + width / 2)
        ys.append(np.median(y[m]) if m.sum() >= 3 else np.nan)
    return xs, np.array(ys)

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)

ax = axes[0]
for key, label, col, ls in series:
    te = np.array(d[key]["test"]); m = te[:, 0] <= 16
    ax.plot(te[m, 0], te[m, 1], color=col, ls=ls, lw=1.4, marker=".", ms=3.5, label=label)
ax.set_xlim(0, 16); ax.set_ylim(0, 70); ax.set_xlabel("époques"); ax.set_ylabel("pass@1 test (%)")
ax.set_title("(a) pass@1, 16 premières époques", loc="left")

ax = axes[1]
for key, label, col, ls in series:
    x, y = rolling_median(d[key]["train"]); ax.plot(x, np.clip(y, 1e-4, 1e3), color=col, ls=ls, lw=1.4)
ax.axhspan(0.05, 0.1, color="#e6e5e1", zorder=0)
ax.set_yscale("log"); ax.set_ylim(3e-4, 1e3); ax.set_xlim(0, 16)
ax.set_xlabel("époques"); ax.set_ylabel("KL vers la référence (méd. 1 ép.)")
ax.set_title("(b) divergence KL", loc="left")

ax = axes[2]
for key, label, col, ls in series[:1] + series[2:3]:
    te = np.array(d[key]["test"]); ax.plot(te[:, 0], te[:, 1], color=col, ls=ls, lw=1.2, marker=".", ms=3, alpha=0.9)
ax.set_xlim(0, 115); ax.set_ylim(0, 70); ax.set_xlabel("époques"); ax.set_ylabel("pass@1 test (%)")
ax.set_title("(c) référence mobile, durée complète", loc="left")

for ax in axes:
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")

# chiffres de contrôle
for key, *_ in series:
    te = np.array(d[key]["test"]); tr = np.array(d[key]["train"])
    best = te[:, 1].max(); ib = te[:, 1].argmax()
    m = (tr[:, 0] >= 8) & (tr[:, 0] < 10)
    print(f"{key:12s} best {best:.0f} @ ep {te[ib,0]:.1f} | dernière éval {te[-1,1]:.0f} @ ep {te[-1,0]:.1f} | KL méd ep8-10 {np.median(tr[m,1]) if m.sum() else float('nan'):.3g}")
for key in ("mobile_b01", "mobile_b001"):
    te = np.array(d[key]["test"]); m = (te[:, 0] >= 49) & (te[:, 0] <= 58)
    print(f"{key}: pass@1 ep 49-58 : moy {te[m,1].mean():.1f} (min {te[m,1].min():.0f}, max {te[m,1].max():.0f})")
