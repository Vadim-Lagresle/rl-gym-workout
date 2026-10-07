"""Figure 4.2.3 : dérive KL vers la référence, full finetuning vs LoRA à référence fixe (12 premières époques).
Source : logs TRL ('kl', 'epoch'), extraits dans kl_grid.json. Médiane glissante sur une époque, échelle log."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))

def rolling_median(pts, width=1.0, xmax=12.0):
    a = np.array(pts); x, y = a[:, 0], a[:, 1]
    xs = np.arange(0.5, min(x.max(), xmax), 0.1); ys = []
    for c in xs:
        m = (x >= c - width / 2) & (x < c + width / 2)
        ys.append(np.median(y[m]) if m.sum() >= 3 else np.nan)
    return xs, np.array(ys)

INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
series = [  # (clé, label, couleur, style)
    ("fullft",        "full finetuning, $\\beta=10^{-3}$",          INK,    "-"),
    ("r8_b01_fixe",   "LoRA $r=8$, $\\beta=10^{-2}$",                BLUE,   "-"),
    ("r16_b01",       "LoRA $r=16$, $\\beta=10^{-2}$",               ORANGE, "-"),
    ("r16_b001",      "LoRA $r=16$, $\\beta=10^{-3}$",               ORANGE, (0, (4, 2))),
    ("r64_b01",       "LoRA $r=64$, $\\beta=10^{-2}$",               AQUA,   "-"),
    ("r64_b001",      "LoRA $r=64$, $\\beta=10^{-3}$",               AQUA,   (0, (4, 2))),
]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
ax.axhspan(0.05, 0.1, color="#e6e5e1", zorder=0)
ax.text(0.15, 0.07, "niveau franchi par tous les runs LoRA à référence fixe, 1,5 à 4 époques avant divergence", fontsize=6.5, color=MUTED, va="center")
for key, label, col, ls in series:
    x, y = rolling_median(d[key]); y = np.clip(y, 1e-4, 1e3)
    ax.plot(x, y, color=col, ls=ls, lw=1.4, label=label)
ax.set_yscale("log"); ax.set_ylim(5e-4, 1e3); ax.set_xlim(0, 12)
ax.set_xlabel("époques"); ax.set_ylabel("KL vers la référence (méd. sur 1 époque)")
ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)

# chiffres de contrôle pour le texte
for key, *_ in series + [("r8_b01_mobile",)]:
    a = np.array(d[key]); 
    def med(lo, hi):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi); return np.median(a[m, 1]) if m.sum() else float("nan")
    def ent(lo, hi):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi); return np.median(a[m, 2]) if m.sum() else float("nan")
    print(f"{key:14s} KL méd ep2-4 {med(2,4):.4f} | ep4-6 {med(4,6):.4f} | ep6-8 {med(6,8):.3g} | ep8-10 {med(8,10):.3g} || entropie ep4-6 {ent(4,6):.2f}")
