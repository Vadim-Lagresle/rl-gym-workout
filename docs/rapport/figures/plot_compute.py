"""Figure 4.4 : pass@1 test en fonction des tokens traités cumulés (prompt + trajectoire), par lignée.
Source : compute.json (num_tokens cumulé de TRL à chaque évaluation)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
INK, MUTED = "#0b0b0b", "#52514e"
style = {"Horizon": ("#2a78d6", "-"), "Profondeur": ("#eb6834", "-"), "Budget": ("#1baf7a", "-"),
         "Contrôle-G16": ("#e87ba4", "-"), "Ancre-Mobile (G=8)": (MUTED, (0, (4, 2))), "Réplication-FT": (INK, (0, (1, 2)))}
def smooth(y, w=5):
    y = np.array(y, float)
    if len(y) < w: return y
    ys = np.convolve(y, np.ones(w) / w, mode="same"); ys[:w//2] = y[:w//2]; ys[-(w//2):] = y[-(w//2):]; return ys
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
for name, r in d.items():
    a = np.array(r["evals_tok"]); col, ls = style[name]
    ax.plot(a[:, 0] / 1e9, smooth(a[:, 1]), color=col, ls=ls, lw=1.4, label=name)
ax.set_xlabel("tokens traités cumulés (milliards, prompt et trajectoire)"); ax.set_ylabel("pass@1 test (%), lissé sur 5 évaluations")
ax.set_xlim(0, 0.9); ax.set_ylim(0, 90); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("ok")
