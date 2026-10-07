"""Figure 4.2.2 : stabilité de la lignée full finetuning (exp23.1 → 23.4).
Source : logs TRL (reward, kl, entropy par pas ; pass@1 test périodique), extraits dans
fullft_lineage.json par le script inline de la session du 03/09. Trois panneaux, un axe chacun."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1]; out = sys.argv[2]
d = json.load(open(src))
tr = np.array(d["train"]); te = np.array(d["test"])
ep, rew, kl, ent = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
order = np.argsort(ep); ep, rew, kl, ent = ep[order], rew[order], kl[order], ent[order]

def rolling(x, y, width=1.0, fn=np.mean):
    """statistique glissante sur une fenêtre de `width` epochs, centrée."""
    xs = np.arange(x.min(), x.max(), 0.25); ys = []
    for c in xs:
        m = (x >= c - width / 2) & (x < c + width / 2)
        ys.append(fn(y[m]) if m.sum() >= 3 else np.nan)
    return xs, np.array(ys)

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
bounds = [s["start"] for s in d["segments"]][1:]

# (a) taux de réussite : reward train lissé + pass@1 test (même échelle 0-1)
ax = axes[0]
x, y = rolling(ep, rew, 1.0, np.mean)
ax.plot(x, y, color=BLUE, lw=1.4, label="récompense train (moy. 1 époque)")
ax.plot(te[:, 0], te[:, 1] / 100, ".", color=ORANGE, ms=3.5, alpha=0.85, label="pass@1 test")
ax.set_ylim(0, 1); ax.set_ylabel("taux de réussite"); ax.set_title("(a) récompense et pass@1", loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.2, borderaxespad=0.3)

# (b) KL vers la référence fixe (médiane glissante, échelle log)
ax = axes[1]
x, y = rolling(ep, kl, 1.0, np.median)
ax.plot(x, y, color=BLUE, lw=1.4)
ax.set_yscale("log"); ax.set_ylim(3e-4, 1.0)
ax.set_ylabel("KL vers la référence (méd. 1 époque)"); ax.set_title("(b) divergence KL", loc="left")

# (c) entropie de la politique
ax = axes[2]
x, y = rolling(ep, ent, 1.0, np.mean)
ax.plot(x, y, color=BLUE, lw=1.4)
ax.set_ylim(0, 1.6); ax.set_ylabel("entropie (moy. 1 époque)"); ax.set_title("(c) entropie", loc="left")

for ax in axes:
    for b in bounds:
        ax.axvline(b, color=MUTED, lw=0.8, ls=(0, (3, 3)), alpha=0.7)
    ax.set_xlabel("époques cumulées"); ax.set_xlim(0, ep.max())
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("ok", out)
