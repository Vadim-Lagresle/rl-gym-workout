"""Figures 4.3.2 : (1) pass@1 test et entropie vs mises à jour pour les trois curriculums et les deux
références ; (2) pass@1 par profondeur ; (3) erreurs de format vs erreurs de planification par épisode.
Source : logs TRL extraits dans curricula.json (évals périodiques + séries d'entraînement)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, outdir = sys.argv[1], sys.argv[2]
d = json.load(open(src))
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# lignées : (clé, offset en époques, ...) ; x final en mises à jour = epoch × steps/epoch
lineages = {
    "Horizon":        ([("horizon", 0.0)], BLUE, "-"),
    "Profondeur":     ([("prof_auto", 0.0), ("prof_auto_reprise", 43.2)], ORANGE, "-"),
    "Budget":         ([("budget", 0.0), ("budget_reprise", 63.7)], AQUA, "-"),
    "Contrôle-G16":   ([("controle_g16", 0.0)], MAGENTA, "-"),
    "Ancre-Mobile (G=8)": ([("ancre_g8", 0.0), ("ancre_g8_reprise", 102.5)], MUTED, (0, (4, 2))),
}
def evals(lin):
    xs, p1, dd = [], [], []
    for key, off in lin:
        spe = d[key]["spe"]
        for e in d[key]["evals"]:
            xs.append((e["epoch"] + off) * spe); p1.append(e["p1"]); dd.append(e["d"])
    o = np.argsort(xs); return np.array(xs)[o], np.array(p1)[o], np.array(dd)[o]
def train(lin, col):
    xs, ys = [], []
    for key, off in lin:
        spe = d[key]["spe"]; a = np.array(d[key]["train"])
        xs += list((a[:, 0] + off) * spe); ys += list(a[:, col])
    xs, ys = np.array(xs), np.array(ys); o = np.argsort(xs); xs, ys = xs[o], ys[o]
    grid = np.arange(50, xs.max(), 25); out = []
    for c in grid:
        m = (xs >= c - 47) & (xs < c + 47); out.append(ys[m].mean() if m.sum() >= 3 else np.nan)
    return grid, np.array(out)
def errs(lin, keys):
    xs, ys = [], []
    for key, off in lin:
        spe = d[key]["spe"]
        for e in d[key]["evals"]:
            xs.append((e["epoch"] + off) * spe); ys.append(sum(e["err"].get(k, 0) for k in keys))
    o = np.argsort(xs); return np.array(xs)[o], np.array(ys)[o]
def smooth(y, w=5):
    if len(y) < w: return y
    k = np.ones(w) / w; ys = np.convolve(y, k, mode="same"); ys[:w//2] = y[:w//2]; ys[-(w//2):] = y[-(w//2):]; return ys
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
def deco(ax):
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("mises à jour (64 trajectoires chacune)")

# ---- Figure 1 : pass@1 et entropie
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
for name, (lin, col, ls) in lineages.items():
    x, p1, _ = evals(lin); axes[0].plot(x, smooth(p1), color=col, ls=ls, lw=1.4, label=name)
    gx, ent = train(lin, 3); axes[1].plot(gx, ent, color=col, ls=ls, lw=1.4)
axes[0].set_ylim(0, 90); axes[0].set_ylabel("pass@1 test (%), lissé sur 5 évaluations"); axes[0].set_title("(a) pass@1 test", loc="left")
axes[1].set_ylim(0, 1.3); axes[1].set_ylabel("entropie (moy. 1 époque)"); axes[1].set_title("(b) entropie de la politique", loc="left")
for ax in axes: deco(ax); ax.set_xlim(0, 8000)
axes[0].legend(loc="lower right", frameon=False, fontsize=7)
fig.savefig(f"{outdir}/fig_curricula_pass1_entropy.pdf"); fig.savefig(f"{outdir}/fig_curricula_pass1_entropy.png", dpi=200)

# ---- Figure 2 : pass@1 par profondeur (d1, d2, d3)
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), constrained_layout=True, sharey=True)
for name, (lin, col, ls) in lineages.items():
    x, _, dd = evals(lin)
    for i in range(3): axes[i].plot(x, smooth(dd[:, i]), color=col, ls=ls, lw=1.3, label=name if i == 0 else None)
for i, ax in enumerate(axes):
    deco(ax); ax.set_xlim(0, 8000); ax.set_ylim(0, 105); ax.set_title(f"({'abc'[i]}) profondeur {i+1}", loc="left")
axes[0].set_ylabel("pass@1 test (%), lissé sur 5 évaluations"); axes[0].legend(loc="lower right", frameon=False, fontsize=6.5)
fig.savefig(f"{outdir}/fig_curricula_depths.pdf"); fig.savefig(f"{outdir}/fig_curricula_depths.png", dpi=200)

# ---- Figure 3 : erreurs de format vs de planification
FORMAT = ["format_error", "multi_action", "wrong_format"]; PLAN = ["recipe_wrong", "missing_items", "item_not_found"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), constrained_layout=True)
for name, (lin, col, ls) in lineages.items():
    x, yf = errs(lin, FORMAT); axes[0].plot(x, smooth(yf), color=col, ls=ls, lw=1.3, label=name)
    x, yp = errs(lin, PLAN); axes[1].plot(x, smooth(yp), color=col, ls=ls, lw=1.3)
axes[0].set_title("(a) erreurs de format par épisode", loc="left"); axes[1].set_title("(b) erreurs de planification par épisode", loc="left")
for ax in axes: deco(ax); ax.set_xlim(0, 8000); ax.set_ylim(0, 14)
axes[0].set_ylabel("erreurs par épisode (test)"); axes[0].legend(loc="upper right", frameon=False, fontsize=7)
fig.savefig(f"{outdir}/fig_curricula_errors.pdf"); fig.savefig(f"{outdir}/fig_curricula_errors.png", dpi=200)
print("ok")
