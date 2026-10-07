"""Annexe F : erreurs par type (six classes de la taxonomie) au fil des évaluations, par régime.
Source : curricula.json (évaluations périodiques, clés eval/err_*_per_ep)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
INK, MUTED = "#0b0b0b", "#52514e"
lineages = {"Horizon": ([("horizon", 0.0)], "#2a78d6", "-"), "Profondeur": ([("prof_auto", 0.0), ("prof_auto_reprise", 43.2)], "#eb6834", "-"),
            "Budget": ([("budget", 0.0), ("budget_reprise", 63.7)], "#1baf7a", "-"), "Contrôle-G16": ([("controle_g16", 0.0)], "#e87ba4", "-"),
            "Ancre-Mobile (G=8)": ([("ancre_g8", 0.0), ("ancre_g8_reprise", 102.5)], MUTED, (0, (4, 2)))}
classes = [("format_error", "action mal formée"), ("multi_action", "plusieurs actions par tour"), ("wrong_format", "nom d'objet mal écrit"),
           ("recipe_wrong", "recette invalide"), ("missing_items", "ingrédients insuffisants"), ("item_not_found", "objet introuvable")]
def series(lin, key):
    xs, ys = [], []
    for k, off in lin:
        spe = d[k]["spe"]
        for e in d[k]["evals"]: xs.append((e["epoch"] + off) * spe); ys.append(e["err"].get(key, 0.0))
    o = np.argsort(xs); return np.array(xs)[o], np.array(ys)[o]
def smooth(y, w=5):
    if len(y) < w: return y
    ys = np.convolve(y, np.ones(w)/w, mode="same"); ys[:w//2] = y[:w//2]; ys[-(w//2):] = y[-(w//2):]; return ys
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5})
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), constrained_layout=True)
for ax, (key, label) in zip(axes.flat, classes):
    for name, (lin, col, ls) in lineages.items():
        x, y = series(lin, key); ax.plot(x, smooth(y), color=col, ls=ls, lw=1.2, label=name)
    ax.set_title(label, loc="left"); ax.set_xlim(0, 8000); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[1]: ax.set_xlabel("mises à jour")
for ax in axes[:, 0]: ax.set_ylabel("erreurs par épisode (test)")
axes[0, 0].legend(loc="upper right", frameon=False, fontsize=6.5)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
# chiffres : début (1re éval) → plateau (10 dernières) par classe et régime
for name, (lin, *_ ) in lineages.items():
    parts = []
    for key, label in classes:
        x, y = series(lin, key); parts.append(f"{key} {y[0]:.1f}→{y[-10:].mean():.1f}")
    print(f"{name:20s} " + " | ".join(parts))
