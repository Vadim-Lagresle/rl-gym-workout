"""Figures 4.4 (heures GPU, tours d'environnement) et exploration des métriques sous-exploitées.
Source : metrics_all.json (pas d'entraînement + évaluations périodiques, six lignées)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, outdir = sys.argv[1], sys.argv[2]
D = json.load(open(src))
INK, MUTED = "#0b0b0b", "#52514e"
style = {"Horizon": ("#2a78d6", "-"), "Profondeur": ("#eb6834", "-"), "Budget": ("#1baf7a", "-"),
         "Contrôle-G16": ("#e87ba4", "-"), "Ancre-Mobile (G=8)": (MUTED, (0, (4, 2))), "Réplication-FT": (INK, (0, (1, 2)))}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
def deco(ax, xlabel="mises à jour (64 trajectoires chacune)"):
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True); ax.set_xlabel(xlabel)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def smooth(y, w=5):
    y = np.array(y, float)
    if len(y) < w: return y
    ys = np.convolve(y, np.ones(w)/w, mode="same"); ys[:w//2] = y[:w//2]; ys[-(w//2):] = y[-(w//2):]; return ys
def roll(x, y, width=93.0, fn=np.mean, xmax=None):
    x, y = np.array(x, float), np.array(y, float); m = ~np.isnan(y); x, y = x[m], y[m]
    grid = np.arange(width/2, (xmax or x.max()), width/4); out = []
    for c in grid:
        mm = (x >= c-width/2) & (x < c+width/2); out.append(fn(y[mm]) if mm.sum() >= 3 else np.nan)
    return grid, np.array(out)
def horizon_cap(step_abs, name):
    if name != "Horizon": return 30
    ep = step_abs/93.5; return 10 if ep < 15 else (20 if ep < 30 else 30)

# ---------- cumuls par lignée : heures GPU et tours d'environnement à chaque éval
cum = {}
for name, r in D.items():
    st = sorted(r["steps"], key=lambda s: s["step_abs"]); ev = sorted(r["evals"], key=lambda e: e["step_abs"])
    xs = np.array([s["step_abs"] for s in st]); tt = np.array([s["st"] or 0 for s in st])
    # temps : les logs n'écrivent pas forcément chaque pas → moyenne du step_time × pas écoulés
    hours, turns, prev, prev_r, ch, ct = [], [], 0.0, None, 0.0, 0.0
    for e in ev:
        m = (xs > prev) & (xs <= e["step_abs"]); mean_st = tt[m].mean() if m.sum() else (tt[xs <= e["step_abs"]].mean() if (xs <= e["step_abs"]).any() else 0)
        ch += mean_st * (e["step_abs"] - prev) / 3600
        r_here = min(e["rounds"] or 30, horizon_cap(e["step_abs"], name))
        ct += 64 * (e["step_abs"] - prev) * (prev_r if prev_r is not None else r_here)
        prev, prev_r = e["step_abs"], r_here
        hours.append(ch); turns.append(ct)
    cum[name] = {"x": [e["step_abs"] for e in ev], "p1": [e["p1"]*100 for e in ev], "hours": hours, "turns": turns,
                 "rounds": [e["rounds"] for e in ev], "rounds_d": [e["rounds_d"] for e in ev]}

for key, xlabel, fname, scale in [("hours", "heures GPU cumulées", "fig_pass1_gpuhours", 1.0), ("turns", "tours d'environnement cumulés (millions, estimés)", "fig_pass1_envturns", 1e-6)]:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for name, c in cum.items():
        col, ls = style[name]; ax.plot(np.array(c[key])*scale, smooth(c["p1"]), color=col, ls=ls, lw=1.4, label=name)
    ax.set_ylim(0, 90); ax.set_ylabel("pass@1 test (%), lissé sur 5 évaluations"); deco(ax, xlabel); ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.savefig(f"{outdir}/{fname}.pdf"); fig.savefig(f"{outdir}/{fname}.png", dpi=200)

# ---------- exploration des métriques
# (a) tours moyens par épisode (test) + (b) tours par profondeur au plateau
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
for name, c in cum.items():
    col, ls = style[name]; axes[0].plot(c["x"], smooth(c["rounds"]), color=col, ls=ls, lw=1.3, label=name)
axes[0].set_ylim(0, 30); axes[0].set_ylabel("tours par épisode (test, moy.)"); axes[0].set_title("(a) longueur des épisodes de test", loc="left"); deco(axes[0]); axes[0].set_xlim(0, 8000)
names = [n for n in cum if n != "Contrôle-G16"]; w = 0.15
for i, name in enumerate(names):
    rd = np.array([r for r in cum[name]["rounds_d"][-10:] if None not in r]); mean = rd.mean(axis=0)
    axes[1].bar(np.arange(4) + (i-2)*w, mean, width=w, color=style[name][0], label=name, alpha=0.9 if name != "Réplication-FT" else 0.6)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["d1", "d2", "d3", "d4"]); axes[1].set_ylabel("tours par épisode (10 dernières évals)")
axes[1].set_title("(b) tours par profondeur au plateau", loc="left"); deco(axes[1], "profondeur"); axes[1].legend(loc="upper left", frameon=False, fontsize=6.5)
fig.savefig(f"{outdir}/explo_rounds.pdf"); fig.savefig(f"{outdir}/explo_rounds.png", dpi=200)

# (c) récompense train et écart-type intra-groupe ; (d) fraction de groupes sans gradient
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
for name, r in D.items():
    col, ls = style[name]; st = r["steps"]; x = [s["step_abs"] for s in st]
    for ax, key in zip(axes, ("reward", "reward_std", "fzs")):
        gx, gy = roll(x, [s[key] if s[key] is not None else np.nan for s in st], xmax=8000); ax.plot(gx, gy, color=col, ls=ls, lw=1.3, label=name if key == "reward" else None)
axes[0].set_title("(a) récompense train (moy.)", loc="left"); axes[0].set_ylim(0, 1)
axes[1].set_title("(b) écart-type intra-groupe", loc="left"); axes[1].set_ylim(0, 0.5)
axes[2].set_title("(c) groupes sans gradient (fraction)", loc="left"); axes[2].set_ylim(0, 1)
for ax in axes: deco(ax); ax.set_xlim(0, 8000)
axes[0].legend(loc="lower right", frameon=False, fontsize=6.5)
fig.savefig(f"{outdir}/explo_reward_groups.pdf"); fig.savefig(f"{outdir}/explo_reward_groups.png", dpi=200)

# (e) longueur des séquences d'entraînement ; (f) norme du gradient
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
for name, r in D.items():
    col, ls = style[name]; st = sorted(r["steps"], key=lambda s: s["step_abs"]); x = [s["step_abs"] for s in st]
    gx, gy = roll(x, [s["len"] if s["len"] is not None else np.nan for s in st], xmax=8000); axes[0].plot(gx, gy, color=col, ls=ls, lw=1.3, label=name)
    gx, gy = roll(x, [s["gn"] if s["gn"] is not None else np.nan for s in st], fn=np.median, xmax=8000); axes[1].plot(gx, gy, color=col, ls=ls, lw=1.3)
axes[0].set_title("(a) longueur des séquences d'entraînement", loc="left"); axes[0].set_ylabel("tokens (prompt + trajectoire)"); axes[0].set_ylim(0, 2500)
axes[1].set_title("(b) norme du gradient (médiane)", loc="left"); axes[1].set_yscale("log")
for ax in axes: deco(ax); ax.set_xlim(0, 8000)
axes[0].legend(loc="upper right", frameon=False, fontsize=6.5)
fig.savefig(f"{outdir}/explo_length_grad.pdf"); fig.savefig(f"{outdir}/explo_length_grad.png", dpi=200)
print("ok")
