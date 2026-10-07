"""Entropie des curriculums face au contrôle G=16, même recette (ReLoRA r8, G=16, 4 tâches/pas, ancre /4 ép.).
(a) 0-80 époques ; (b) zoom 0-16. Traits verticaux au-dessus : changements de palier (couleur du run).
Depth auto : d≤2 à 2,14, d≤3 à 12,14, d≤4 à 22,14. Depth fixe (exp33, mort purge ép. 22) : d≤2 à 12, d≤3 à 25.
Budget : 512 tokens à 15, 1024 à 35. Horizon : 20 tours à 15, 30 à 30."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {  # run: (couleur, style, label, paliers)
    "exp36_n16":          ("#eb6834", (0, (4, 2)), "Control-G16, no curriculum", []),
    "exp32_horizon":      ("#a3405c", "-", "Horizon (turns 10/20/30 at ep. 0/15/30)", [15, 30]),
    "exp35_budget1024":   ("#b8860b", "-", "Budget (256/512/1024 tokens at ep. 0/15/35)", [15, 35]),
    "exp33.1_depth_auto": ("#7b52ab", "-", "Depth, auto stages (d≤2 at 2.1, d≤3 at 12.1, d≤4 at 22.1)", [2.14, 12.14, 22.14]),
    "exp33_depth":        ("#c4a3e8", (0, (2, 2)), "Depth, fixed stages (d≤2 at 12, d≤3 at 25; killed ep. 22)", [12]),
}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
def load(run):
    ep, ent, kl = [], [], []
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d and "entropy" in d: ep.append(float(d["epoch"])); ent.append(float(d["entropy"])); kl.append(float(d["kl"]))
    return tuple(map(np.array, (ep, ent, kl)))
def series(ep, ent, kl, xmax, step):
    xs, ye, yk = [], [], []
    for c in np.arange(0.25, min(xmax, ep[-1]) + 1e-9, step):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ye.append(ent[m].mean()); yk.append(np.median(kl[m]))
    xs, ye, yk = map(np.array, (xs, ye, yk)); cut = np.where(yk > 1)[0]; i = (cut[0] + 1) if len(cut) else len(xs)
    return xs[:i], ye[:i], bool(len(cut))
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
data = {run: load(run) for run in RUNS}
for k_run, (run, (col, ls, label, stages)) in enumerate(RUNS.items()):
    ep, ent, kl = data[run]
    for ax, xmax, step in ((axes[0], 80, 0.5), (axes[1], 16, 0.25)):
        xs, ye, collapsed = series(ep, ent, kl, xmax, step)
        ax.plot(xs, ye, color=col, ls=ls, lw=1.5, label=label if ax is axes[0] else None)
        if collapsed and xs[-1] <= xmax: ax.plot(xs[-1], ye[-1], "x", ms=7, mew=1.8, color=col)
        for s_ in stages:
            if s_ <= xmax: ax.plot([s_], [1.0 + 0.03 * k_run], marker="|", ms=5, color=col, transform=ax.get_xaxis_transform(), clip_on=False)
axes[0].set_xlim(0, 80); axes[0].set_title("(a) full runs", loc="left", pad=16)
axes[1].set_xlim(0, 16); axes[1].set_title("(b) first 16 epochs", loc="left", pad=16)
for ax in axes:
    ax.set_ylim(0, 1.6); ax.set_xlabel("epochs"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].set_ylabel("policy entropy (mean, 1 epoch)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.2, "Same ReLoRA recipe, G=16. Ticks above the panels: stage changes, one row per run. Cross: collapse (KL median > 1).", ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_entropy_curricula"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
# chiffres pour le rapport : entropie aux paliers de Depth auto et aux ép. 1, 2, 3
ep, ent, kl = data["exp33.1_depth_auto"]
for e in (0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 12, 13, 22, 23):
    m = (ep >= e - 0.5) & (ep < e + 0.5); print(f"depth-auto ep {e:5.1f} : entropie {ent[m].mean():.2f}")
ep, ent, kl = data["exp33_depth"]
for e in (1, 2, 4, 6, 8, 10, 12, 13, 14, 16, 20):
    m = (ep >= e - 0.5) & (ep < e + 0.5); print(f"depth-fixed ep {e:5.1f} : entropie {ent[m].mean():.2f}")
