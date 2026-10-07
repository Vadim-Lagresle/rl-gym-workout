"""Entropie : Contrôle-G16 (exp36, sans curriculum) contre Horizon (exp32), même recette ReLoRA r8, G=16,
4 tâches/pas, ancre /4 ép. Seul delta : le curriculum d'horizon (10/20/30 tours aux ép. 0/15/30).
Moyenne glissante sur 1 époque. Croix = collapse du contrôle (KL médiane > 1)."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {"exp36_n16": ("#eb6834", (0, (4, 2)), "Control-G16: same recipe, no curriculum"),
        "exp32_horizon": ("#a3405c", "-", "Horizon: turns 10 → 20 → 30 at epochs 0 / 15 / 30")}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
def load(run):
    ep, ent, kl = [], [], []
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d and "entropy" in d: ep.append(float(d["epoch"])); ent.append(float(d["entropy"])); kl.append(float(d["kl"]))
    return map(np.array, (ep, ent, kl))
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, ax = plt.subplots(figsize=(7.4, 3.0), constrained_layout=True)
for run, (col, ls, label) in RUNS.items():
    ep, ent, kl = load(run); xs, ye, yk = [], [], []
    for c in np.arange(0.5, ep[-1] + 1e-9, 0.5):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ye.append(ent[m].mean()); yk.append(np.median(kl[m]))
    xs, ye, yk = map(np.array, (xs, ye, yk)); cut = np.where(yk > 1)[0]; i = (cut[0] + 1) if len(cut) else len(xs)
    ax.plot(xs[:i], ye[:i], color=col, ls=ls, lw=1.8, label=label)
    if len(cut): ax.plot(xs[i-1], ye[i-1], "x", ms=8, mew=2, color=col); ax.annotate("collapse, epoch 10", (xs[i-1], ye[i-1]), xytext=(8, 6), textcoords="offset points", fontsize=7.5, color=col)
for e, t in ((15, "20 turns"), (30, "30 turns")):
    ax.axvline(e, color="#a3405c", lw=0.8, alpha=0.5); ax.text(e + 0.6, 1.5, t, fontsize=7, color="#a3405c")
ax.text(0.6, 1.5, "10 turns", fontsize=7, color="#a3405c")
ax.set_xlim(0, 80); ax.set_ylim(0, 1.6); ax.set_xlabel("epochs"); ax.set_ylabel("policy entropy (mean, 1 epoch)")
ax.set_title("Same recipe, one delta: the horizon curriculum", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=8)
out = "docs/rapport/figures/fig_entropy_control_vs_horizon"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
