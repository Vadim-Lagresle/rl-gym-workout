"""Contrôle-G16 + borne k3 (exp43) contre Horizon (exp32) : même recette ReLoRA r8, G=16, 4 tâches/pas,
ancre /4 ép. Deux deltas opposés au contrôle mort : borner la KL dans la perte, ou un curriculum d'horizon.
(a) pass@1 test lissé 3 évals ; (b) KL médiane par époque, log ; (c) entropie moyenne par époque."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {"exp43_g16_klclamp10": ("#eb6834", "-", "Control-G16 + KL bound (k3 ≤ 10), no curriculum"),
        "exp32_horizon":       ("#a3405c", "-", "Horizon curriculum (turns 10/20/30 at ep. 0/15/30)")}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})"); eval_re = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100")
def load(run):
    ep, ent, kl, ev = [], [], [], {}
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d and "entropy" in d: ep.append(float(d["epoch"])); ent.append(float(d["entropy"])); kl.append(float(d["kl"]))
        m = eval_re.search(line)
        if m: ev[int(m.group(1))] = int(m.group(2))
    ep, ent, kl = map(np.array, (ep, ent, kl)); spe = len(ep) / ep[-1]
    es = np.array(sorted(ev)); pv = np.array([ev[s] for s in es]); return ep, ent, kl, es / spe, pv
def win(ep, y, med, step=0.5):
    xs, ys = [], []
    for c in np.arange(0.5, ep[-1] + 1e-9, step):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ys.append(np.median(y[m]) if med else y[m].mean())
    return np.array(xs), np.array(ys)
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), constrained_layout=True)
last = {}
for run, (col, ls, label) in RUNS.items():
    ep, ent, kl, eep, pv = load(run); last[run] = ep[-1]
    pr = np.array([pv[max(0, i - 2):i + 1].mean() for i in range(len(pv))])
    axes[0].plot(eep, pr, color=col, ls=ls, lw=1.5, label=label)
    xk, yk = win(ep, kl, True); axes[1].plot(xk, np.clip(yk, 3e-4, 1e3), color=col, ls=ls, lw=1.5)
    xe, ye = win(ep, ent, False); axes[2].plot(xe, ye, color=col, ls=ls, lw=1.5)
for e in (15, 30):
    for ax in axes: ax.axvline(e, color="#a3405c", lw=0.7, alpha=0.4)
pass  # exp43 terminé (80 ép., 23/09) : plus de marqueur de fin de run
pass
axes[0].set_ylim(0, 85); axes[0].set_ylabel("pass@1 test (mean of 3 evals)"); axes[0].set_title("(a) test performance", loc="left")
axes[1].set_yscale("log"); axes[1].set_ylim(3e-4, 10); axes[1].set_ylabel("KL to the reference (median, 1 ep.)"); axes[1].set_title("(b) KL divergence", loc="left")
axes[2].set_ylim(0, 1.4); axes[2].set_ylabel("policy entropy (mean, 1 ep.)"); axes[2].set_title("(c) entropy", loc="left")
for ax in axes:
    ax.set_xlim(0, 80); ax.set_xlabel("epochs"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.09))
fig.text(0.5, -0.17, "Same ReLoRA recipe (r=8, G=16, 4 tasks/step, anchor every 4 epochs). Thin vertical lines: Horizon stage changes. One epoch ≈ 2× the compute of a G=8 epoch for both.", ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_bound_vs_horizon"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok", {k: round(v, 1) for k, v in last.items()})
