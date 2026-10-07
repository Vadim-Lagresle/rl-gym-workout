"""Un seul delta : référence KL fixe (exp30) contre référence mobile ReLoRA /4 ép. (exp25). Même recette
(LoRA r8, α32, LR 3e-6, β0,01, G=8, 8 tâches/pas). (a) KL médiane par demi-époque, log ; (b) pass@1 test
lissé sur 3 évaluations. La dérive KL du run fixe précède son collapse de pass@1."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {"exp30_r8_fixedanchor": ("#eb6834", (0, (4, 2)), "fixed reference (LoRA, base model)"),
        "exp25_r8_anchor4ep":   ("#0e9f8a", "-",         "moving reference (ReLoRA, every 4 epochs)")}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})"); eval_re = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100")
anch_re = re.compile(r"RÉ-ANCRAGE #\d+ @ step (\d+)")
def load(run):
    ep, kl, ev, anchors = [], [], {}, []
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d: ep.append(float(d["epoch"])); kl.append(float(d["kl"]))
        m = eval_re.search(line)
        if m: ev[int(m.group(1))] = int(m.group(2))
        m = anch_re.search(line)
        if m: anchors.append(int(m.group(1)))
    ep, kl = np.array(ep), np.array(kl); spe = len(ep) / ep[-1] if len(ep) else 46
    es = np.array(sorted(ev)); pv = np.array([ev[s] for s in es])
    return ep, kl, es / spe, pv, np.array(sorted(set(anchors))) / spe
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
XMAX = 20
for run, (col, ls, label) in RUNS.items():
    ep, kl, eep, pv, anch = load(run)
    xs, ys = [], []
    for c in np.arange(0.5, min(XMAX, ep[-1]) + 1e-9, 0.5):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ys.append(np.median(kl[m]))
    xs, ys = np.array(xs), np.array(ys)
    axes[0].plot(xs, np.clip(ys, 3e-4, 1e3), color=col, ls=ls, lw=1.6, label=label)
    k = eep <= XMAX; pr = np.array([pv[max(0, i - 2):i + 1].mean() for i in range(len(pv))])
    axes[1].plot(eep[k], pr[k], color=col, ls=ls, lw=1.6, marker=".", ms=3)
    if "fixed" in run:
        axes[0].plot(xs[-1], min(ys[-1], 1e3), "x", ms=7, mew=1.8, color=col); axes[1].plot(eep[k][-1], pr[k][-1], "x", ms=7, mew=1.8, color=col)
        axes[1].annotate("collapse", (eep[k][-1], pr[k][-1]), xytext=(6, -4), textcoords="offset points", fontsize=7, color=col)
    else:
        axes[1].annotate("continues to 73 at epoch 111", (eep[k][-1], pr[k][-1]), xytext=(-118, 12), textcoords="offset points", fontsize=7, color=col)
        for a in anch[anch <= XMAX]:
            for ax in axes: ax.axvline(a, color=col, lw=0.6, alpha=0.35)
axes[0].set_yscale("log"); axes[0].set_ylim(3e-4, 1e3); axes[0].set_ylabel("KL to the reference (median, 1 epoch)")
axes[0].set_title("(a) KL divergence", loc="left"); axes[0].axhline(0.05, color=MUTED, lw=0.7, ls=(0, (2, 2)))
axes[0].text(19.8, 0.062, "point of no return ≈ 0.05", ha="right", fontsize=6.5, color=MUTED)
axes[1].set_ylim(0, 60); axes[1].set_ylabel("pass@1 test (mean of 3 evals)"); axes[1].set_title("(b) test performance", loc="left")
for ax in axes:
    ax.set_xlim(0, XMAX); ax.set_xlabel("epochs"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
fig.text(0.5, -0.15, "Same recipe (LoRA r=8, lr 3e-6, β=0.01, G=8), one delta: the KL reference. Thin vertical lines: re-anchoring of the moving run. Cross: collapse.", ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_fixed_vs_moving"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
