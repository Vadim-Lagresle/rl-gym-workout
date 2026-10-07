"""G=16 : quel signal part en premier ? exp36 (ancre /4 ép.) : la KL est remise à 1e-3 à l'ép. 4 et 8, et le run
meurt quand même par l'entropie (1,05 → 0,15 entre les ép. 7 et 10, KL médiane ≤ 0,01 jusqu'à l'ép. 10).
exp40 (ancre /8 ép.) : 1er cycle trop long, la KL dérive d'abord comme une référence fixe, l'entropie suit.
Référence G=8 (exp25) en gris. Entropie = moyenne glissante 1 ép. ; KL = médiane glissante 1 ép."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {"exp25_r8_anchor4ep": ("#9aa0a6", "-", 1.3, "G=8 reference (ReLoRA, anchor every 4 epochs)"),
        "exp36_n16": ("#eb6834", "-", 1.8, "G=16, anchor every 4 epochs (Control-G16)"),
        "exp40_g16_8tasks_anchor8": ("#c8102e", (0, (4, 2)), 1.5, "G=16, anchor every 8 epochs (exp40)")}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})"); anch_re = re.compile(r"RÉ-ANCRAGE #\d+ @ step (\d+)")
def load(run):
    ep, ent, kl, anchors = [], [], [], []
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d and "entropy" in d: ep.append(float(d["epoch"])); ent.append(float(d["entropy"])); kl.append(float(d["kl"]))
        m = anch_re.search(line)
        if m: anchors.append(int(m.group(1)))
    ep, ent, kl = map(np.array, (ep, ent, kl)); spe = len(ep) / ep[-1]
    return ep, ent, kl, np.array(sorted(set(anchors))) / spe
def series(ep, y, med, xmax=16):
    xs, ys = [], []
    for c in np.arange(0.5, min(xmax, ep[-1]) + 1e-9, 0.25):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ys.append(np.median(y[m]) if med else y[m].mean())
    return np.array(xs), np.array(ys)
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
for run, (col, ls, lw, label) in RUNS.items():
    ep, ent, kl, anch = load(run)
    xe, ye = series(ep, ent, False); xk, yk = series(ep, kl, True)
    cut = np.where(yk > 1)[0]; i = (cut[0] + 1) if len(cut) else len(xk)
    axes[0].plot(xe[:i], ye[:i], color=col, ls=ls, lw=lw, label=label); axes[1].plot(xk[:i], np.clip(yk[:i], 3e-4, 1e3), color=col, ls=ls, lw=lw)
    if len(cut):
        for ax, x, y in ((axes[0], xe[i-1], ye[i-1]), (axes[1], xk[i-1], min(yk[i-1], 1e3))): ax.plot(x, y, "x", ms=7, mew=1.8, color=col)
    for a in anch[anch <= 16]:
        for ax in axes: ax.axvline(a, color=col, lw=0.7, alpha=0.45, ls=":" if "exp40" in run else "-")
# annotations sur exp36
axes[0].annotate("exp36: entropy 1.05 → 0.15\nbetween epochs 7 and 10,\nwhile KL median ≤ 0.01", xy=(9.0, 0.6), xytext=(10.6, 1.25), fontsize=7, color="#eb6834",
                 arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.7))
axes[1].annotate("exp36: anchors at 4 and 8\nreset KL to 1e-3;\nexplosion only at epoch 10", xy=(8.2, 1.2e-3), xytext=(9.3, 3e-2), fontsize=7, color="#eb6834",
                 arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.7))
axes[1].annotate("exp40: KL drifts first\n(0.1 at epoch 5.5),\nentropy follows", xy=(5.6, 0.12), xytext=(0.6, 30), fontsize=7, color="#c8102e",
                 arrowprops=dict(arrowstyle="-", color="#c8102e", lw=0.7))
axes[0].set_ylim(0, 1.6); axes[0].set_ylabel("entropy (mean, 1 epoch)"); axes[0].set_title("(a) policy entropy", loc="left")
axes[1].set_yscale("log"); axes[1].set_ylim(3e-4, 1e3); axes[1].set_ylabel("KL to the reference (median, 1 epoch)"); axes[1].set_title("(b) KL divergence", loc="left")
for ax in axes:
    ax.set_xlim(0, 16); ax.set_xlabel("epochs"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
fig.text(0.5, -0.15, "Same ReLoRA recipe, only G changes. Vertical lines: re-anchoring (solid exp36 and reference, dotted exp40). Cross: collapse, trace stopped.", ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_g16_entropy_first"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
