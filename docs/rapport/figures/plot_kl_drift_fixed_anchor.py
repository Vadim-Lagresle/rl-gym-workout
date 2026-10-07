"""Dérive de la KL à référence FIXE : géométrique ou linéaire ? LoRA ou taux d'apprentissage ? (16/09/2026)
KL médiane par demi-époque, échelle log : une dérive géométrique est une droite, une dérive linéaire s'aplatit.
(a) en fonction de l'époque ; (b) en fonction du taux d'apprentissage cumulé Σ lr (lu pas à pas dans le log,
donc exact même avec des coupes de LR) : si le LR pilote la dérive, les LoRA se superposent.
Référence mobile (exp25) en pointillé gris pour l'échelle. Runs coupés au collapse (KL médiane > 1)."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; DASH = (0, (4, 2))
RUNS = {  # run: (couleur, style, label)
    "exp23.1_verl_100ep":        ("#1f5fb4", "-",  "full-FT, LR 1e-6, β0,001 (Réplication, exp23.1)"),
    "exp30_r8_fixedanchor":      ("#eb6834", "-",  "LoRA r8, LR 3e-6, β0,01 (exp30)"),
    "exp24_r16_lr3e-6_b0.01":    ("#a3405c", "-",  "LoRA r16, LR 3e-6, β0,01 (exp24)"),
    "exp24_r64_lr3e-6_b0.001":   ("#b8860b", "-",  "LoRA r64, LR 3e-6, β0,001 (exp24)"),
    "exp24_r64_lr1e-5_b0.001":   ("#c8102e", "-",  "LoRA r64, LR 1e-5, β0,001 (exp24)"),
    "exp24_r16_lr1e-5_b0.001":   ("#7b52ab", "-",  "LoRA r16, LR 1e-5, β0,001 (exp24)"),
    "exp25_r8_anchor4ep":        ("#8a8f96", DASH, "référence MOBILE /4 ép. : LoRA r8, LR 3e-6 (exp25)"),
}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
def load(run):
    rows = {}; last_lr = None
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "learning_rate" in d: last_lr = float(d["learning_rate"])
            if "kl" in d and last_lr is not None:   # certains dicts (eval, etc.) n'ont pas le LR : on garde le dernier vu
                rows[int(m.group(1))] = (float(d["epoch"]), float(d["kl"]), last_lr)
    st = np.array(sorted(rows)); ep = np.array([rows[s][0] for s in st]); kl = np.array([rows[s][1] for s in st]); lr = np.array([rows[s][2] for s in st])
    return st, ep, kl, np.cumsum(lr)
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True)
print("run                        | KL médiane à ép. 2, 4, 6, 8, 10, 14, 20 | facteur ×/2 ép. (4→6, 6→8, 8→10)")
for run, (col, ls, label) in RUNS.items():
    try: st, ep, kl, clr = load(run)
    except FileNotFoundError: continue
    if len(st) < 50: print(run, "log trop court"); continue
    xs_e, xs_l, ys = [], [], []
    for e in np.arange(0, ep[-1], 0.5):
        m = (ep >= e) & (ep < e + 0.5)
        if m.sum() >= 3:
            med = np.median(kl[m]); xs_e.append(e + 0.25); xs_l.append(clr[m].mean() * 1e3); ys.append(med)
            if med > 1 and "anchor4ep" not in run: break
    xs_e, xs_l, ys = map(np.array, (xs_e, xs_l, ys))
    axes[0].plot(xs_e, ys, color=col, ls=ls, lw=1.4, label=label); axes[1].plot(xs_l, ys, color=col, ls=ls, lw=1.4)
    if ys[-1] > 1:
        axes[0].plot(xs_e[-1], ys[-1], "x", ms=6, color=col, mew=1.5); axes[1].plot(xs_l[-1], ys[-1], "x", ms=6, color=col, mew=1.5)
    at = lambda e: (ys[np.argmin(np.abs(xs_e - e))] if xs_e[-1] >= e - 0.3 else float("nan"))
    vals = [at(e) for e in (2, 4, 6, 8, 10, 14, 20)]
    fac = [at(b) / at(a) if not np.isnan(at(a)) and not np.isnan(at(b)) else float("nan") for a, b in ((4, 6), (6, 8), (8, 10))]
    print(f"{run[:26]:26s} | " + " ".join(f"{v:8.2g}" for v in vals) + " | " + " ".join(f"{f:5.1f}" for f in fac))
for ax in axes:
    ax.set_yscale("log"); ax.set_ylim(3e-4, 30); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axhline(10, color=MUTED, lw=0.7, ls=(0, (2, 2))); ax.axhline(0.05, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].text(0.3, 12, "borne k3 = 10 (verl)", fontsize=6.5, color=MUTED); axes[0].text(0.3, 0.06, "point de non-retour observé ≈ 0,05", fontsize=6.5, color=MUTED)
axes[0].set_xlim(0, 30); axes[0].set_xlabel("époque"); axes[0].set_ylabel("KL médiane par demi-époque (log)")
axes[0].set_title("(a) dérive par époque", loc="left")
axes[1].set_xlim(0, 2.5); axes[1].set_xlabel("taux d'apprentissage cumulé Σ lr (×10⁻³)"); axes[1].set_title("(b) dérive par LR cumulé", loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.22, "croix : collapse (KL médiane > 1), tracé arrêté. Tous zero-shot, LR constant, référence KL = Qwen de base (sauf exp25, mobile).",
         ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_kl_drift_fixed_anchor"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight"); print("ok")
