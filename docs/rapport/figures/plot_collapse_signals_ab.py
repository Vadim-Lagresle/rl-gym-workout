"""Deux variantes de la figure « signaux qui précèdent un collapse » (16/09/2026, refonte des slides).
(a) fig_collapse_signals_a : sans G=16 — références stables (full-FT, ReLoRA G=8 β0,01 et β0,001) contre les
    collapses à référence fixe (grille LoRA, exp30, exp31.1). Lecture : la KL part d'abord.
(b) fig_collapse_signals_b : les mêmes références G=8 en gris + les collapses G=16 sans curriculum
    (exp36, exp36.1, exp40). Lecture : l'entropie part d'abord, la KL reste à 1e-3 jusqu'au bout.
Entropie = moyenne glissante 1 époque ; KL = médiane glissante 1 époque ; 16 premières époques ; tracé arrêté
à la première demi-époque de KL médiane > 1 (croix). Style : stables trait plein, collapses tirets.
Légende sous la figure. Données : logs/<run>.log."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; DASH = (0, (4, 2))
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
def load(run):
    ep, ent, kl = [], [], []
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "kl" in d and "entropy" in d: ep.append(float(d["epoch"])); ent.append(float(d["entropy"])); kl.append(float(d["kl"]))
    return np.array(ep), np.array(ent), np.array(kl)
def series(run, xmax=16, step=0.25):
    ep, ent, kl = load(run); xs, ye, yk = [], [], []
    for c in np.arange(0.5, min(xmax, ep[-1]) + 1e-9, step):
        m = (ep >= c - 0.5) & (ep < c + 0.5)
        if m.sum() >= 3: xs.append(c); ye.append(ent[m].mean()); yk.append(np.median(kl[m]))
    xs, ye, yk = map(np.array, (xs, ye, yk)); cut = np.where(yk > 1)[0]
    if len(cut): i = cut[0] + 1; return xs[:i], ye[:i], yk[:i], True
    return xs, ye, yk, False
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
def make(runs, out, note):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for run, (col, ls, lw, label) in runs.items():
        xs, ye, yk, collapsed = series(run)
        axes[0].plot(xs, ye, color=col, ls=ls, lw=lw, label=label); axes[1].plot(xs, np.clip(yk, 3e-4, 1e3), color=col, ls=ls, lw=lw)
        if collapsed:
            axes[0].plot(xs[-1], ye[-1], "x", ms=6, mew=1.5, color=col); axes[1].plot(xs[-1], min(yk[-1], 1e3), "x", ms=6, mew=1.5, color=col)
    axes[0].set_ylim(0, 1.6); axes[0].set_ylabel("entropie (moy. 1 époque)"); axes[0].set_title("(a) entropie de la politique", loc="left")
    axes[1].set_yscale("log"); axes[1].set_ylim(3e-4, 1e3); axes[1].set_ylabel("KL vers la référence (méd. 1 ép.)"); axes[1].set_title("(b) divergence KL", loc="left")
    for ax in axes:
        ax.set_xlim(0, 16); ax.set_xlabel("époques"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.26, note, ha="center", fontsize=6.5, color=MUTED)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok", out)
STABLE = {"exp23.1_verl_100ep": ("#1f5fb4", "-", 1.5, "stable · full-FT, référence fixe"),
          "exp25_r8_anchor4ep": ("#0e9f8a", "-", 1.5, "stable · ReLoRA r8 G=8, réf. mobile /4 ép., β0,01 (→ 73)"),
          "exp31_r8_anchor_b0001": ("#6fb3d6", "-", 1.5, "stable · ReLoRA r8 G=8, réf. mobile /4 ép., β0,001 (→ 53)")}
FIXED = {"exp30_r8_fixedanchor": ("#eb6834", DASH, 1.3, "collapse · LoRA r8, réf. FIXE, β0,01"),
         "exp31.1_r8_fixedanchor_b00001": ("#e0a020", DASH, 1.3, "collapse · LoRA r8, réf. fixe, β0,0001"),
         "exp24_r16_lr3e-6_b0.01": ("#a3405c", DASH, 1.3, "collapse · LoRA r16, réf. fixe, β0,01"),
         "exp24_r16_lr3e-6_b0.001": ("#c8102e", DASH, 1.3, "collapse · LoRA r16, réf. fixe, β0,001"),
         "exp24_r64_lr3e-6_b0.001": ("#8b4513", DASH, 1.3, "collapse · LoRA r64, réf. fixe, β0,001"),
         "exp24_r64_lr3e-6_b0.01": ("#d2691e", DASH, 1.3, "collapse · LoRA r64, réf. fixe, β0,01")}
make(STABLE | FIXED, "docs/rapport/figures/fig_collapse_signals_a",
     "croix : première demi-époque de KL médiane > 1, tracé arrêté. Tous à G=8, LR 3e-6 (full-FT : 1e-6). Trait plein = stable, tirets = collapse.")
GREY = {"exp25_r8_anchor4ep": ("#9aa0a6", "-", 1.2, "référence · ReLoRA r8 G=8, réf. mobile /4 ép. (stable, 73)"),
        "exp41_g8_8tasks_anchor8": ("#c2c6ca", "-", 1.2, "référence · ReLoRA r8 G=8, réf. mobile /8 ép. (stable, 64)")}
G16 = {"exp36_n16": ("#eb6834", DASH, 1.5, "collapse · G=16, 4 tâches/pas, réf. mobile /4 ép. (exp36)"),
       "exp40_g16_8tasks_anchor8": ("#c8102e", DASH, 1.5, "collapse · G=16, 8 tâches/pas, réf. mobile /8 ép. (exp40)"),
       "exp36.1_anchor12": ("#e87ba4", DASH, 1.5, "collapse · G=16, 4 tâches/pas, réf. mobile /12 ép. (exp36.1)")}
make(GREY | G16, "docs/rapport/figures/fig_collapse_signals_b",
     "Même recette ReLoRA que les références, seul changement G=8 → 16. L'entropie s'effondre alors que la KL est encore sous 0,1 (exp36) ou monte avec elle (exp40) ; l'explosion de KL vient après.")
