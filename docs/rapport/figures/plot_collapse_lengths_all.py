"""Panorama : longueur d'épisode, tours et tokens par tour sur 70 époques, 16 runs (8 stables, 8 collapses).
Trois panneaux en colonne. Données : collapse_lengths_all.json (extract_lengths.py all, avec la KL par pas).
Point plein = première demi-époque dont la KL médiane dépasse 1 (collapse), étiqueté du nom du run."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("docs/rapport/figures/collapse_lengths_all.json"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DASH = (0, (4, 2))
# deux familles de teintes : froides = stables, chaudes = collapses ; tirets = G=8 ou full-FT
STYLE = {
    "exp23.1_verl_100ep":            ("#1f5fb4", DASH, "Réplication-FT (G=8, ancre fixe)"),
    "exp25_r8_anchor4ep":            ("#0e9f8a", DASH, "Ancre-Mobile β0,01 (G=8)"),
    "exp31_r8_anchor_b0001":         ("#6fb3d6", DASH, "Ancre-Mobile β0,001 (G=8)"),
    "exp41_g8_8tasks_anchor8":       ("#2a78d6", DASH, "exp41 G=8, ancre /8 ép."),
    "exp39_g16_8tasks":              ("#3b3fa0", "-",  "exp39 G=16, 8 tâches, ancre /4 ép."),
    "exp32_horizon":                 ("#7b52ab", "-",  "Horizon (G=16)"),
    "exp35_budget1024":              ("#4a9d5b", "-",  "Budget (G=16)"),
    "exp33.1_depth_auto":            ("#a97fd6", "-",  "Profondeur (G=16)"),
    "exp36_n16":                     ("#eb6834", "-",  "Contrôle-G16 (4 tâches)"),
    "exp40_g16_8tasks_anchor8":      ("#c8102e", "-",  "exp40 G=16, 8 tâches, ancre /8 ép."),
    "exp36.1_anchor12":              ("#e87ba4", "-",  "exp36.1 G=16, ancre /12 ép."),
    "exp34_magellan":                ("#a3405c", "-",  "MAGELLAN (G=16)"),
    "exp30_r8_fixedanchor":          ("#b8860b", DASH, "exp30 G=8, ancre FIXE β0,01"),
    "exp31.1_r8_fixedanchor_b00001": ("#e0a020", DASH, "exp31.1 G=8, ancre fixe β0,0001"),
    "exp24_r16_lr3e-6_b0.01":        ("#8b4513", DASH, "grille LoRA r16 β0,01 (ancre fixe)"),
    "exp24_r64_lr3e-6_b0.001":       ("#d2691e", DASH, "grille LoRA r64 β0,001 (ancre fixe)"),
}
NUM = {r: i + 1 for i, r in enumerate(k for k, v in STYLE.items() if k in (
    "exp36_n16", "exp40_g16_8tasks_anchor8", "exp36.1_anchor12", "exp34_magellan", "exp30_r8_fixedanchor",
    "exp31.1_r8_fixedanchor_b00001", "exp24_r16_lr3e-6_b0.01", "exp24_r64_lr3e-6_b0.001"))}
# reprises depuis le best du run parent (ancre repositionnée sur le best, Adam à zéro) : pointillé, décalé
CONT = {"exp33.1_depth_auto": ("exp33.2_from72", 44.53), "exp25_r8_anchor4ep": ("exp25.2_from65", 111.5)}
XMAX, BIN = 70, 0.5
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.6), constrained_layout=True, sharex=True)
for run, (col, ls, label) in STYLE.items():
    r = np.array(d[run]["rows"]); ep, ln, tr, kl = r[:, 1], r[:, 2], r[:, 3], r[:, 5]
    edges = np.arange(0, XMAX + BIN, BIN); x, y_len, y_tr, y_kl = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ep >= a) & (ep < b)
        if m.sum() >= 3:
            x.append((a + b) / 2); y_len.append(ln[m].mean()); y_tr.append(tr[m].mean()); y_kl.append(np.median(kl[m]))
    x, y_len, y_tr, y_kl = map(np.array, (x, y_len, y_tr, y_kl))
    stable = d[run]["cat"] == "stable"
    # collapse = première demi-époque à KL médiane > 1 ; on coupe le tracé 2 époques après (le régime dégénéré
    # sort de l'échelle et n'apporte rien de plus)
    # KL médiane > 1 sur 4 demi-époques consécutives (écarte la dérive transitoire du 1er cycle d'exp40, ép. 7)
    sustained = [i for i in range(len(y_kl) - 3) if np.all(y_kl[i:i + 4] > 1)]
    i_c = sustained[0] if (sustained and not stable) else None
    keep = slice(None) if i_c is None else slice(0, min(len(x), i_c + 5))
    if i_c is not None:
        label = f"{NUM[run]} · {label} — collapse ép. {x[i_c]:.0f}"
    for k_ax, (ax, y) in enumerate(zip(axes, (y_len, y_tr, y_len / y_tr))):
        ax.plot(x[keep], y[keep], color=col, ls=ls, lw=1.6 if not stable else 1.3, alpha=1.0 if not stable else 0.9, label=label)
    if run in CONT and CONT[run][0] in d:
        child, off = CONT[run]; rc = np.array(d[child]["rows"]); epc, lnc, trc = rc[:, 1] + off, rc[:, 2], rc[:, 3]
        xc, yl, yt = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (epc >= a) & (epc < b)
            if m.sum() >= 3: xc.append((a + b) / 2); yl.append(lnc[m].mean()); yt.append(trc[m].mean())
        xc, yl, yt = map(np.array, (xc, yl, yt))
        if len(xc):
            for ax, y in zip(axes, (yl, yt, yl / yt)): ax.plot(xc, y, color=col, ls=(0, (2, 2)), lw=1.0, alpha=0.8)
        if i_c is not None:
            ax.plot(x[i_c], y[i_c], "o", ms=9, color=col, mec="white", mew=0.8)
            ax.annotate(str(NUM[run]), (x[i_c], y[i_c]), ha="center", va="center", fontsize=6, color="white", fontweight="bold")
axes[0].set_yscale("log"); axes[0].set_ylim(150, 12000); axes[0].set_ylabel("tokens par épisode (moy.)")
axes[0].set_title("(a) longueur d'épisode — actions, observations et balises confondues", loc="left")
axes[1].set_ylim(0, 31); axes[1].set_ylabel("tours par épisode (moy.)"); axes[1].set_title("(b) tours par épisode (cap 30 ; Horizon : 10 puis 20 jusqu'à l'époque 30)", loc="left")
axes[2].set_yscale("log"); axes[2].set_ylim(7, 1000); axes[2].set_ylabel("tokens par tour (≈ len / tours)")
axes[2].set_title("(c) tokens par tour — point numéroté : KL médiane > 1 pendant 2 époques consécutives (collapse)", loc="left")
axes[2].set_xlabel("époque")
for ax in axes:
    ax.set_xlim(0, XMAX); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03), columnspacing=2.5)
fig.text(0.27, -0.028, "stables (teintes froides)", ha="center", fontsize=7, color=MUTED, fontweight="bold")
fig.text(0.73, -0.028, "collapses (teintes chaudes), numérotés · pointillé fin = reprise depuis le best après purge", ha="center", fontsize=7, color=MUTED, fontweight="bold")
out = "docs/rapport/figures/fig_collapse_lengths_all"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight"); print("ok")
