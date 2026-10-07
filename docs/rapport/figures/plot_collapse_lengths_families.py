"""Même figure que plot_collapse_lengths.py, pour les six runs désignés du rapport (15/09/2026) :
Réplication-FT (exp23.1), Ancre-Mobile (exp25), Contrôle-G16 (exp36), Horizon (exp32), Budget (exp35),
Profondeur (exp33.1). Données : collapse_lengths_families.json (extract_lengths.py families)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("docs/rapport/figures/collapse_lengths_families.json"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DASH = (0, (4, 2))
STYLE = {  # palette validée (validate_palette.js, light) ; tirets = références G=8
    "exp23.1_verl_100ep":  ("#1f5fb4", DASH, "Réplication-FT · full-FT, G=8"),
    "exp25_r8_anchor4ep":  ("#0e9f8a", DASH, "Ancre-Mobile · LoRA r8, G=8"),
    "exp36_n16":           ("#eb6834", "-",  "Contrôle-G16 · sans curriculum"),
    "exp32_horizon":       ("#a3405c", "-",  "Horizon · tours 10/20/30, G=16"),
    "exp35_budget1024":    ("#b8860b", "-",  "Budget · tokens/tour croissants, G=16"),
    "exp33.1_depth_auto":  ("#7b52ab", "-",  "Profondeur · recettes croissantes, G=16"),
}
COLLAPSE = {"exp36_n16": 10.0}
# reprises depuis le best du run parent (ancre repositionnée sur le best, Adam à zéro) : pointillé, décalé
CONT = {"exp33.1_depth_auto": ("exp33.2_from72", 44.53), "exp25_r8_anchor4ep": ("exp25.2_from65", 111.5)}
XMAX, BIN = 40, 1.0
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), constrained_layout=True)
for k_run, (run, (col, ls, label)) in enumerate(STYLE.items()):
    r = np.array(d[run]["rows"]); ep, ln, tr = r[:, 1], r[:, 2], r[:, 3]
    edges = np.arange(0, XMAX + BIN, BIN); x, y_len, y_tr = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ep >= a) & (ep < b)
        if m.sum() >= 3:
            x.append((a + b) / 2); y_len.append(ln[m].mean()); y_tr.append(tr[m].mean())
    x, y_len, y_tr = map(np.array, (x, y_len, y_tr))
    for ax, y in zip(axes, (y_len, y_tr, y_len / y_tr)):
        ax.plot(x, y, color=col, ls=ls, lw=1.3, label=label)
        for s_ in d[run]["anchors"]:
            e = s_ / d[run]["spe"]
            if e <= XMAX: ax.plot([e], [1.0 + 0.022 * k_run], marker="|", ms=4, color=col,
                                  transform=ax.get_xaxis_transform(), clip_on=False)
        if run in COLLAPSE:
            i = np.argmin(np.abs(x - COLLAPSE[run])); ax.plot(x[i], y[i], "o", ms=5, color=col)
    if run in CONT and CONT[run][0] in d:
        child, off = CONT[run]; rc = np.array(d[child]["rows"]); epc, lnc, trc = rc[:, 1] + off, rc[:, 2], rc[:, 3]
        xc, yl, yt = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (epc >= a) & (epc < b)
            if m.sum() >= 3: xc.append((a + b) / 2); yl.append(lnc[m].mean()); yt.append(trc[m].mean())
        xc, yl, yt = map(np.array, (xc, yl, yt))
        if len(xc):
            for ax, y in zip(axes, (yl, yt, yl / yt)): ax.plot(xc, y, color=col, ls=(0, (2, 2)), lw=1.1, alpha=0.8)
axes[0].set_yscale("log"); axes[0].set_ylim(150, 12000); axes[0].set_ylabel("tokens par épisode (moy.)")
axes[0].set_title("(a) longueur d'épisode", loc="left", pad=18)
axes[1].set_ylim(0, 31); axes[1].set_ylabel("tours par épisode (moy.)"); axes[1].set_title("(b) tours (cap 30)", loc="left", pad=18)
axes[2].set_yscale("log"); axes[2].set_ylim(7, 1000); axes[2].set_ylabel("tokens par tour (≈ len / tours)")
axes[2].set_title("(c) tokens par tour", loc="left", pad=18)
for ax in axes:
    ax.set_xlim(0, XMAX); ax.set_xlabel("époque"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.13))
fig.text(0.5, -0.18, "pointillé fin : reprise depuis le best après une purge (Profondeur → exp33.2, ép. 44,5) · traits au-dessus : ré-ancrages · point plein : 1re explosion de KL · "
         "Horizon plafonne les tours à 10 puis 20 jusqu'à l'époque 30", ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_collapse_lengths_families"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
