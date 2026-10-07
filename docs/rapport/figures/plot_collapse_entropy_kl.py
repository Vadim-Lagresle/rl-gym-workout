"""Étape 1 de la validation de l'hypothèse k3 (15/09/2026) : les pics de KL sont
concentrés sur quelques tokens (le reward du pas reste normal) et ne deviennent
terminaux que quand l'entropie est basse. Données : collapse_entropy_kl.json
(produit depuis logs/<run>.log, une ligne par pas : epoch, kl, entropie, reward, longueur).
(a) médianes glissantes (11 pas) KL vs entropie, tous les pas de 9 runs ;
(b) le pas suivant un pic KL>1 : KL(t+1) en fonction de l'entropie au pic.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("docs/rapport/figures/collapse_entropy_kl.json"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # stable / collapse — même paire que le reste du deck
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)

W = 11  # fenêtre de la médiane glissante (pas)
def rollmed(x, w=W):
    x = np.asarray(x, float); h = w // 2
    return np.array([np.median(x[max(0, i-h):i+h+1]) for i in range(len(x))])

# (a) régime soutenu : médianes glissantes, hors régime dégénéré (épisodes > 3000 tokens)
ax = axes[0]
for run, r in d.items():
    st = np.array(r["steps"]); keep = st[:, 5] < 3000
    H, K = rollmed(st[:, 3]), rollmed(st[:, 2])
    col = BLUE if r["cat"] == "stable" else ORANGE
    ax.scatter(H[keep], np.clip(K[keep], 3e-4, 1e6), s=4, color=col, alpha=0.25 if r["cat"] == "stable" else 0.45,
               lw=0, rasterized=True)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(0.49, 3e5, "entropie 0,5", ha="right", va="top", fontsize=7, color=MUTED)
ax.set_yscale("log"); ax.set_ylim(3e-4, 1e6); ax.set_xlim(0, 1.6)
ax.set_xlabel("entropie du pas (médiane glissante, 11 pas)"); ax.set_ylabel("KL du pas (médiane glissante)")
ax.set_title("(a) KL soutenue > 1 : seulement sous entropie 0,5", loc="left")
ax.text(1.55, 1.5e-3, "régime sain 10⁻³", ha="right", va="bottom", fontsize=7, color=MUTED)

# (b) un pic isolé et sa suite : KL au pas suivant vs entropie au moment du pic
ax = axes[1]
pts = {"stable": [], "collapse": []}
for run, r in d.items():
    st = r["steps"]
    for i, s in enumerate(st[:-1]):
        prev = np.median([x[2] for x in st[max(0, i-10):i]]) if i else 0.0
        if s[2] > 1 and s[5] < 3000 and prev < 0.1:   # pic KL>1 isolé : régime sain juste avant (médiane 10 pas < 0,1)
            pts[r["cat"]].append((s[3], max(x[2] for x in st[i+1:i+11]), s[4], run, s[0]))
for cat, col, lab in (("stable", BLUE, "runs stables (5)"), ("collapse", ORANGE, "runs qui collapsent (4)")):
    p = np.array([(a, b) for a, b, *_ in pts[cat]])
    ax.scatter(p[:, 0], np.clip(p[:, 1], 3e-4, 1e6), s=14, color=col, alpha=0.75, lw=0, label=lab)
ax.axhline(10, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(1.58, 13, "escalade", ha="right", va="bottom", fontsize=7, color=MUTED)
ax.text(1.58, 7, "pic absorbé", ha="right", va="top", fontsize=7, color=MUTED)
# étiquettes directes de deux cas emblématiques (point = (H au pic, max KL des 10 pas suivants))
def pt(run, step):
    st = d[run]["steps"]; i = [k for k, s in enumerate(st) if s[0] == step][0]
    return st[i][3], max(x[2] for x in st[i+1:i+11])
ax.annotate("Horizon (exp32), pas 253, entropie 0,90 :\nKL 137 puis 0,03 au pas suivant, max 1,25", pt("exp32_horizon", 253),
            xytext=(0.62, 4e-3), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("G=16 sans curriculum (exp40), pas 696, entropie 0,28 :\nKL 4 puis 30, 126, 2 400 dans les 15 pas", pt("exp40_g16_8tasks_anchor8", 696),
            xytext=(0.36, 4e3), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_yscale("log"); ax.set_ylim(3e-4, 1e6); ax.set_xlim(0, 1.6)
ax.set_xlabel("entropie au moment du pic (KL > 1)"); ax.set_ylabel("KL max des 10 pas suivants")
ax.set_title("(b) pic isolé depuis un régime sain : absorbé ou escalade ?", loc="left")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, markerscale=1.6, bbox_to_anchor=(0.5, -0.06))

for ax in axes:
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
out = "docs/rapport/figures/fig_collapse_entropy_kl"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
n_st = sum(len(v) for v in pts.values()); print("ok, pics KL>1 hors dégénéré :", {k: len(v) for k, v in pts.items()})
