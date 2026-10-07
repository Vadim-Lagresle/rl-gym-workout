"""Figures post-rapport à partir de runs_data.json (produit par extract_runs.py).

  fig1_g_x_curriculum.png  Pass@1 test vs époque, panneau G=8 | panneau G=16
  fig2_klclamp_mechanism.png  KL et entropie : effet de --kl-clamp 10 (ancre mobile G=16 | ancre fixe G=8)
  fig3_summary.png          best et plateau (moyenne des 5 dernières évals) par run
  fig4_equal_compute.png    Pass@1 vs PAS d'optimisation (64 traj/pas partout = calcul égal)
  fig5_relora_ablation.png  quel ingrédient de ReLoRA stabilise ? (merge vs ref, ref+LR÷3, ref+purge Adam)
  fig6_drift_clock.png      la dérive KL suit une horloge en PAS : ré-ancrer avant ~370 pas, quel que soit G
  fig7_exp51_anchor.png     exp51 (ancre 444 pas) vs exp51.1 (372 pas) vs exp43 : le rythme d'ancre tranche
Usage : python docs/post_rapport/plot_runs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
D = json.loads((HERE / "runs_data.json").read_text())
OUT = HERE / "figures"

# Couleur = traitement (même sens dans toutes les figures)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": "white", "savefig.dpi": 160})

def evals(run, k=5):
    """(époques, pass@1 brut, moyenne glissante sur k évals)."""
    d = D[run]; ev = np.array(d["eval"])
    ep, p = ev[:, 0] / d["steps_per_epoch"], ev[:, 1] * 100
    sm = np.array([p[max(0, i - k + 1):i + 1].mean() for i in range(len(p))])
    return ep, p, sm

def per_epoch(run, key):
    """Médiane par époque d'une métrique train (robuste aux pics de KL)."""
    d = D[run]; t = d["train"]
    ep = np.array(t["step"]) / d["steps_per_epoch"]
    v = np.array([np.nan if x is None else x for x in t[key]], float)
    bins = np.arange(0, np.ceil(ep.max()) + 1)
    idx = np.digitize(ep, bins) - 1
    med = np.array([np.nanmedian(v[idx == i]) if (idx == i).any() else np.nan for i in range(len(bins))])
    return bins + 0.5, med

REFS = [(18, "Qwen2.5-3B nu 18"), (75, "papier AgentGym-RL 75")]

def refs(ax):
    for y, lab in REFS:
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.text(0.01, y + 1, lab, transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=8, color=INK2)

def refs_v(ax):
    for x, lab in REFS:
        ax.axvline(x, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.text(x, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=INK2)

def curve(ax, run, color, label, dashed=False, dy=0, steps=False):
    ep, p, sm = evals(run)
    if steps:
        ep = ep * D[run]["steps_per_epoch"]
    ax.scatter(ep, p, s=6, color=color, alpha=0.18, lw=0, zorder=2)
    ls = (0, (5, 2)) if dashed else "-"
    ax.plot(ep, sm, color=color, lw=2, ls=ls, label=label, zorder=3)
    ax.annotate(label.split(" ")[0], (ep[-1], sm[-1]), xytext=(4, dy), textcoords="offset points",
                color=INK, fontsize=8, va="center")

# ── Figure 1 : G × curriculum ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ax = axes[0]
curve(ax, "exp25_r8_anchor4ep", BLUE, "exp25 sans curriculum, ancre /4 ép.")
curve(ax, "exp41_g8_8tasks_anchor8", VIOLET, "exp41 sans curriculum, ancre /8 ép.")
curve(ax, "exp44_horizon_g8", ORANGE, "exp44 curriculum Horizon 10/20/30")
ax.set_title("G = 8 (8 tâches × 8 rollouts / pas)", loc="left", fontsize=11, color=INK)
ax = axes[1]
curve(ax, "exp36_n16", BLUE, "exp36 sans curriculum (collapse ép. 10-15)", dy=6)
curve(ax, "exp40_g16_8tasks_anchor8", VIOLET, "exp40 8 tâches×16, ancre /8 (collapse ép. 15)", dy=-6)
curve(ax, "exp43_g16_klclamp10", AQUA, "exp43 sans curriculum + kl-clamp 10", dy=-6)
curve(ax, "exp32_horizon", ORANGE, "exp32 curriculum Horizon 10/20/30", dashed=True, dy=6)
ax.set_title("G = 16 (4 tâches × 16 rollouts / pas)", loc="left", fontsize=11, color=INK)
for ax in axes:
    refs(ax); ax.set_xlim(0, 118); ax.set_ylim(0, 90); ax.set_xlabel("époque")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
axes[0].set_ylabel("Pass@1 test (%) — moy. glissante 5 évals")
fig.suptitle("À G=16, le curriculum et la borne KL mènent au même plateau (~78-82) ; à G=8, le curriculum ne change presque rien (~60)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig1_g_x_curriculum.png"); plt.close(fig)

# ── Figure 2 : mécanisme de la borne KL ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
pairs = [("Ancre mobile /4 ép., G=16", "exp36_n16", "exp43_g16_klclamp10", 20),
         ("Référence FIXE (Qwen nu), G=8", "exp30_r8_fixedanchor", "exp45_fixedanchor_klclamp10", 15)]
for c, (title, base, clamp, xmax) in enumerate(pairs):
    for r, key in enumerate(["kl", "entropy"]):
        ax = axes[r, c]
        for run, col, lab in [(base, BLUE, f"{base.split('_')[0]} sans borne"),
                              (clamp, AQUA, f"{clamp.split('_')[0]} kl-clamp 10")]:
            x, y = per_epoch(run, key)
            ax.plot(x, y, color=col, lw=2, marker="o", ms=3.5, label=lab)
        if key == "kl":
            ax.set_yscale("log"); ax.set_ylabel("KL k3 (médiane / époque, log)")
            ax.axhline(0.05, color=INK2, lw=0.8, ls=(0, (4, 3)))
            ax.text(0.01, 0.05, " 0.05 = dérive", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2, va="bottom")
            ax.set_title(title, loc="left", fontsize=11, color=INK)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        else:
            ax.set_ylabel("entropie (médiane / époque)"); ax.set_xlabel("époque"); ax.set_ylim(0, 1.5)
            ax.axhline(0.15, color=INK2, lw=0.8, ls=(0, (4, 3)))
            ax.text(0.01, 0.15, " 0.15 = collapse", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2, va="bottom")
        ax.set_xlim(0, xmax)
axes[0, 0].set_xlim(0, 80); axes[1, 0].set_xlim(0, 80)
fig.suptitle("La borne k3 supprime l'explosion de KL dans les deux cas — mais ne sauve la politique que si l'ancre est mobile",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig2_klclamp_mechanism.png"); plt.close(fig)

# ── Figure 3 : synthèse best / plateau ───────────────────────────────────────
rows = [("exp45 réf. fixe + clamp, G8", "exp45_fixedanchor_klclamp10", AQUA),
        ("exp30 réf. fixe, G8", "exp30_r8_fixedanchor", BLUE),
        ("exp40 8×16, ancre /8", "exp40_g16_8tasks_anchor8", VIOLET),
        ("exp36 G16 sans curriculum", "exp36_n16", BLUE),
        ("exp44 G8 + Horizon", "exp44_horizon_g8", ORANGE),
        ("exp41 G8 ancre /8", "exp41_g8_8tasks_anchor8", VIOLET),
        ("exp25 G8 ancre /4", "exp25_r8_anchor4ep", BLUE),
        ("exp43 G16 + kl-clamp", "exp43_g16_klclamp10", AQUA),
        ("exp32 G16 + Horizon", "exp32_horizon", ORANGE)]
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (lab, run, col) in enumerate(rows):
    _, p, _ = evals(run)
    best, last5 = p.max(), p[-5:].mean()
    ax.plot([last5, best], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(last5, i, s=60, color=col, zorder=3, edgecolor="white", lw=1.5)
    ax.scatter(best, i, s=60, facecolor="white", edgecolor=col, lw=2, zorder=3)
    ax.text(best + 1.5, i, f"{best:.0f}", va="center", fontsize=8, color=INK)
    ax.text(last5 - 1.5, i, f"{last5:.0f}", va="center", ha="right", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.grid(axis="y", visible=False)
refs_v(ax); ax.set_xlim(0, 95); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("Pass@1 test (%)  —  ● moyenne des 5 dernières évals   ○ meilleure éval")
ax.set_title("Best et plateau final par run", loc="left", fontsize=11, color=INK, pad=16)
fig.tight_layout(); fig.savefig(OUT / "fig3_summary.png"); plt.close(fig)
# ── Figure 4 : même abscisse en pas (= trajectoires, 64 / pas pour tous ces runs) ──
fig, ax = plt.subplots(figsize=(8, 4.4))
curve(ax, "exp25_r8_anchor4ep", BLUE, "exp25 G8, ancre /4 ép. (184 pas)", steps=True)
curve(ax, "exp41_g8_8tasks_anchor8", VIOLET, "exp41 G8, ancre /8 ép. (374 pas)", steps=True, dy=4)
curve(ax, "exp44_horizon_g8", ORANGE, "exp44 G8 + Horizon", steps=True, dy=-4)
curve(ax, "exp43_g16_klclamp10", AQUA, "exp43 G16 + kl-clamp", steps=True, dashed=True, dy=-4)
curve(ax, "exp32_horizon", ORANGE, "exp32 G16 + Horizon", steps=True, dashed=True, dy=4)
refs(ax); ax.set_xlim(0, 5600); ax.set_ylim(0, 90)
ax.set_xlabel("pas d'optimisation (64 trajectoires / pas)"); ax.set_ylabel("Pass@1 test (%) — moy. glissante 5 évals")
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.set_title("À calcul égal, l'écart G=16 / G=8 se réduit : exp41 (G8) suit les G16 jusqu'au pas ~2 700",
             loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig4_equal_compute.png"); plt.close(fig)
# ── Figure 5 : ablation des ingrédients de ReLoRA (G=8) ─────────────────────
ABL = [("exp25_r8_anchor4ep", BLUE, "exp25 ReLoRA (merge : reset B·A + Adam)", 0),
       ("exp46_movingref_g8", ORANGE, "exp46 référence mobile seule (ref)", 4),
       ("exp48_movingref_lr1e-6", VIOLET, "exp48 ref + LR 1e-6", -4),
       ("exp49_movingref_resetadam", AQUA, "exp49 ref + purge Adam", 0)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for run, col, lab, dy in ABL:
    curve(ax, run, col, lab, dy=dy)
refs(ax); ax.set_xlim(0, 60); ax.set_ylim(0, 90); ax.set_xlabel("époque")
ax.set_ylabel("Pass@1 test (%) — moy. glissante 5 évals")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("Pass@1 : exp49 suit exp25 jusqu'à l'ép. 40", loc="left", fontsize=10, color=INK)
ax = axes[1]
for run, col, lab, _ in ABL:
    x, y = per_epoch(run, "entropy")
    ax.plot(x, y, color=col, lw=2, label=lab.split(" ")[0])
ax.axhline(0.15, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.text(59, 0.17, "0.15 = collapse", fontsize=8, color=INK2, ha="right")
ax.set_xlim(0, 60); ax.set_ylim(0, 1.6); ax.set_xlabel("époque"); ax.set_ylabel("entropie (médiane / époque)")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("Entropie : seul exp25 reste exploratoire", loc="left", fontsize=10, color=INK)
fig.suptitle("Ingrédients de ReLoRA : seul le reset B·A garde une dynamique saine ; la purge d'Adam sauve le score mais pas l'exploration",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig5_relora_ablation.png"); plt.close(fig)

# ── Figure 6 : horloge de dérive en pas ──────────────────────────────────────
def kl_by_steps(run, width=23):
    t = D[run]["train"]; st = np.array(t["step"]); v = np.array([np.nan if x is None else x for x in t["kl"]], float)
    edges = np.arange(0, st.max() + width, width); idx = np.digitize(st, edges) - 1
    med = np.array([np.nanmedian(v[idx == i]) if (idx == i).any() else np.nan for i in range(len(edges))])
    return edges + width / 2, med
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for run, col, lab, ls in [("exp30_r8_fixedanchor", BLUE, "exp30 G8 réf. fixe, LR 3e-6", "-"),
                          ("exp47_fixedanchor_lr1e-6", VIOLET, "exp47 G8 réf. fixe, LR 1e-6", "-"),
                          ("exp36.1_anchor12", ORANGE, "exp36.1 G16 ancre /12 ép. (~1 100 pas)", (0, (5, 2))),
                          ("exp50_g16_horizon_klclamp_anchor10", AQUA, "exp50 G16 ancre /10 ép. + borne + curriculum", (0, (5, 2)))]:
    x, y = kl_by_steps(run); ax.plot(x, y, color=col, lw=2, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_xlim(0, 1000); ax.set_ylim(5e-4, 1e4)
ax.axvline(372, color=INK2, lw=0.8, ls=(0, (4, 3))); ax.text(378, 3e3, "372 pas = ancre /4 ép. à G=16", fontsize=8, color=INK2)
ax.axhline(0.1, color=INK2, lw=0.8, ls=(0, (1, 2))); ax.text(990, 0.13, "KL 0.1", fontsize=8, color=INK2, ha="right")
ax.set_xlabel("pas d'optimisation (sans ré-ancrage sur cette plage)"); ax.set_ylabel("KL k3 (médiane / 23 pas, log)")
ax.set_title("Sans reset : KL ≈ 0.1 vers ~370 pas à LR 3e-6\n(G=8 comme G=16), ~780 pas à LR 1e-6", loc="left", fontsize=10, color=INK)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
ax = axes[1]
curve(ax, "exp43_g16_klclamp10", AQUA, "exp43 ancre tous les 372 pas", steps=True)
curve(ax, "exp51_depthbal_uniform", ORANGE, "exp51 ancre tous les 444 pas (+ rééquilibrage d1-d4)", steps=True)
refs(ax); ax.set_xlim(0, 2500); ax.set_ylim(0, 90); ax.set_xlabel("pas d'optimisation")
ax.set_ylabel("Pass@1 test (%) — moy. glissante 5 évals"); ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.set_title("G=16 + borne : 444 pas entre ancres\n(+ rééquilibrage d1-d4) suffisent à casser", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig6_drift_clock.png"); plt.close(fig)

# ── Figure 7 : exp51 vs exp51.1 (seul delta : ancre 444 → 372 pas) ──────────
def per_steps(run, key, width=46):
    t = D[run]["train"]; st = np.array(t["step"]); v = np.array([np.nan if x is None else x for x in t[key]], float)
    edges = np.arange(0, st.max() + width, width); idx = np.digitize(st, edges) - 1
    return edges + width / 2, np.array([np.nanmedian(v[idx == i]) if (idx == i).any() else np.nan for i in range(len(edges))])
S7 = [("exp43_g16_klclamp10", BLUE, "exp43 374 tâches, tirage naturel, ancre 372 pas"),
      ("exp51_depthbal_uniform", ORANGE, "exp51 444 tâches, 1/4 par profondeur, ancre 444 pas"),
      ("exp51.1_depthbal_uniform_anchor372", AQUA, "exp51.1 idem exp51, ancre 372 pas")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for run, col, lab in S7:
    curve(axes[0], run, col, lab, steps=True)
    x, y = per_steps(run, "entropy"); axes[1].plot(x, y, color=col, lw=2, label=lab.split(" ")[0])
refs(axes[0]); axes[0].set_xlim(0, 2000); axes[0].set_ylim(0, 90)
axes[0].set_xlabel("pas d'optimisation"); axes[0].set_ylabel("Pass@1 test (%) — moy. glissante 5 évals")
axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
axes[0].set_title("Pass@1 : exp51.1 ne casse pas, mais monte moins vite", loc="left", fontsize=10, color=INK)
axes[1].axhline(0.15, color=INK2, lw=0.8, ls=(0, (4, 3)))
axes[1].set_xlim(0, 2000); axes[1].set_ylim(0, 1.8); axes[1].set_xlabel("pas d'optimisation")
axes[1].set_ylabel("entropie (médiane / 46 pas)"); axes[1].legend(loc="upper right", fontsize=8, frameon=False)
axes[1].set_title("Entropie : le tirage équilibré garde l'exploration haute", loc="left", fontsize=10, color=INK)
fig.suptitle("Seul delta exp51 → exp51.1 : ancre 444 → 372 pas. exp51.1 tient 2× plus longtemps qu'exp51 (coupé à 1 760 pas, sans collapse)",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig7_exp51_anchor.png"); plt.close(fig)
print("figures écrites dans", OUT)
