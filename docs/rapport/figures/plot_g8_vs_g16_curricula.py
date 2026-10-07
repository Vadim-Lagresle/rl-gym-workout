"""Variante avec les trois curriculums (16/09/2026) : G=8 (froid) vs G=16 sans curriculum (chaud) vs
curriculums G=16 (violets/vert). Sans les runs collapsés ni exp31. G=8 en teintes froides, G=16 en chaudes. Deux axes : époque (= une passe
sur les 374 tâches, donc 2× plus de trajectoires à G=16) et trajectoires cumulées (compute comparable).
Pass@1 = moyenne glissante sur 5 évaluations ; reward train = moyenne glissante sur 1 époque."""
import re, ast, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; DASH = (0, (4, 2))
RUNS = {  # run: (traj/pas, couleur, style, label)
    "exp25_r8_anchor4ep":       (64,  "#0e9f8a", DASH, "G=8 · 8 tâches, ancre /4 ép. (exp25)"),
    "exp41_g8_8tasks_anchor8":  (64,  "#2a78d6", DASH, "G=8 · 8 tâches, ancre /8 ép. (exp41)"),
    "exp39_g16_8tasks":         (128, "#1f5fb4", "-",  "G=16 · 8 tâches = 128 traj/pas, ancre /4 ép. (exp39)"),
    "exp43_g16_klclamp10":      (64,  "#eb6834", "-",  "G=16 · 4 tâches, k3 borné (exp43, en cours)"),
    "exp32_horizon":            (64,  "#a3405c", "-",  "Horizon · G=16, 4 tâches (exp32)"),
    "exp35_budget1024":         (64,  "#b8860b", "-",  "Budget · G=16, 4 tâches (exp35)"),
    "exp33.1_depth_auto":       (64,  "#7b52ab", "-",  "Profondeur · G=16, 4 tâches (exp33.1)"),
}
# reprises depuis le best du parent après purge (ancre repositionnée sur le best, Adam à zéro) : pointillé fin,
# décalées de l'époque et du nombre de pas atteints par le parent
CONT = {"exp33.1_depth_auto": ("exp33.2_from72", 44.53, 4141), "exp25_r8_anchor4ep": ("exp25.2_from65", 111.5, 5131)}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})"); eval_re = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100")
def load(run):
    rows, evals = {}, {}
    for line in open(f"logs/{run}.log", errors="replace"):
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "reward" in d: rows[int(m.group(1))] = (float(d["epoch"]), float(d["reward"]), float(d["kl"]))
        m = eval_re.search(line)
        if m: evals[int(m.group(1))] = int(m.group(2))
    steps = np.array(sorted(rows)); ep = np.array([rows[s][0] for s in steps]); rew = np.array([rows[s][1] for s in steps])
    kl = np.array([rows[s][2] for s in steps]); spe = steps[-1] / ep[-1]
    es = np.array(sorted(evals)); ev = np.array([evals[s] for s in es])
    return steps, ep, rew, kl, spe, es, ev
def roll(y, w):
    return np.array([np.mean(y[max(0, i - w + 1):i + 1]) for i in range(len(y))])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), constrained_layout=True)
for run, (tps, col, ls, label) in RUNS.items():
    steps, ep, rew, kl, spe, es, ev = load(run)
    # on coupe au collapse (KL médiane > 1 sur une demi-époque) pour ne pas tracer le régime dégénéré
    cut = None
    for e in np.arange(0, ep[-1], 0.5):
        m = (ep >= e) & (ep < e + 0.5)
        if m.sum() >= 3 and np.median(kl[m]) > 1: cut = e; break
    if cut is not None:
        keep_s = steps <= cut * spe; keep_e = es <= cut * spe
        steps, ep, rew, es, ev = steps[keep_s], ep[keep_s], rew[keep_s], es[keep_e], ev[keep_e]
    ev_r = roll(ev, 5); rew_r = roll(rew, int(spe))
    traj_e = es * tps / 1000; traj_s = steps * tps / 1000
    axes[0, 0].plot(es / spe, ev_r, color=col, ls=ls, lw=1.4, label=label)
    axes[0, 1].plot(traj_e, ev_r, color=col, ls=ls, lw=1.4)
    axes[1, 0].plot(ep, rew_r, color=col, ls=ls, lw=1.4)
    axes[1, 1].plot(traj_s, rew_r, color=col, ls=ls, lw=1.4)
    if cut is not None:
        for ax, x in ((axes[0, 0], es[-1] / spe), (axes[0, 1], traj_e[-1]), (axes[1, 0], ep[-1]), (axes[1, 1], traj_s[-1])):
            y = ev_r[-1] if ax in axes[0] else rew_r[-1]; ax.plot(x, y, "x", ms=6, color=col, mew=1.5)
    if run in CONT:
        child, off_ep, off_step = CONT[run]
        try: cs, cep, crew, ckl, cspe, ces, cev = load(child)
        except FileNotFoundError: continue
        cev_r = roll(cev, 5); crew_r = roll(crew, int(cspe)); dd = dict(color=col, ls=(0, (2, 2)), lw=1.0, alpha=0.85)
        axes[0, 0].plot(ces / cspe + off_ep, cev_r, **dd); axes[0, 1].plot((ces + off_step) * tps / 1000, cev_r, **dd)
        axes[1, 0].plot(cep + off_ep, crew_r, **dd);       axes[1, 1].plot((cs + off_step) * tps / 1000, crew_r, **dd)
for ax in axes[0]: ax.set_ylim(0, 85); ax.set_ylabel("pass@1 test (moy. gliss. 5 évals)")
for ax in axes[1]: ax.set_ylim(0, 0.85); ax.set_ylabel("reward train (moy. gliss. 1 ép.)")
for ax in axes[:, 0]: ax.set_xlim(0, 90); ax.set_xlabel("époque (1 passe sur les 374 tâches)")
for ax in axes[:, 1]: ax.set_xlim(0, 520); ax.set_xlabel("trajectoires cumulées (milliers) ≈ compute")
axes[0, 0].set_title("(a) pass@1 par époque", loc="left"); axes[0, 1].set_title("(b) pass@1 par trajectoires", loc="left")
axes[1, 0].set_title("(c) reward train par époque", loc="left"); axes[1, 1].set_title("(d) reward train par trajectoires", loc="left")
for ax in axes.flat:
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.text(0.5, -0.16, "Pointillé fin : reprise depuis le best après purge (Profondeur → exp33.2 à l'ép. 44,5 ; exp25 → exp25.2 à l'ép. 111).\nUne époque à G=16 = 2× les trajectoires et ≈ 2× le temps GPU d'une époque à G=8. \nReward train d'Horizon : épisodes plafonnés à 10 puis 20 tours jusqu'à l'ép. 30 (plus durs) ; Profondeur : recettes faciles d'abord (reward saturé au départ).",
         ha="center", fontsize=6.5, color=MUTED)
out = "docs/rapport/figures/fig_g8_vs_g16_curricula"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight"); print("ok")
