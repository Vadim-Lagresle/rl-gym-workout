"""pass@1 test et reward train en fonction de cinq mesures de coût : époques, trajectoires, tokens générés,
heures GPU (temps mur cumulé, reprises raccordées), tours d'environnement. Neuf runs. 17/09/2026.
Sources : logs TRL (dict par pas : epoch, reward, num_tokens cumulés ; barre tqdm : temps écoulé ;
lignes [rollout] : n_active_turns par épisode ; [test_eval] : pass@1)."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; DASH = (0, (4, 2))
RUNS = {  # run: (traj/pas, couleur, style, label)
    "exp23.1_verl_100ep":      (64,  "#1f5fb4", DASH, "Replication-FT (full-FT, G=8; exp23.1 segment)"),
    "exp25_r8_anchor4ep":      (64,  "#0e9f8a", DASH, "Moving-Anchor (ReLoRA, G=8, anchor /4)"),
    "exp41_g8_8tasks_anchor8": (64,  "#2a78d6", DASH, "ReLoRA G=8, anchor /8 (exp41)"),
    "exp39_g16_8tasks":        (128, "#3b3fa0", "-",  "ReLoRA G=16, 8 tasks, anchor /4 (exp39)"),
    "exp43_g16_klclamp10":     (64,  "#eb6834", "-",  "Control-G16 + KL bound (exp43)"),
    "exp36_n16":               (64,  "#e87ba4", (0, (2, 2)), "Control-G16 (collapse)"),
    "exp32_horizon":           (64,  "#a3405c", "-",  "Horizon"),
    "exp35_budget1024":        (64,  "#b8860b", "-",  "Budget"),
    "exp33.1_depth_auto":      (64,  "#7b52ab", "-",  "Depth"),
}
dict_re = re.compile(r"(\d+)/(\d+) \[([\d:]+)<[^\]]*\]\s*(\{'.*?\})")
eval_re = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100")
roll_re = re.compile(r"\[rollout\] n=\d+ .*?n_active_turns=(\[[^\]]*\])")
def hms(s):
    p = [int(x) for x in s.split(":")]; return (p[0] * 3600 + p[1] * 60 + p[2]) if len(p) == 3 else (p[0] * 60 + p[1])
def load(run, tps):
    rows, evals = {}, {}; cum_turns = 0; off = 0.0; last_el = 0.0
    for line in open(f"logs/{run}.log", errors="replace"):
        m = roll_re.search(line)
        if m: cum_turns += sum(ast.literal_eval(m.group(1)))
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(4))
            except Exception: continue
            if "reward" not in d: continue
            el = hms(m.group(3))
            if el < last_el - 60: off += last_el          # reprise : le compteur tqdm repart de zéro
            last_el = el; s = int(m.group(1))
            rows[s] = dict(ep=float(d["epoch"]), rew=float(d["reward"]), tok=float(d.get("num_tokens", "nan")),
                           hrs=(off + el) / 3600, traj=s * tps / 1e3, turns=cum_turns / 1e6)
        m = eval_re.search(line)
        if m: evals[int(m.group(1))] = int(m.group(2))
    steps = np.array(sorted(rows)); A = {k: np.array([rows[s][k] for s in steps]) for k in ("ep", "rew", "tok", "hrs", "traj", "turns")}
    A["tok"] = A["tok"] / 1e9; A["step"] = steps
    es = np.array([s for s in sorted(evals) if s in rows]); pv = np.array([evals[s] for s in es])
    return A, es, pv
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5})
AXES = [("ep", "epochs", 80), ("traj", "trajectories (×1000)", 520), ("tok", "tokens (billions)", None), ("hrs", "GPU hours", None), ("turns", "env. turns (millions)", None)]
fig, axes = plt.subplots(2, 5, figsize=(7.8, 3.9), constrained_layout=True, sharey="row")
for run, (tps, col, ls, label) in RUNS.items():
    A, es, pv = load(run, tps); spe = A["step"][-1] / A["ep"][-1]
    # collapse : on coupe à la 1re demi-époque de KL médiane > 1 (lue via reward ? non : via une passe KL séparée) → on coupe simplement exp36 à l'ép. 10.5
    keep = A["ep"] <= (10.5 if run == "exp36_n16" else 1e9)
    pr = np.array([pv[max(0, i - 2):i + 1].mean() for i in range(len(pv))]); rr = np.array([A["rew"][max(0, i - int(spe) + 1):i + 1].mean() for i in range(len(A["rew"]))])
    idx = np.searchsorted(A["step"], es); idx = np.clip(idx, 0, len(A["step"]) - 1)
    for j, (key, lab, xmax) in enumerate(AXES):
        xe = A[key][idx]; ke = A["ep"][idx] <= (10.5 if run == "exp36_n16" else 1e9)
        axes[0, j].plot(xe[ke], pr[ke], color=col, ls=ls, lw=1.2, label=label if j == 0 else None)
        axes[1, j].plot(A[key][keep], rr[keep], color=col, ls=ls, lw=1.2)
for j, (key, lab, xmax) in enumerate(AXES):
    axes[0, j].set_title(f"({chr(97+j)}) per {lab.split(' (')[0]}", loc="left"); axes[1, j].set_xlabel(lab)
    if xmax: axes[0, j].set_xlim(0, xmax); axes[1, j].set_xlim(0, xmax)
axes[0, 0].set_ylabel("pass@1 test (mean of 3 evals)"); axes[1, 0].set_ylabel("train reward (rolling 1 epoch)")
axes[0, 0].set_ylim(0, 85); axes[1, 0].set_ylim(0, 0.9)
for ax in axes.flat:
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
fig.text(0.5, -0.19, "GPU hours: wall clock from the training logs, resumes chained; the full-FT run predates the co-located inference engine. Environment turns: summed over rollouts.", ha="center", fontsize=6.3, color=MUTED)
out = "docs/rapport/figures/fig_compute_axes"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); print("ok")
for run, (tps, *_ ) in RUNS.items():
    A, es, pv = load(run, tps); print(f"{run:26s} ep {A['ep'][-1]:6.1f} | traj {A['traj'][-1]:6.0f}k | tok {A['tok'][-1]:5.2f}G | hrs {A['hrs'][-1]:6.1f} | turns {A['turns'][-1]:5.2f}M")
