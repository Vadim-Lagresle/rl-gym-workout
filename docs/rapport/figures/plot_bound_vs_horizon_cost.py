"""Contrôle-G16 + borne k3 (exp43, 80 ép. complètes) contre les trois curriculums et la référence G=8,
sur les axes de COÛT (23/09/2026). Question : une fois le collapse évité par la borne, que reste-t-il au
curriculum ? Sources : logs/<run>.log — dict TRL par pas (epoch, num_tokens cumulés, step_time,
completions/mean_length, entropy), lignes [rollout] (n_active_turns par épisode), [test_eval] (pass@1).
Reprises : on garde la DERNIÈRE entrée par pas (les pas rejoués après reprise ne comptent qu'une fois).
Figure 1 : pass@1 en fonction de trajectoires / tokens / heures GPU / tours d'env.
Figure 2 : par époque, tours par épisode, tokens par pas, entropie.  Table : coût pour atteindre 70 et 75."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = {  # run: (couleur, style, largeur, label)
    "exp43_g16_klclamp10": ("#eb6834", "-", 1.7, "Control-G16 + KL bound, no curriculum (exp43)"),
    "exp32_horizon":       ("#a3405c", "-", 1.7, "Horizon curriculum (exp32)"),
    "exp35_budget1024":    ("#b8860b", "-", 1.0, "Budget curriculum (exp35)"),
    "exp33.1_depth_auto":  ("#7b52ab", "-", 1.0, "Depth curriculum (exp33.1)"),
    "exp25_r8_anchor4ep":  ("#0e9f8a", (0, (4, 2)), 1.0, "Moving-Anchor G=8, no curriculum (exp25)"),
}
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
eval_re = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100")
roll_re = re.compile(r"\[rollout\] n=\d+ .*?n_active_turns=(\[[^\]]*\])")

def load(run):
    rows, evals, pend = {}, {}, None
    for line in open(f"logs/{run}.log", errors="replace"):
        m = roll_re.search(line)
        if m: t = ast.literal_eval(m.group(1)); pend = (sum(t), len(t))
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "reward" not in d: continue
            s = int(m.group(1)); tu = pend if pend else (np.nan, 64); pend = None
            rows[s] = dict(ep=float(d["epoch"]), tok=float(d.get("num_tokens", "nan")), dt=float(d.get("step_time", "nan")),
                           ent=float(d.get("entropy", "nan")), rew=float(d["reward"]), turns=tu[0], n=tu[1],
                           clen=float(d.get("completions/mean_length", "nan")))
        m = eval_re.search(line)
        if m: evals[int(m.group(1))] = int(m.group(2))
    steps = np.array(sorted(rows)); g = lambda k: np.array([rows[s][k] for s in steps])
    A = dict(step=steps, ep=g("ep"), tok=g("tok") / 1e9, ent=g("ent"), rew=g("rew"), clen=g("clen"),
             turns_ep=g("turns") / g("n"), traj=np.cumsum(g("n")) / 1e3,
             hrs=np.cumsum(np.nan_to_num(g("dt"))) / 3600, cturns=np.cumsum(np.nan_to_num(g("turns"))) / 1e6)
    es = np.array([s for s in sorted(evals) if s in rows]); pv = np.array([evals[s] for s in es], float)
    pr = np.array([pv[max(0, i - 2):i + 1].mean() for i in range(len(pv))])   # lissage 3 évals
    idx = np.clip(np.searchsorted(steps, es), 0, len(steps) - 1)
    return A, es, pv, pr, idx

def per_epoch(ep, y, w=1.0):
    xs, ys = [], []
    for c in np.arange(0.5, ep[-1] + 1e-9, 0.5):
        m = (ep >= c - w / 2) & (ep < c + w / 2)
        if m.sum() >= 3: xs.append(c); ys.append(np.nanmean(y[m]))
    return np.array(xs), np.array(ys)

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
data = {run: load(run) for run in RUNS}

# ---------------- Figure 1 : pass@1 en fonction du coût
AX1 = [("traj", "trajectories (×1000)"), ("tok", "tokens processed (billions)"), ("hrs", "GPU hours"), ("cturns", "environment turns (millions)")]
fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.7), constrained_layout=True, sharey=True)
for run, (col, ls, lw, label) in RUNS.items():
    A, es, pv, pr, idx = data[run]
    for j, (k, lab) in enumerate(AX1):
        axes[j].plot(A[k][idx], pr, color=col, ls=ls, lw=lw, label=label if j == 0 else None)
for j, (k, lab) in enumerate(AX1):
    axes[j].set_xlabel(lab); axes[j].set_title(f"({chr(97 + j)}) per {lab.split(' (')[0]}", loc="left")
    axes[j].grid(axis="y", color=GRID, lw=0.6); axes[j].set_axisbelow(True)
    for s_ in ("top", "right"): axes[j].spines[s_].set_visible(False)
axes[0].set_ylim(0, 85); axes[0].set_ylabel("pass@1 test (mean of 3 evals)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.2, "Same ReLoRA recipe (r=8, anchor every 4 epochs); G=16 with 4 tasks/step for exp43 and the curricula, G=8 with 8 tasks/step for exp25. "
         "Costs summed over the training logs (resumed steps counted once).", ha="center", fontsize=6.3, color=MUTED)
out = "docs/rapport/figures/fig_bound_vs_horizon_cost"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")

# ---------------- Figure 2 : ce qui se passe par époque
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), constrained_layout=True)
for run, (col, ls, lw, label) in RUNS.items():
    A = data[run][0]
    x, y = per_epoch(A["ep"], A["turns_ep"]); axes[0].plot(x, y, color=col, ls=ls, lw=lw, label=label)
    x, y = per_epoch(A["ep"], A["clen"]);     axes[1].plot(x, y, color=col, ls=ls, lw=lw)
    x, y = per_epoch(A["ep"], A["ent"]);      axes[2].plot(x, y, color=col, ls=ls, lw=lw)
axes[0].set_ylabel("turns per training episode (mean)"); axes[0].set_title("(a) episode length in turns", loc="left"); axes[0].set_ylim(0, 31)
axes[1].set_ylabel("tokens per episode sequence (mean)"); axes[1].set_title("(b) episode length in tokens", loc="left")
axes[2].set_ylabel("policy entropy (mean, 1 ep.)"); axes[2].set_title("(c) entropy", loc="left"); axes[2].set_ylim(0, 1.3)
for e in (15, 30):
    axes[0].axvline(e, color="#a3405c", lw=0.7, alpha=0.4)
for ax in axes:
    ax.set_xlim(0, 80); ax.set_xlabel("epochs"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.2, "Token length = full training sequence of an episode (prompt excluded, actions + observations + template). Thin vertical lines: Horizon stage changes (10 → 20 → 30 turns).",
         ha="center", fontsize=6.3, color=MUTED)
out = "docs/rapport/figures/fig_bound_vs_horizon_dynamics"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")

# ---------------- Table
def first_reach(A, idx, pr, thr, k):
    j = np.argmax(pr >= thr) if (pr >= thr).any() else None
    return A[k][idx[j]] if j is not None else np.nan
print(f"{'run':22s} {'ep':>5s} {'traj k':>7s} {'tok G':>6s} {'hrs':>6s} {'turns M':>8s} | {'best':>4s} {'plat10':>6s} | cost to reach 70 (mean3): traj / tok / hrs / turns | to 75")
for run in RUNS:
    A, es, pv, pr, idx = data[run]
    c70 = [first_reach(A, idx, pr, 70, k) for k in ("traj", "tok", "hrs", "cturns")]
    c75 = [first_reach(A, idx, pr, 75, k) for k in ("traj", "tok", "hrs", "cturns")]
    print(f"{run:22s} {A['ep'][-1]:5.1f} {A['traj'][-1]:7.0f} {A['tok'][-1]:6.2f} {A['hrs'][-1]:6.1f} {A['cturns'][-1]:8.2f} | {pv.max():4.0f} {pv[-10:].mean():6.1f} | "
          + " / ".join(f"{v:.2f}" if v == v else "  —  " for v in c70) + " | " + " / ".join(f"{v:.2f}" if v == v else "  —  " for v in c75))
    # coût moyen par pas sur la 2e moitié
    h = A["ep"] > A["ep"][-1] / 2
    print(f"{'':22s} 2nd half: {np.nanmean(np.diff(A['hrs'][h]))*3600:5.1f} s/step, {np.nanmean(A['turns_ep'][h]):5.1f} turns/episode, {np.nanmean(A['clen'][h]):6.0f} tokens/episode, entropy {np.nanmean(A['ent'][h]):.3f}")
