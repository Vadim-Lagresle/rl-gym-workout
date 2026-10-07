"""Reward train du curriculum Profondeur (exp33.1) sur les 10 premières époques, avec les paliers du
curriculum auto-déclenché (16/09/2026). Le pic à 0,8 des époques 0-2 = palier depth<=1 (109 tâches
faciles) presque saturé ; le passage à depth<=2 (330 tâches) à l'époque 2,14 fait retomber le reward."""
import re, ast, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})"); pal_re = re.compile(r">>> PALIER depth<=(\d) \(epoch ([\d.]+)")
ep, rew, pal = [], [], []
for line in open("logs/exp33.1_depth_auto.log", errors="replace"):
    for m in dict_re.finditer(line):
        try: d = ast.literal_eval(m.group(3))
        except Exception: continue
        if "reward" in d: ep.append(float(d["epoch"])); rew.append(float(d["reward"]))
    m = pal_re.search(line)
    if m: pal.append((int(m.group(1)), float(m.group(2))))
ep, rew = np.array(ep), np.array(rew); k = (ep <= 10)
roll = lambda y, w: np.array([np.mean(y[max(0, i - w + 1):i + 1]) for i in range(len(y))])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
fig, ax = plt.subplots(figsize=(7.4, 3.0), constrained_layout=True)
ax.plot(ep[k], rew[k], color="#7b52ab", lw=0.6, alpha=0.35, label="reward moyen du pas (64 trajectoires)")
ax.plot(ep[k], roll(rew, 20)[k], color="#7b52ab", lw=1.8, label="moyenne glissante 20 pas")
ax.axhline(0.8, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.text(9.95, 0.815, "seuil de passage 0,8 (reward moyen sur 1 époque)", ha="right", fontsize=7, color=MUTED)
stages = [(1, 0.0)] + pal
for i, (dpt, e0) in enumerate(stages):
    e1 = stages[i + 1][1] if i + 1 < len(stages) else 10
    if e0 > 10: break
    ax.axvspan(e0, min(e1, 10), color="#7b52ab", alpha=0.05 + 0.05 * i, lw=0)
    n = {1: 109, 2: 330, 3: 373, 4: 374}[dpt]
    ax.text((e0 + min(e1, 10)) / 2, 0.06, f"depth ≤ {dpt}\n{n} tâches", ha="center", fontsize=7.5, color=INK)
    if e0 > 0: ax.axvline(e0, color=MUTED, lw=0.8)
ax.set_xlim(0, 10); ax.set_ylim(0, 1.0); ax.set_xlabel("époque"); ax.set_ylabel("reward train")
ax.set_title("Curriculum Profondeur (exp33.1) : reward train et paliers auto-déclenchés, époques 0 à 10", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 0.12))
out = "docs/rapport/figures/fig_depth_reward_early"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight"); print("ok", pal)
