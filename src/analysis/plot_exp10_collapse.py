#!/usr/bin/env python
"""exp10 (GRPO LoRA B200) — anatomie de l'effondrement.

Parse logs/exp10_grpo_lora_b200.log et trace, sur un axe step commun :
  (A) reward train + pass@1 test       (B) KL vs reference (echelle log)
  (C) entropy                          (D) longueur moyenne de completion

Met en evidence : pic eval (step 92), divergence KL/grad (~step 200),
etat mort degenere (reward_std=0 partout, step >= 230).

Sortie : docs/dashboard/10_exp10_collapse.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / "exp10_grpo_lora_b200.log"
OUT = REPO / "docs" / "dashboard" / "10_exp10_collapse.png"

KEYS = ["reward", "kl", "entropy", "completions/mean_length"]


def parse():
    txt = LOG.read_text(encoding="utf-8", errors="ignore")
    steps = {k: [] for k in KEYS}
    xs = []
    for i, m in enumerate(re.finditer(r"\{[^{}]*'reward':[^{}]*\}", txt), start=1):
        d = m.group(0)
        xs.append(i)
        for k in KEYS:
            mm = re.search(r"'%s': '?([0-9.eE+-]+)'?" % re.escape(k), d)
            steps[k].append(float(mm.group(1)) if mm else float("nan"))
    evals = [(int(s), int(p)) for s, p in
             re.findall(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/100", txt)]
    return xs, steps, evals


def main():
    xs, s, evals = parse()
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    # bandes de phases
    def bands(ax):
        ax.axvspan(0, 92, color="#2ca02c", alpha=0.06)
        ax.axvspan(92, 180, color="#ff7f0e", alpha=0.08)
        ax.axvspan(180, 230, color="#d62728", alpha=0.10)
        ax.axvspan(230, max(xs), color="#7f0000", alpha=0.12)

    # (A) reward + pass@1
    ax = axes[0]
    bands(ax)
    ax.plot(xs, s["reward"], color="#1f77b4", lw=1.2, label="reward train")
    ax.set_ylabel("reward train", color="#1f77b4")
    ax.set_ylim(-0.02, 1.0)
    ax2 = ax.twinx()
    if evals:
        ex, ey = zip(*evals)
        ax2.plot(ex, ey, "o-", color="#d62728", lw=1.8, ms=6, label="pass@1 test")
        best = max(evals, key=lambda t: t[1])
        ax2.annotate(f"best {best[1]}/100\n(step {best[0]})", xy=best,
                     xytext=(best[0] + 15, best[1] + 6), color="#d62728",
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax2.set_ylabel("pass@1 test (/100)", color="#d62728")
    ax2.set_ylim(-1, 35)
    ax.set_title("exp10 GRPO LoRA — anatomie de l'effondrement (LR=1.5e-5, beta=0.001)", fontsize=12)

    # (B) KL log
    ax = axes[1]
    bands(ax)
    ax.plot(xs, s["kl"], color="#9467bd", lw=1.2)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("KL vs ref (symlog)")
    ax.axhline(1.0, color="grey", ls=":", lw=1)

    # (C) entropy
    ax = axes[2]
    bands(ax)
    ax.plot(xs, s["entropy"], color="#8c564b", lw=1.2)
    ax.set_ylabel("entropy")

    # (D) longueur
    ax = axes[3]
    bands(ax)
    ax.plot(xs, s["completions/mean_length"], color="#e377c2", lw=1.2)
    ax.set_ylabel("longueur moy.\ncompletion (tokens)")
    ax.set_xlabel("step")

    # legende des phases
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#2ca02c", alpha=0.3, label="1. apprentissage sain (->step 92, pic eval 27)"),
        Patch(color="#ff7f0e", alpha=0.3, label="2. effondrement entropie (92-180)"),
        Patch(color="#d62728", alpha=0.3, label="3. divergence KL/grad (180-230)"),
        Patch(color="#7f0000", alpha=0.3, label="4. etat mort (reward_std=0, >=230)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[plot] ecrit {OUT} ({OUT.stat().st_size} octets) — {len(xs)} steps, {len(evals)} evals")


if __name__ == "__main__":
    main()
