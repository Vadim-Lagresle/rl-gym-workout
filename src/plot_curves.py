#!/usr/bin/env python3
"""
Parse un run.log de `verl.agent_trainer.main_ppo` et plot les courbes
clés en PNG.

Le trainer émet à chaque step une ligne du type
    step:N - key1:v1 - key2:v2 - ...
(préfixée par les logs Ray colorés `(main_task pid=X)`). On extrait
toutes les paires key:value flottantes pour chaque step et on plot les
8 métriques les plus utiles pour suivre un training GRPO multi-tour.

Usage :
    python scratch/plot_training_curves.py [LOG_PATH] [--out PNG] [--loss-out PNG]

Si LOG_PATH n'est pas donné, on prend le run.log le plus récent dans
saves/agentgym_rl_4gpu/. Par défaut deux PNG : vue d'ensemble + panneau
« losses » acteur (pg / kl / entropy / grad_norm).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"step:(\d+)(.*)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/]*):(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")

DEFAULT_METRICS = [
    ("critic/rewards/mean", "Reward (mean per batch)"),
    ("critic/task_score/mean", "Task score (= env reward 0/1)"),
    ("actor/pg_loss", "Policy gradient loss"),
    ("actor/kl_loss", "KL(policy || ref)"),
    ("actor/entropy_loss", "Entropy"),
    ("actor/grad_norm", "Grad norm"),
    ("response_length/mean", "Mean response length (tokens)"),
    ("critic/task_round/mean", "Mean episode rounds"),
]

# Ce que verl log comme « losses » côté acteur (pas de scalar unique « total loss »
# dans la ligne step: — c’est la somme PG + KL + entropie dans le code d’update).
LOSS_METRICS = [
    ("actor/pg_loss", "actor/pg_loss (clipped surrogate)"),
    ("actor/kl_loss", "actor/kl_loss (low-var KL term)"),
    ("actor/entropy_loss", "actor/entropy_loss (−entropy × coef)"),
    ("actor/grad_norm", "actor/grad_norm"),
]


def parse_log(path: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            try:
                step = int(m.group(1))
            except ValueError:
                continue
            tail = m.group(2)
            kvs = {k: float(v) for k, v in KV_RE.findall(tail)}
            if kvs:
                rows.setdefault(step, {}).update(kvs)
    return rows


def latest_run_log() -> str | None:
    candidates = sorted(
        glob.glob("saves/agentgym_rl_4gpu/agentgym_rl_qwen3b_4gpu_fixed_*/run.log"),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def plot_loss_panel(
    rows: dict[int, dict[str, float]],
    log_path: str,
    out_path: str,
) -> None:
    steps = sorted(rows.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, LOSS_METRICS):
        xs, ys = [], []
        for s in steps:
            if key in rows[s]:
                xs.append(s)
                ys.append(rows[s][key])
        if not xs:
            ax.set_title(f"{title}\n(absent)", fontsize=10)
            ax.axis("off")
            continue
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color="#1f77b4")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        last_y = ys[-1]
        ax.annotate(f"{last_y:.4g}", xy=(xs[-1], last_y),
                    xytext=(5, 0), textcoords="offset points", fontsize=8)
    run_name = os.path.basename(os.path.dirname(log_path))
    fig.suptitle(
        f"Actor losses — {run_name}\n"
        f"(verl ne log pas un « total loss » unique sur la ligne step:)",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default=None,
                        help="Path to run.log (default: latest in saves/agentgym_rl_4gpu/)")
    parser.add_argument("--out", default="runs/training_curves.png",
                        help="Output PNG path (overview 4×2)")
    parser.add_argument(
        "--loss-out",
        default="runs/training_losses.png",
        help="Output PNG for actor loss panel only (2×2). Use empty string to skip.",
    )
    args = parser.parse_args()

    log_path = args.log or latest_run_log()
    if not log_path or not os.path.exists(log_path):
        print("No run.log found", file=sys.stderr)
        return 1

    rows = parse_log(log_path)
    if not rows:
        print(f"No step metrics found in {log_path}", file=sys.stderr)
        return 1

    steps = sorted(rows.keys())
    print(f"[{log_path}] -> {len(steps)} steps parsed (range {steps[0]}..{steps[-1]})")

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, DEFAULT_METRICS):
        xs, ys = [], []
        for s in steps:
            if key in rows[s]:
                xs.append(s)
                ys.append(rows[s][key])
        if not xs:
            ax.set_title(f"{title}\n(metric '{key}' absent)", fontsize=10)
            ax.axis("off")
            continue
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        last_y = ys[-1]
        ax.annotate(f"{last_y:.4g}", xy=(xs[-1], last_y),
                    xytext=(5, 0), textcoords="offset points", fontsize=8)

    run_name = os.path.basename(os.path.dirname(log_path))
    fig.suptitle(f"Training curves — {run_name}\n(parsed from {log_path})",
                 fontsize=12)
    fig.tight_layout()
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")

    if args.loss_out:
        plot_loss_panel(rows, log_path, args.loss_out)

    print("\nLast metrics:")
    last = rows[steps[-1]]
    for key, title in DEFAULT_METRICS:
        v = last.get(key)
        if v is not None:
            print(f"  step {steps[-1]:>3}  {key:<28} = {v:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
