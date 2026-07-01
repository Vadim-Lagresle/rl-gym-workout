"""Dashboard comparatif des familles d'expériences RL sur TextCraft.

Compare trois familles d'approches (axe temporel = EPOCH, comparable entre runs) :
  (a) GRPO pur sans curriculum  — exp7.3 (full-ft) : courbe eval successive we5
                                   (19,21,26,27,23,29,25) + best 32 (oracle).
  (b) ScalingInter (max_rounds progressif) — exp8.1, exp8.2 (exp8.2 = batch papier).
  (c) Curriculum par depth (staged d1→d2→d3→d3+4) — exp9.

NB : exp10 (LoRA) est EXCLU — encore en entraînement, courbes non figées.

Lecture SEULE des données d'expérience. Le script n'écrit QUE :
  - les PNG dans docs/dashboard/,
  - runs/exp9_curriculum_depth_final/analysis.txt (via analyze_eval).

Axe des abscisses = EPOCH (et non step : grad_accum diffère entre runs) :
  - reward train : le champ 'epoch' de chaque step-dict TRL est utilisé directement ;
    exp9 = 4 stages dont 'epoch' RESET → epochs cumulés (offsets 0,3,7,9).
  - pass@1 eval : les lignes [test_eval] ne donnent qu'un step global → conversion
    step→epoch par interpolation sur les step-dicts du MÊME run.
  - exp7.3 : pas de log local (wandb perdu) → 7 evals we5 placés uniformément sur
    epoch ≈ 8.5 → 21 (APPROXIMATION, reconstruite depuis docs/hebdo).

Usage :
    /home/criteo/envs/agentgym-rl/bin/python src/analysis/plot_training_dashboard.py
"""
from __future__ import annotations

import ast
import glob
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "analysis"))
from analyze_eval import (  # noqa: E402
    analyze_episode,
    load_depth_map,
    ERROR_PATTERNS,
)
from compare_dashboard import ERROR_COLORS  # noqa: E402

# ── Configuration : expérience → chemins de données ───────────────────────────
WANDB = REPO / "wandb"
LOGS = REPO / "logs"
RUNS_DIR = REPO / "runs"
OUTDIR = REPO / "docs" / "dashboard"

DEPTHS = [1, 2, 3, 4]
ETYPES = [e[0] for e in ERROR_PATTERNS]

# Lignes de référence
BASELINE_PASS1 = 18          # Qwen2.5-3B, 0 training
EXP73_BEST = 32              # exp7.3 GRPO pur full-ft, meilleur (we6 oracle, docs/hebdo)

# exp7.3 : courbe eval successive du run we5 (ckpt400→~ckpt1000), 7 checkpoints.
# Pas de log local → epochs APPROXIMÉS : grad_accum=8, N=8 → ~47 steps/epoch,
# ckpt400 ≈ epoch 8.5, ckpt1000 ≈ epoch 21 → 7 points uniformes sur [8.5, 21].
EXP73_WE5_PASS1 = [19, 21, 26, 27, 23, 29, 25]
EXP73_EPOCH_RANGE = (8.5, 21.0)

# exp9 : durée (en epochs) de chaque stage du curriculum depth → offsets cumulés.
EXP9_STAGE_EPOCHS = [3, 4, 2, 1]  # d1, d2, d3, d3+4  (total 10)
EXP9_FINAL_EPOCH = sum(EXP9_STAGE_EPOCHS)  # 10

# Couleur dédiée et stable par expérience sur tous les graphes
EXP_COLORS = {
    "exp7.3": "#9c755f",   # brun  — GRPO pur full-ft (référence)
    "exp8.1": "#f28e2b",   # orange — ScalingInter
    "exp8.2": "#e15759",   # rouge — ScalingInter (papier)
    "exp9":   "#59a14f",   # vert  — curriculum depth
}

EXP = {
    "exp8.1": {
        "label": "exp8.1 ScalingInter",
        "wandb": WANDB / "run-20260622_134137-otj4lzbl" / "files",
    },
    "exp8.2": {
        "label": "exp8.2 ScalingInter (papier)",
        "wandb": WANDB / "run-20260623_181832-u083ibg4" / "files",
    },
    "exp9": {
        "label": "exp9 curriculum depth",
        "log": LOGS / "exp9_curriculum_depth.log",
        "eval_logs": RUNS_DIR / "exp9_curriculum_depth_final" / "eval_logs",
        "run_dir": RUNS_DIR / "exp9_curriculum_depth_final",
    },
    "exp7.3": {
        "label": "exp7.3 GRPO pur full-ft",
        "eval_logs": RUNS_DIR / "exp7.3_ckpt400" / "eval_logs",
    },
}

# Clés numériques extraites de chaque step-dict TRL
DICT_KEYS = ["reward", "reward_std", "frac_reward_zero_std", "entropy",
             "completions/mean_length", "epoch"]

# Un step-dict TRL est un dict python (single quotes, valeurs str), sans accolade
# imbriquée → on capture la plus petite accolade contenant 'reward'.
DICT_RE = re.compile(r"\{[^{}]*'reward'[^{}]*\}")
STAGE_RE = re.compile(r"=== Stage (\S+) : depth=([\d,]+)")
EVAL_RE = re.compile(r"step (\d+) [—\-] Pass@1 = (\d+)/100")


def warn(msg: str) -> None:
    print(f"[dashboard][WARN] {msg}")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_from_dict(d: dict) -> dict:
    return {k: _f(d.get(k)) for k in DICT_KEYS}


# ── Parsing des logs ──────────────────────────────────────────────────────────

def parse_reward_dicts(path: Path) -> list[dict]:
    """Retourne la liste ordonnée des step-dicts (un par logging step)."""
    rows: list[dict] = []
    if not path or not path.exists():
        warn(f"log introuvable : {path}")
        return rows
    text = path.read_text(errors="replace")
    for m in DICT_RE.finditer(text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and "reward" in d:
            rows.append(_row_from_dict(d))
    # step implicite = index (logging_steps=1) ; x temporel = epoch (champ TRL)
    for i, r in enumerate(rows, start=1):
        r["step"] = i
        r["x"] = r["epoch"]  # axe = epoch
    return rows


def parse_eval_curve(path: Path) -> list[tuple[int, int]]:
    """Retourne [(step, pass@1 /100), ...] trié, dédupliqué."""
    pts: list[tuple[int, int]] = []
    if not path or not path.exists():
        warn(f"log eval introuvable : {path}")
        return pts
    for m in EVAL_RE.finditer(path.read_text(errors="replace")):
        pts.append((int(m.group(1)), int(m.group(2))))
    return sorted(dict(pts).items())


def parse_exp9_stages(path: Path) -> list[dict]:
    """Parse les 4 stages séquentiels d'exp9 depuis son log unique."""
    stages: list[dict] = []
    if not path or not path.exists():
        warn(f"log exp9 introuvable : {path}")
        return stages
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        sm = STAGE_RE.search(line)
        if sm:
            cur = {"name": sm.group(1), "depth": sm.group(2).strip(" ,"), "rows": []}
            stages.append(cur)
            continue
        if cur is None:
            continue
        dm = DICT_RE.search(line)
        if dm:
            try:
                d = ast.literal_eval(dm.group(0))
            except (ValueError, SyntaxError):
                continue
            if isinstance(d, dict) and "reward" in d:
                cur["rows"].append(_row_from_dict(d))
    return stages


def stitch_stages(stages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Concatène les stages en EPOCHS cumulés. Retourne (rows, boundaries).

    Chaque stage a son 'epoch' qui RESET (0→durée du stage). On ajoute un offset
    cumulé (0, 3, 7, 9) basé sur EXP9_STAGE_EPOCHS.
    boundaries = [{epoch, depth, name}] = epoch de DÉBUT de chaque stage.
    """
    rows: list[dict] = []
    boundaries: list[dict] = []
    offset = 0.0
    for i, st in enumerate(stages):
        planned = EXP9_STAGE_EPOCHS[i] if i < len(EXP9_STAGE_EPOCHS) else (
            max((r["epoch"] or 0.0) for r in st["rows"]) if st["rows"] else 0.0)
        boundaries.append({"epoch": offset, "depth": st["depth"], "name": st["name"]})
        for r in st["rows"]:
            rr = dict(r)
            rr["x"] = offset + (r["epoch"] or 0.0)  # epoch cumulé
            rows.append(rr)
        offset += planned
    return rows, boundaries


# ── Lecture des summaries wandb ───────────────────────────────────────────────

def load_summary(files_dir: Path) -> dict | None:
    if not files_dir:
        return None
    p = files_dir / "wandb-summary.json"
    if not p.exists():
        warn(f"summary absent (run probablement en cours) : {p}")
        return None
    try:
        return json.load(p.open())
    except (json.JSONDecodeError, OSError) as e:
        warn(f"summary illisible {p}: {e}")
        return None


def summary_perdepth_pass1(summary: dict) -> dict:
    out = {}
    for d in DEPTHS:
        v = summary.get(f"train/eval/pass1_d{d}")
        out[d] = 100 * v if v is not None else None
    return out


def summary_pertype_err(summary: dict) -> dict:
    return {et: summary.get(f"train/eval/err_{et}_per_ep") for et in ETYPES}


# ── Lecture des eval_logs post-hoc ────────────────────────────────────────────

def load_eval_logs(eval_dir: Path) -> list[dict]:
    if not eval_dir or not eval_dir.exists():
        warn(f"eval_logs introuvable : {eval_dir}")
        return []
    paths = glob.glob(str(eval_dir / "*.json"))
    return [analyze_episode(Path(p)) for p in paths]


def evallogs_perdepth_pass1(res: list[dict], depth_map: dict) -> dict:
    out = {}
    for d in DEPTHS:
        items = [r for r in res if depth_map.get(r["item_id"]) == d]
        out[d] = 100 * sum(1 for r in items if r["reward"] > 0) / len(items) if items else None
    return out


def evallogs_overall_pass1(res: list[dict]) -> float | None:
    if not res:
        return None
    return 100 * sum(1 for r in res if r["reward"] > 0) / len(res)


def evallogs_pertype_err(res: list[dict]) -> dict:
    if not res:
        return {et: None for et in ETYPES}
    n = len(res)
    counts: Counter = Counter()
    for r in res:
        for et, c in r["error_counts"].items():
            counts[et] += c
    return {et: counts.get(et, 0) / n for et in ETYPES}


def evallogs_pertype_by_depth(res: list[dict], depth_map: dict) -> dict:
    out = {}
    for d in DEPTHS:
        items = [r for r in res if depth_map.get(r["item_id"]) == d]
        if not items:
            out[d] = {et: 0.0 for et in ETYPES}
            continue
        counts: Counter = Counter()
        for r in items:
            for et, c in r["error_counts"].items():
                counts[et] += c
        out[d] = {et: counts.get(et, 0) / len(items) for et in ETYPES}
    return out


# ── analyse exp9 (analysis.txt manquant) ──────────────────────────────────────

def write_exp9_analysis() -> None:
    eval_dir = EXP["exp9"]["eval_logs"]
    if not eval_dir.exists():
        warn(f"exp9 eval_logs absent, analysis.txt non généré : {eval_dir}")
        return
    cmd = [sys.executable, str(REPO / "src" / "analysis" / "analyze_eval.py"),
           "--eval-dir", str(eval_dir)]
    print(f"[dashboard] génération analysis.txt exp9 : {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[dashboard]   → {EXP['exp9']['run_dir'] / 'analysis.txt'}")
    except subprocess.CalledProcessError as e:
        warn(f"analyze_eval exp9 a échoué : {e.stderr[-500:] if e.stderr else e}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _depth_label(depth: str) -> str:
    return "d" + depth.replace(",", "+")


def make_plots(D: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUTDIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUTDIR / name, dpi=130)
        plt.close(fig)
        written.append(name)

    def eval_to_epoch(pts, rows):
        """Convertit [(step, pass1)] → [(epoch, pass1)] via interpolation sur rows."""
        if not pts:
            return []
        if not rows:
            return [(float(s), p) for s, p in pts]
        xp = np.array([r["step"] for r in rows], dtype=float)
        fp = np.array([r["epoch"] if r["epoch"] is not None else np.nan
                       for r in rows], dtype=float)
        return [(float(np.interp(s, xp, fp)), p) for s, p in pts]

    def draw_exp9_boundaries(ax, ymax_text=None):
        c = EXP_COLORS["exp9"]
        for b in D["exp9"]["boundaries"]:
            if b["epoch"] > 0:
                ax.axvline(b["epoch"], color=c, ls="--", lw=1, alpha=0.6)
            if ymax_text is not None:
                ax.text(b["epoch"] + 0.05, ymax_text, _depth_label(b["depth"]),
                        color=c, fontsize=8, ha="left", va="bottom")

    # ── 1) Reward train vs EPOCH ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in ("exp8.1", "exp8.2"):
        rows = D[key]["reward_rows"]
        if not rows:
            continue
        x = np.array([r["x"] for r in rows], dtype=float)
        y = np.array([r["reward"] for r in rows], dtype=float)
        sd = np.array([r["reward_std"] or 0.0 for r in rows], dtype=float)
        c = EXP_COLORS[key]
        ax.plot(x, y, "-", color=c, lw=2, label=EXP[key]["label"])
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.10)
    rows9 = D["exp9"]["reward_rows"]
    if rows9:
        x = np.array([r["x"] for r in rows9], dtype=float)
        y = np.array([r["reward"] for r in rows9], dtype=float)
        sd = np.array([r["reward_std"] or 0.0 for r in rows9], dtype=float)
        c = EXP_COLORS["exp9"]
        ax.plot(x, y, "-o", color=c, lw=2, ms=3, label=EXP["exp9"]["label"])
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.10)
        draw_exp9_boundaries(ax, ymax_text=max(y.max(), 0.4) * 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Reward moyen (bande = ±reward_std)")
    ax.set_title("① Reward d'entraînement vs epoch — ScalingInter vs curriculum depth\n"
                 "(exp9 : epochs cumulés sur les 4 stages ; lignes = changements de depth)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "01_reward_train_vs_epoch.png")

    # ── 2) Pass@1 eval vs EPOCH ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in ("exp8.1", "exp8.2"):
        pts = eval_to_epoch(D[key]["eval_curve"], D[key]["reward_rows"])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = EXP_COLORS[key]
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=5, label=EXP[key]["label"])
        bi = int(np.argmax(ys))
        ax.scatter([xs[bi]], [ys[bi]], s=120, facecolors="none",
                   edgecolors=c, linewidths=2, zorder=5)
        ax.annotate(f"best {ys[bi]}", (xs[bi], ys[bi]), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, color=c, ha="center")
    # exp7.3 : 7 evals we5 placés uniformément sur [8.5, 21] (APPROX)
    e0, e1 = EXP73_EPOCH_RANGE
    xs73 = list(np.linspace(e0, e1, len(EXP73_WE5_PASS1)))
    ys73 = EXP73_WE5_PASS1
    c73 = EXP_COLORS["exp7.3"]
    ax.plot(xs73, ys73, "-s", color=c73, lw=2, ms=5,
            label="exp7.3 GRPO pur full-ft (we5, epochs approx.)")
    bi = int(np.argmax(ys73))
    ax.scatter([xs73[bi]], [ys73[bi]], s=120, facecolors="none",
               edgecolors=c73, linewidths=2, zorder=5)
    ax.annotate(f"best {ys73[bi]}", (xs73[bi], ys73[bi]), textcoords="offset points",
                xytext=(0, 8), fontsize=8, color=c73, ha="center")
    # exp9 : point final unique (epoch 10)
    if D["exp9"]["overall_pass1"] is not None:
        v9 = D["exp9"]["overall_pass1"]
        ax.scatter([EXP9_FINAL_EPOCH], [v9], marker="*", s=240,
                   color=EXP_COLORS["exp9"], zorder=6,
                   label=f"exp9 curriculum depth (final {v9:.0f})")
    # références horizontales
    ax.axhline(EXP73_BEST, color=c73, ls="--", lw=1.5,
               label=f"exp7.3 GRPO pur full-ft — best {EXP73_BEST}")
    ax.axhline(BASELINE_PASS1, color="gray", ls=":", lw=1.5,
               label=f"baseline non entraîné ({BASELINE_PASS1})")
    ax.text(0.01, 0.98,
            "exp7.3 : epochs reconstruits depuis docs/hebdo (wandb perdu) — approximatif",
            transform=ax.transAxes, fontsize=7, color=c73, ha="left", va="top",
            style="italic")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Pass@1 test /100")
    ax.set_title("② Pass@1 test vs epoch — GRPO pur (exp7.3) vs curricula (exp8.x, exp9)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "02_pass1_eval_vs_epoch.png")

    # ── 3) Pass@1 par depth (barres groupées) — CENTERPIECE ──────────────────
    series = []  # (key, label, {depth: %}) — ordre : GRPO pur, puis curricula
    if D["exp7.3"]["perdepth_pass1"]:
        series.append(("exp7.3", "exp7.3 GRPO pur (full-ft)", D["exp7.3"]["perdepth_pass1"]))
    if D["exp8.2"]["perdepth_pass1"]:
        series.append(("exp8.2", "exp8.2 ScalingInter", D["exp8.2"]["perdepth_pass1"]))
    if D["exp9"]["perdepth_pass1"]:
        series.append(("exp9", "exp9 curriculum depth", D["exp9"]["perdepth_pass1"]))
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(DEPTHS))
    n = max(len(series), 1)
    width = 0.8 / n
    for j, (key, label, pd) in enumerate(series):
        vals = [pd.get(d) if pd.get(d) is not None else 0 for d in DEPTHS]
        bars = ax.bar(x + (j - (n - 1) / 2) * width, vals, width,
                      label=label, color=EXP_COLORS[key])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth {d}" for d in DEPTHS])
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("③ Pass@1 par profondeur — les curricula n'améliorent PAS le GRPO pur\n"
                 "ScalingInter (exp8.2) & curriculum depth (exp9) ≤ GRPO pur (exp7.3) ; "
                 "effondrement à 0 en depth 3–4")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "03_pass1_par_depth.png")

    # ── 4) Erreurs par type (barres groupées) ────────────────────────────────
    series = []
    for key in ("exp7.3", "exp8.1", "exp8.2", "exp9"):
        pe = D[key].get("pertype_err")
        if pe and any(v is not None for v in pe.values()):
            series.append((key, EXP[key]["label"], pe))
    active = [et for et in ETYPES if any((s[2].get(et) or 0) > 0 for s in series)]
    if series and active:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(active))
        n = max(len(series), 1)
        width = 0.8 / n
        for j, (key, label, pe) in enumerate(series):
            vals = [pe.get(et) or 0 for et in active]
            ax.bar(x + (j - (n - 1) / 2) * width, vals, width,
                   label=label, color=EXP_COLORS[key])
        ax.set_xticks(x)
        ax.set_xticklabels(active, rotation=20, ha="right")
        ax.set_ylabel("Erreurs par épisode")
        ax.set_title("④ Erreurs par type, par expérience (eval test set)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "04_erreurs_par_type.png")
    else:
        warn("plot 4 sauté : aucune donnée d'erreur par type")

    # ── 5) frac_reward_zero_std vs EPOCH ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for key in ("exp8.1", "exp8.2", "exp9"):
        rows = D[key]["reward_rows"]
        if not rows:
            continue
        x = np.array([r["x"] for r in rows], dtype=float)
        y = np.array([r["frac_reward_zero_std"] if r["frac_reward_zero_std"] is not None
                      else np.nan for r in rows], dtype=float)
        ax.plot(x, y, "-", color=EXP_COLORS[key], lw=2, label=EXP[key]["label"])
        plotted = True
    if D["exp9"]["reward_rows"]:
        draw_exp9_boundaries(ax)
    ax.set_xlabel("epoch")
    ax.set_ylabel("frac_reward_zero_std")
    ax.set_title("⑤ Fraction de groupes GRPO sans signal (reward std=0) — goulot sparse-reward")
    ax.set_ylim(0, 1.02)
    if plotted:
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "05_frac_reward_zero_std.png")

    # ── 6) Erreurs par type × depth (exp9 ; exp8.2 non décomposable) ──────────
    pdbt = D["exp9"].get("pertype_by_depth")
    if pdbt:
        active6 = [et for et in ETYPES if any(pdbt[d].get(et, 0) > 0 for d in DEPTHS)]
        fig, ax = plt.subplots(figsize=(11, 6))
        bottoms = np.zeros(len(DEPTHS))
        xd = np.arange(len(DEPTHS))
        for et in active6:
            vals = np.array([pdbt[d].get(et, 0) for d in DEPTHS])
            ax.bar(xd, vals, bottom=bottoms, label=et,
                   color=ERROR_COLORS.get(et, "#333333"), edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_xticks(xd)
        ax.set_xticklabels([f"depth {d}" for d in DEPTHS])
        ax.set_ylabel("Erreurs par épisode (empilées par type)")
        ax.set_title("⑥ Types d'erreur par profondeur — exp9 curriculum depth\n"
                     "(exp8.2 : décomposition type×depth non récupérable depuis le summary)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "06_erreurs_par_depth.png")
    else:
        warn("exp9 pertype_by_depth absent — plot 6 sauté")

    # ── 7) best vs final pass@1 ──────────────────────────────────────────────
    bf = []  # (key, label, best, final)
    for key in ("exp8.1", "exp8.2"):
        pts = D[key]["eval_curve"]
        if not pts:
            continue
        ys = [p[1] for p in pts]
        bf.append((key, EXP[key]["label"], max(ys), ys[-1]))
    # exp7.3 : best oracle 32 vs dernier eval we5 (25)
    bf.append(("exp7.3", "exp7.3 GRPO pur full-ft", EXP73_BEST, EXP73_WE5_PASS1[-1]))
    if D["exp9"]["overall_pass1"] is not None:
        v = round(D["exp9"]["overall_pass1"])
        bf.append(("exp9", EXP["exp9"]["label"], v, v))
    if bf:
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(bf))
        width = 0.38
        b1 = ax.bar(x - width / 2, [b[2] for b in bf], width, label="best",
                    color=[EXP_COLORS[b[0]] for b in bf])
        b2 = ax.bar(x + width / 2, [b[3] for b in bf], width, label="final / dernier",
                    color=[EXP_COLORS[b[0]] for b in bf], alpha=0.5, hatch="//")
        ax.bar_label(b1, fmt="%.0f", fontsize=8)
        ax.bar_label(b2, fmt="%.0f", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([b[1] for b in bf], rotation=12, ha="right", fontsize=8)
        ax.set_ylabel("Pass@1 /100")
        ax.set_title("⑦ Best vs final — importance du best-checkpoint tracking\n"
                     "(exp7.3 best = 32 oracle we6 ; final = dernier eval we5)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "07_best_vs_final_pass1.png")
    else:
        warn("aucune courbe eval — plot 7 sauté")

    # ── 8) Completion length & entropy vs EPOCH ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    plotted = False
    for key in ("exp8.1", "exp8.2", "exp9"):
        rows = D[key]["reward_rows"]
        if not rows:
            continue
        x = np.array([r["x"] for r in rows], dtype=float)
        ml = np.array([r["completions/mean_length"] if r["completions/mean_length"] is not None
                       else np.nan for r in rows], dtype=float)
        en = np.array([r["entropy"] if r["entropy"] is not None else np.nan for r in rows],
                      dtype=float)
        ax1.plot(x, ml, "-", color=EXP_COLORS[key], lw=2, label=EXP[key]["label"])
        ax2.plot(x, en, "-", color=EXP_COLORS[key], lw=2, label=EXP[key]["label"])
        plotted = True
    if D["exp9"]["reward_rows"]:
        draw_exp9_boundaries(ax1)
        draw_exp9_boundaries(ax2)
    ax1.set_ylabel("completions/mean_length (tokens)")
    ax1.set_title("⑧ Longueur de complétion (haut) & entropie (bas) vs epoch — collapse / blow-up")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("entropy")
    if plotted:
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    save(fig, "08_completion_len_entropy.png")

    return written


# ── Chargement global ─────────────────────────────────────────────────────────

def load_all() -> dict:
    depth_map = load_depth_map(REPO)
    if not depth_map:
        warn("depth map vide — pass@1 par depth indisponible")

    D: dict = {k: {} for k in EXP}

    # exp8.1
    s = load_summary(EXP["exp8.1"]["wandb"])
    D["exp8.1"]["reward_rows"] = parse_reward_dicts(EXP["exp8.1"]["wandb"] / "output.log")
    D["exp8.1"]["eval_curve"] = parse_eval_curve(EXP["exp8.1"]["wandb"] / "output.log")
    D["exp8.1"]["perdepth_pass1"] = summary_perdepth_pass1(s) if s else None
    D["exp8.1"]["pertype_err"] = summary_pertype_err(s) if s else None

    # exp8.2
    s = load_summary(EXP["exp8.2"]["wandb"])
    D["exp8.2"]["reward_rows"] = parse_reward_dicts(EXP["exp8.2"]["wandb"] / "output.log")
    D["exp8.2"]["eval_curve"] = parse_eval_curve(EXP["exp8.2"]["wandb"] / "output.log")
    D["exp8.2"]["perdepth_pass1"] = summary_perdepth_pass1(s) if s else None
    D["exp8.2"]["pertype_err"] = summary_pertype_err(s) if s else None

    # exp9 (logs + eval_logs post-hoc)
    stages = parse_exp9_stages(EXP["exp9"]["log"])
    rows9, bounds9 = stitch_stages(stages)
    D["exp9"]["reward_rows"] = rows9
    D["exp9"]["boundaries"] = bounds9
    D["exp9"]["eval_curve"] = []  # pas d'eval pendant l'entraînement
    res9 = load_eval_logs(EXP["exp9"]["eval_logs"])
    D["exp9"]["perdepth_pass1"] = evallogs_perdepth_pass1(res9, depth_map) if res9 else None
    D["exp9"]["overall_pass1"] = evallogs_overall_pass1(res9)
    D["exp9"]["pertype_err"] = evallogs_pertype_err(res9) if res9 else None
    D["exp9"]["pertype_by_depth"] = evallogs_pertype_by_depth(res9, depth_map) if res9 else None

    # exp7.3 (GRPO pur full-ft : référence, eval_logs + valeurs successives we5)
    res73 = load_eval_logs(EXP["exp7.3"]["eval_logs"])
    D["exp7.3"]["reward_rows"] = []
    D["exp7.3"]["eval_curve"] = []
    D["exp7.3"]["perdepth_pass1"] = evallogs_perdepth_pass1(res73, depth_map) if res73 else None
    D["exp7.3"]["overall_pass1"] = evallogs_overall_pass1(res73)
    D["exp7.3"]["pertype_err"] = evallogs_pertype_err(res73) if res73 else None

    return D


def print_summary(D: dict) -> None:
    import numpy as np
    print("\n" + "=" * 72)
    print("RÉSUMÉ DES CHIFFRES OBSERVÉS (axe = epoch ; exp10 EXCLU)")
    print("=" * 72)
    for key in ("exp7.3", "exp8.1", "exp8.2", "exp9"):
        d = D[key]
        print(f"\n{key:<8} {EXP[key]['label']}")
        if key == "exp7.3":
            e0, e1 = EXP73_EPOCH_RANGE
            print(f"  pass@1 we5 (epochs≈{e0}→{e1}, APPROX) : {EXP73_WE5_PASS1} "
                  f"(best {max(EXP73_WE5_PASS1)}, oracle best {EXP73_BEST})")
        if d.get("eval_curve"):
            ys = [p[1] for p in d["eval_curve"]]
            rows = d.get("reward_rows") or []
            if rows:
                xp = np.array([r["step"] for r in rows], float)
                fp = np.array([r["epoch"] for r in rows], float)
                e_first = float(np.interp(d["eval_curve"][0][0], xp, fp))
                e_last = float(np.interp(d["eval_curve"][-1][0], xp, fp))
                print(f"  pass@1 eval : best={max(ys)} final={ys[-1]} "
                      f"(epochs {e_first:.1f}→{e_last:.1f}, n={len(ys)})")
            else:
                print(f"  pass@1 eval : best={max(ys)} final={ys[-1]} (n={len(ys)})")
        if d.get("overall_pass1") is not None:
            print(f"  pass@1 global (eval_logs) = {d['overall_pass1']:.0f}/100")
        if d.get("perdepth_pass1"):
            pp = d["perdepth_pass1"]
            print("  pass@1 par depth : " +
                  ", ".join(f"d{k}={'-' if pp[k] is None else f'{pp[k]:.0f}'}" for k in DEPTHS))
        if d.get("reward_rows"):
            rr = d["reward_rows"]
            print(f"  reward steps : n={len(rr)}, epochs {rr[0]['x']:.2f}→{rr[-1]['x']:.2f}, "
                  f"reward {rr[0]['reward']:.3f}→{rr[-1]['reward']:.3f}")


def cleanup_stale() -> None:
    """Supprime les anciens PNG basés sur 'step' (renommés en 'epoch')."""
    for stale in ("01_reward_train_vs_step.png", "02_pass1_test_vs_step.png"):
        p = OUTDIR / stale
        if p.exists():
            p.unlink()
            print(f"[dashboard] supprimé (renommé) : {stale}")


def main() -> None:
    print(f"[dashboard] repo = {REPO}")
    write_exp9_analysis()
    cleanup_stale()
    D = load_all()
    written = make_plots(D)
    print_summary(D)
    print("\n" + "=" * 72)
    print(f"[dashboard] {len(written)} PNG écrits dans {OUTDIR}/ :")
    for name in written:
        p = OUTDIR / name
        size = p.stat().st_size if p.exists() else 0
        print(f"  {name:<34} {size/1024:6.1f} KB")
    print("=" * 72)


if __name__ == "__main__":
    main()
