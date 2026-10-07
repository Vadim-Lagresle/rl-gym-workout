"""Extrait des logs d'entraînement (logs/<run>.log) les séries utiles aux figures post-rapport.

Pour chaque run : Pass@1 test périodique (lignes [test_eval]) et métriques TRL par pas
(reward, kl, entropy, longueur moyenne), indexées par le pas global lu sur la barre de
progression de la même ligne. En cas de relance/reprise, la DERNIÈRE occurrence d'un pas gagne.
Sortie : docs/post_rapport/runs_data.json
Usage : python docs/post_rapport/extract_runs.py
"""
import ast, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["exp25_r8_anchor4ep", "exp30_r8_fixedanchor", "exp32_horizon", "exp36_n16",
        "exp40_g16_8tasks_anchor8", "exp41_g8_8tasks_anchor8", "exp43_g16_klclamp10",
        "exp44_horizon_g8", "exp45_fixedanchor_klclamp10", "exp46_movingref_g8",
        "exp36.1_anchor12", "exp47_fixedanchor_lr1e-6", "exp48_movingref_lr1e-6",
        "exp49_movingref_resetadam", "exp50_g16_horizon_klclamp_anchor10", "exp51_depthbal_uniform",
        "exp51.1_depthbal_uniform_anchor372"]
EVAL = re.compile(r"\[test_eval\] step (\d+) — Pass@1 = (\d+)/(\d+)")
BAR = re.compile(r"(\d+)/(\d+) \[")
KEYS = {"reward": "reward", "kl": "kl", "entropy": "entropy",
        "completions/mean_length": "length", "epoch": "epoch"}

def parse(path):
    evals, steps = {}, {}
    for line in path.open(errors="replace"):
        m = EVAL.search(line)
        if m:
            evals[int(m[1])] = int(m[2]) / int(m[3])
            continue
        i = line.rfind("{'loss'")
        if i < 0:
            continue
        bars = BAR.findall(line[:i])
        if not bars:
            continue
        try:
            d = ast.literal_eval(line[i:line.index("}", i) + 1])
        except (ValueError, SyntaxError):
            continue
        steps[int(bars[-1][0])] = {v: float(d[k]) for k, v in KEYS.items() if d.get(k) is not None}
    s = sorted(steps)
    # pas → époque : régression sur les pas loggés (l'époque TRL est linéaire en pas)
    spe = s[-1] / steps[s[-1]]["epoch"] if s and steps[s[-1]].get("epoch") else None
    return {"steps_per_epoch": spe,
            "eval": [[k, evals[k]] for k in sorted(evals)],
            "train": {"step": s, **{v: [steps[k].get(v) for k in s] for v in KEYS.values()}}}

out = {}
for r in RUNS:
    p = ROOT / "logs" / f"{r}.log"
    if p.exists():
        out[r] = parse(p)
        e = out[r]["eval"]
        print(f"{r:32s} evals={len(e):3d} steps={len(out[r]['train']['step']):5d} "
              f"spe={out[r]['steps_per_epoch'] or 0:.1f} best={max((v for _, v in e), default=0):.2f}")
(Path(__file__).parent / "runs_data.json").write_text(json.dumps(out))
