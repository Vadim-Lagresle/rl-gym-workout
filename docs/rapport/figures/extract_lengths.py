"""Longueurs d'épisode et nombre de tours par pas, depuis logs/<run>.log (15/09/2026).
TRL logue completions/mean_length (flux complet d'un épisode : actions + observations + template) ;
la ligne [rollout] donne n_active_turns par épisode. Les deux sont appariées dans l'ordre du log
(un rollout puis un dict par pas). Sortie : collapse_lengths.json {run: {cat, anchors, spe, rows:[step, epoch, mean_len, mean_turns, frac_cap30]}}."""
import re, ast, json, sys
dict_re = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'.*?\})")
roll_re = re.compile(r"\[rollout\] n=\d+ .*?n_active_turns=(\[[^\]]*\])")
anch_re = re.compile(r"RÉ-ANCRAGE #\d+ @ step (\d+)")
SETS = {
    # figure 1 : le carré G × ancre sans curriculum
    "control": ({"exp36_n16": "collapse", "exp40_g16_8tasks_anchor8": "collapse",
                 "exp41_g8_8tasks_anchor8": "stable", "exp25_r8_anchor4ep": "stable"}, "collapse_lengths.json"),
    # figure 2 : les six runs désignés du rapport (références, contrôle, curriculums)
    "families": ({"exp23.1_verl_100ep": "stable", "exp25_r8_anchor4ep": "stable", "exp36_n16": "collapse",
                  "exp32_horizon": "stable", "exp35_budget1024": "stable", "exp33.1_depth_auto": "stable",
                  "exp33.2_from72": "stable", "exp25.2_from65": "stable"},
                 "collapse_lengths_families.json"),
    # figure 3 : panorama — les runs les plus importants, stables et collapses, jusqu'à 70 époques
    "all": ({"exp23.1_verl_100ep": "stable", "exp25_r8_anchor4ep": "stable", "exp31_r8_anchor_b0001": "stable",
             "exp32_horizon": "stable", "exp35_budget1024": "stable", "exp33.1_depth_auto": "stable",
             "exp39_g16_8tasks": "stable", "exp41_g8_8tasks_anchor8": "stable",
             "exp33.2_from72": "stable", "exp25.2_from65": "stable",
             "exp36_n16": "collapse", "exp40_g16_8tasks_anchor8": "collapse", "exp36.1_anchor12": "collapse",
             "exp34_magellan": "collapse", "exp30_r8_fixedanchor": "collapse",
             "exp31.1_r8_fixedanchor_b00001": "collapse", "exp24_r16_lr3e-6_b0.01": "collapse",
             "exp24_r64_lr3e-6_b0.001": "collapse"}, "collapse_lengths_all.json"),
}
runs, out_name = SETS[sys.argv[1] if len(sys.argv) > 1 else "control"]
out = {}
for run, cat in runs.items():
    rows, anchors, pending = {}, [], None
    for line in open(f"logs/{run}.log", errors="replace"):
        m = roll_re.search(line)
        if m:
            turns = ast.literal_eval(m.group(1)); pending = (sum(turns) / len(turns), sum(t >= 30 for t in turns) / len(turns))
        m = anch_re.search(line)
        if m: anchors.append(int(m.group(1)))
        for m in dict_re.finditer(line):
            try: d = ast.literal_eval(m.group(3))
            except Exception: continue
            if "completions/mean_length" not in d or pending is None: continue
            rows[int(m.group(1))] = [int(m.group(1)), float(d["epoch"]), float(d["completions/mean_length"]), pending[0], pending[1],
                                     float(d.get("kl", "nan"))]
            pending = None
    r = [rows[k] for k in sorted(rows)]
    out[run] = dict(cat=cat, anchors=sorted(set(anchors)), spe=r[-1][0] / r[-1][1], rows=r)
    print(run, len(r), "pas ; ex :", r[100])
json.dump(out, open("docs/rapport/figures/" + out_name, "w"))
