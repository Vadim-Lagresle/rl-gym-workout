"""
Analyse approfondie des logs d'évaluation TextCraft.

Produit :
  1. Tableau détaillé par épisode
  2. Résumé par depth
  3. Histogramme des types d'erreurs
  4. Analyse de récupération (le modèle s'adapte-t-il après une erreur ?)

Usage :
    # Analyse simple (sans tokens)
    python src/analysis/analyze_eval.py \\
        --eval-dir runs/exp7.1_ckpt1598/eval_logs

    # Avec comptage de tokens (quelques minutes de plus)
    python src/analysis/analyze_eval.py \\
        --eval-dir runs/exp7.1_ckpt1598/eval_logs \\
        --tokenizer saves/trl_grpo/exp7.1_b200_fullft_12ep/checkpoint-1598

    # Export CSV
    python src/analysis/analyze_eval.py \\
        --eval-dir runs/exp7.1_ckpt1598/eval_logs --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ── Constantes ────────────────────────────────────────────────────────────────

MAX_CONTEXT = 32_768  # Qwen2.5-3B-Instruct

# Patterns de classification des observations d'erreur
# Ordre IMPORTANT : classify_error garde le PREMIER pattern qui matche.
# Les patterns spécifiques doivent précéder les résiduels (other_error, generic_fail).
ERROR_PATTERNS = [
    ("format_error",   re.compile(r"could not execute", re.I)),          # action mal formée, parseur rejette
    ("recipe_wrong",   re.compile(r"could not find a valid recipe", re.I)),
    ("missing_items",  re.compile(r"could not find enough items", re.I)),  # quantité insuffisante
    ("item_not_found", re.compile(r"could not find", re.I)),             # "Could not find <objet>" : absent inventaire / mal nommé
    ("wrong_format",   re.compile(r"wrong item format", re.I)),
    ("multi_action",   re.compile(r"only one .action. is allowed", re.I)),  # plusieurs "Action:" dans une réponse
    ("other_error",    re.compile(r"error:", re.I)),                     # résiduel "error:"
    ("generic_fail",   re.compile(r"could not", re.I)),                  # résiduel "could not …"
]

# Description lisible de chaque type d'erreur (affichée dans chaque analyse).
ERROR_DESCRIPTIONS = {
    "format_error":   "« could not execute » — action mal formée, le parseur de l'env la rejette (syntaxe).",
    "recipe_wrong":   "« could not find a valid recipe » — aucune recette valide pour la cible.",
    "missing_items":  "« could not find enough items » — prérequis en quantité insuffisante (erreur de planif).",
    "item_not_found": "« could not find <objet> » — objet absent de l'inventaire / mal nommé.",
    "wrong_format":   "« wrong item format » — nom d'objet mal écrit.",
    "multi_action":   "« only one 'Action' is allowed » — plusieurs actions émises en une réponse (protocole).",
    "other_error":    "résiduel : contient « error: » sans matcher un cas ci-dessus.",
    "generic_fail":   "résiduel : contient « could not » sans matcher un cas ci-dessus.",
}

ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def classify_error(obs: str) -> str | None:
    """Retourne la catégorie d'erreur d'une observation, ou None si succès/neutre."""
    for label, pat in ERROR_PATTERNS:
        if pat.search(obs):
            return label
    return None


def extract_action(text: str) -> str:
    """Extrait la première ligne d'action d'un message assistant."""
    m = ACTION_RE.search(text)
    if not m:
        return ""
    return " ".join(m.group(1).strip().split())


def count_tokens_for_messages(tokenizer, messages: list[dict]) -> int:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(prompt, add_special_tokens=False))


# ── Analyse d'un épisode ──────────────────────────────────────────────────────

def analyze_episode(log_path: Path, tokenizer=None) -> dict:
    with log_path.open() as f:
        data = json.load(f)

    transcript = data.get("transcript", [])
    rounds     = data["rounds"]
    reward     = float(data["reward"])
    item_id    = data["item_id"]

    # Sépare les tours : chaque tour = (assistant_msg, user_msg)
    # Le transcript est : [system, user(rules), assistant(ack), user(obs0), asst, user, ...]
    # On saute les 4 premiers messages fixes
    turn_pairs: list[tuple[str, str]] = []
    msgs = transcript[4:]  # à partir du premier tour réel
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            asst_text = msgs[i].get("content", "")
            env_text  = msgs[i + 1].get("content", "")
            turn_pairs.append((asst_text, env_text))

    # Erreurs par tour
    error_sequence: list[str | None] = [classify_error(env) for _, env in turn_pairs]
    actions: list[str] = [extract_action(asst) for asst, _ in turn_pairs]

    # Comptage des erreurs
    error_counts: Counter = Counter(e for e in error_sequence if e)
    dominant_error = error_counts.most_common(1)[0][0] if error_counts else None
    n_errors = sum(error_counts.values())

    # Premier tour d'échec
    first_failure_round = next(
        (i + 1 for i, e in enumerate(error_sequence) if e), None
    )

    # Stagnation : plus longue séquence d'actions consécutives identiques
    max_stagnation = 1
    cur_stag = 1
    for i in range(1, len(actions)):
        if actions[i] and actions[i] == actions[i - 1]:
            cur_stag += 1
            max_stagnation = max(max_stagnation, cur_stag)
        else:
            cur_stag = 1

    # Diversité des actions (actions uniques / total actions non vides)
    non_empty = [a for a in actions if a]
    action_diversity = (len(set(non_empty)) / len(non_empty)) if non_empty else 1.0

    # Récupération : après chaque erreur, l'action suivante est-elle différente ?
    recoveries: dict[str, list[bool]] = defaultdict(list)
    for i, err in enumerate(error_sequence):
        if err and i + 1 < len(actions):
            recovered = bool(actions[i + 1] and actions[i + 1] != actions[i])
            recoveries[err].append(recovered)

    # Tokens (optionnel)
    token_stats: dict = {}
    if tokenizer is not None:
        per_turn_tokens = []
        msgs_so_far = transcript[:4]
        for asst_msg, env_msg in zip(
            [m for m in transcript[4:] if m["role"] == "assistant"],
            [m for m in transcript[4:] if m["role"] == "user"],
        ):
            msgs_so_far = msgs_so_far + [asst_msg, env_msg]
            per_turn_tokens.append(count_tokens_for_messages(tokenizer, msgs_so_far))

        token_stats = {
            "total_tokens":   max(per_turn_tokens) if per_turn_tokens else 0,
            "avg_tokens_per_turn": (
                sum(per_turn_tokens) / len(per_turn_tokens) if per_turn_tokens else 0
            ),
            "exceeded_context": (
                max(per_turn_tokens) > MAX_CONTEXT if per_turn_tokens else False
            ),
        }

    return {
        "item_id":            item_id,
        "item_idx":           data.get("item_idx", -1),
        "reward":             reward,
        "rounds":             rounds,
        "error_counts":       dict(error_counts),
        "n_errors":           n_errors,
        "dominant_error":     dominant_error,
        "first_failure_round": first_failure_round,
        "max_stagnation":     max_stagnation,
        "action_diversity":   round(action_diversity, 2),
        "recoveries":         dict(recoveries),
        "log_path":           str(log_path),
        **token_stats,
    }


# ── Chargement du depth ────────────────────────────────────────────────────────

def load_depth_map(repo_root: Path) -> dict[str, int]:
    """Charge les depths depuis les fichiers train et test (les deux si disponibles)."""
    depth_map: dict[str, int] = {}
    for depth_file in [
        repo_root / "data" / "train" / "textcraft_train_with_depth.json",
        repo_root / "data" / "eval"  / "textcraft_test_with_depth.json",
    ]:
        if depth_file.exists():
            with depth_file.open() as f:
                depth_map.update(json.load(f))
    return depth_map


# ── Affichage ─────────────────────────────────────────────────────────────────

def fmt(val, fmt_str="{}", default="-"):
    return default if val is None else fmt_str.format(val)


def print_table(results: list[dict], depth_map: dict, show_tokens: bool) -> None:
    has_tokens = show_tokens and results and "total_tokens" in results[0]

    # En-tête
    cols = ["Item", "Dep", "Rwd", "Rnd", "Err", "DomErr", "Stag", "1stFail", "Div"]
    if has_tokens:
        cols += ["TotTok", "AvgTok/R", "Ctx%"]
    cols.append("Log")

    widths = [16, 3, 3, 3, 4, 15, 4, 7, 4]
    if has_tokens:
        widths += [7, 8, 5]
    widths.append(50)

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

    print("\n" + sep)
    print(hdr)
    print(sep)

    for r in results:
        depth = depth_map.get(r["item_id"], "?")
        dom   = (r["dominant_error"] or "-")[:15]
        div   = f"{r['action_diversity']:.2f}"
        ff    = fmt(r["first_failure_round"], "{:3d}")
        stag  = f"{r['max_stagnation']:2d}" if r["max_stagnation"] > 1 else " 1"
        rwd   = "✓" if r["reward"] > 0 else "✗"

        row = [
            r["item_id"][:16],
            str(depth)[:3],
            rwd,
            str(r["rounds"])[:3],
            str(r["n_errors"])[:4],
            dom,
            stag,
            ff,
            div,
        ]
        if has_tokens:
            tt = r.get("total_tokens", 0)
            at = r.get("avg_tokens_per_turn", 0)
            pct = f"{100*tt/MAX_CONTEXT:.0f}%"
            row += [str(tt)[:7], f"{at:.0f}"[:8], pct[:5]]

        short_log = str(r["log_path"])[-50:]
        row.append(short_log)

        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")

    print(sep)


def print_depth_summary(results: list[dict], depth_map: dict) -> None:
    from statistics import pstdev  # écart-type population (décrit la dispersion observée)

    by_depth: dict[str | int, list] = defaultdict(list)
    for r in results:
        d = depth_map.get(r["item_id"], "?")
        by_depth[d].append(r)

    print("\n── RÉSUMÉ PAR DEPTH " + "─" * 50)
    print(f"{'Depth':>6} | {'Items':>5} | {'Pass@1':>6} | {'MoyRnd':>6} | {'StdRnd':>6} | {'MoyErr':>6} | {'MoyDiv':>6}")
    print("-" * 64)
    for depth in sorted(by_depth.keys(), key=lambda x: (str(x) == "?", x)):
        items = by_depth[depth]
        n      = len(items)
        passed = sum(1 for r in items if r["reward"] > 0)
        rounds = [r["rounds"] for r in items]
        avg_r  = sum(rounds) / n
        std_r  = pstdev(rounds)  # 0.0 si n == 1
        avg_e  = sum(r["n_errors"] for r in items) / n
        avg_d  = sum(r["action_diversity"] for r in items) / n
        print(f"{str(depth):>6} | {n:>5} | {passed:>4}/{n:<2} | {avg_r:>6.1f} | {std_r:>6.1f} | {avg_e:>6.1f} | {avg_d:>6.2f}")


def print_error_histogram(results: list[dict]) -> None:
    total_errors: Counter = Counter()
    failed_errors: Counter = Counter()

    for r in results:
        for etype, count in r["error_counts"].items():
            total_errors[etype] += count
            if r["reward"] == 0:
                failed_errors[etype] += count

    if not total_errors:
        print("\n── HISTOGRAMME DES ERREURS ── Aucune erreur enregistrée.")
        return

    print("\n── HISTOGRAMME DES ERREURS " + "─" * 43)
    col = "Type d'erreur"
    print(f"{col:<18} | {'Total':>6} | {'Sur echecs':>10} | Barre")
    print("-" * 65)
    max_count = max(total_errors.values())

    for etype, count in total_errors.most_common():
        bar_len = int(30 * count / max_count)
        bar = "█" * bar_len
        on_fails = failed_errors.get(etype, 0)
        print(f"{etype:<18} | {count:>6} | {on_fails:>10} | {bar}")

    print(f"\n  Total erreurs : {sum(total_errors.values())} "
          f"sur {len(results)} épisodes "
          f"({sum(1 for r in results if r['n_errors'] > 0)} épisodes avec au moins 1 erreur)")


def print_error_legend() -> None:
    """Affiche la signification de chaque type d'erreur (dans l'ordre de classification)."""
    print("\n── SIGNIFICATION DES TYPES D'ERREUR " + "─" * 34)
    for label, _ in ERROR_PATTERNS:
        print(f"  {label:<14} : {ERROR_DESCRIPTIONS.get(label, '')}")


def print_recovery_analysis(results: list[dict]) -> None:
    # Agrège les récupérations par type d'erreur
    agg: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for etype, bools in r["recoveries"].items():
            agg[etype].extend(bools)

    if not agg:
        print("\n── ANALYSE DE RÉCUPÉRATION ── Pas assez de données.")
        return

    print("\n── ANALYSE DE RÉCUPÉRATION " + "─" * 43)
    print("Après une erreur, le modèle génère-t-il une action DIFFÉRENTE au tour suivant ?")
    col2 = "Type d'erreur"
    print(f"\n{col2:<18} | {'Tentatives':>10} | {'Recuperations':>13} | {'Taux':>6} | Barre")
    print("-" * 70)

    for etype, bools in sorted(agg.items(), key=lambda x: -len(x[1])):
        n      = len(bools)
        recov  = sum(bools)
        rate   = recov / n if n else 0
        bar    = "█" * int(30 * rate) + "░" * (30 - int(30 * rate))
        print(f"{etype:<18} | {n:>10} | {recov:>13} | {rate:>5.0%} | {bar}")

    print("\n  Taux global :")
    all_bools = [b for bools in agg.values() for b in bools]
    if all_bools:
        global_rate = sum(all_bools) / len(all_bools)
        print(f"  {sum(all_bools)}/{len(all_bools)} erreurs suivies d'une action différente ({global_rate:.0%})")


def print_context_warning(results: list[dict]) -> None:
    exceeded = [r for r in results if r.get("exceeded_context")]
    if exceeded:
        print(f"\n⚠️  CONTEXTE DÉPASSÉ : {len(exceeded)} épisode(s) ont dépassé {MAX_CONTEXT} tokens !")
        for r in exceeded:
            print(f"   {r['item_id']}  tokens={r['total_tokens']}")
    else:
        print(f"\n✅ Aucun épisode n'a dépassé la fenêtre de contexte ({MAX_CONTEXT} tokens).")


def export_csv(results: list[dict], depth_map: dict, path: str) -> None:
    fields = [
        "item_id", "item_idx", "depth", "reward", "rounds", "n_errors",
        "dominant_error", "first_failure_round", "max_stagnation",
        "action_diversity", "total_tokens", "avg_tokens_per_turn",
        "exceeded_context", "log_path",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {**r, "depth": depth_map.get(r["item_id"], "?")}
            w.writerow(row)
    print(f"\nCSV exporté : {path}")


# ── Tee stdout vers fichier ───────────────────────────────────────────────────

class _Tee:
    """Duplique stdout vers un fichier tout en continuant d'afficher dans le terminal."""
    def __init__(self, path: Path):
        import sys
        self._file = path.open("w")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        import sys
        sys.stdout = self._stdout
        self._file.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--tokenizer", default="", help="Chemin tokenizer pour comptage tokens (optionnel)")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder les résultats sur disque")
    args = parser.parse_args()

    eval_dir  = Path(args.eval_dir)
    run_dir   = eval_dir.parent          # ex: runs/exp7.1_ckpt1598/
    repo_root = Path(__file__).resolve().parents[2]
    logs      = sorted(eval_dir.glob("textcraft_*.json"))

    if not logs:
        print(f"Aucun log trouvé dans {eval_dir}")
        return

    # Sauvegarde automatique dans le dossier du run
    tee = None
    txt_path = run_dir / "analysis.txt"
    csv_path = run_dir / "analysis.csv"
    if not args.no_save:
        tee = _Tee(txt_path)

    # Tokenizer (optionnel)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"[analyze] Chargement tokenizer depuis {args.tokenizer} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    depth_map = load_depth_map(repo_root)
    if not depth_map:
        print("[analyze] Fichier depth non trouvé — colonne Depth affichée comme '?'")
        print("[analyze] Pour générer : source ~/envs/agentenv-textcraft/bin/activate")
        print("[analyze]   && python src/utils/label_depths.py --split test")

    print(f"[analyze] Analyse de {len(logs)} épisodes dans {eval_dir}\n")

    results = []
    for log_path in logs:
        r = analyze_episode(log_path, tokenizer)
        results.append(r)

    show_tokens = tokenizer is not None

    # ── Sorties ──
    print_table(results, depth_map, show_tokens)
    print_depth_summary(results, depth_map)
    print_error_histogram(results)
    print_error_legend()
    print_recovery_analysis(results)

    if show_tokens:
        print_context_warning(results)

    passed = sum(1 for r in results if r["reward"] > 0)
    print(f"\n{'='*60}")
    print(f"Pass@1 = {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    print(f"{'='*60}\n")

    if not args.no_save:
        export_csv(results, depth_map, str(csv_path))
        if tee:
            tee.close()
        print(f"Résultats sauvegardés dans {run_dir}/")
        print(f"  {txt_path.name}  ← analyse complète (texte)")
        print(f"  {csv_path.name}  ← tableau (CSV)")


if __name__ == "__main__":
    main()
