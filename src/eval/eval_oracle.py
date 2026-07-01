"""
Expérience "oracle" — pass@k sur TextCraft pour estimer la marge exploitable par le RL.

Idée (discussion 2026-06-16, "trick best-of-K") : pour chaque item de test on tire N
trajectoires indépendantes (température 1.0) et on regarde si AU MOINS UNE réussit. Le
pass@k (estimateur non biaisé de Chen et al. 2021) trace, en fonction de k=1..N, le taux
de réussite "avec oracle" — un oracle qui choisirait toujours la meilleure des k tentatives.

Lecture :
  - pass@1   ≈ taux de réussite actuel (une seule trajectoire par item).
  - pass@N >> pass@1  =>  le modèle SAIT parfois résoudre l'item mais pas de façon fiable
                          =>  marge pour le RL (rendre fiable ce qui est sporadique).
  - pass@N ≈ pass@1   =>  plafond de capacité  =>  le RL seul n'y peut pas grand-chose.

C'est une borne SUPÉRIEURE OPTIMISTE : le RL réel n'atteint pas l'oracle (exploration
imparfaite, il doit aussi ne pas casser ce qu'il sait déjà faire). Signal de potentiel,
pas le gain réel.

Le breakdown PAR DEPTH est le point clé pour TextCraft : il dit si le mur depth 3-4 est
un plafond de capacité (pass@N ≈ 0) ou un problème de fiabilité (pass@N élevé). À lancer
sur la baseline Qwen ET sur le meilleur checkpoint RL pour comparer les deux marges.

Pré-requis :
    # serveur vLLM sur le modèle à tester (port 8001)
    bash src/utils/start_vllm_server.sh <model_path>
    # serveur TextCraft (port 36005) déjà lancé

Usage :
    python src/eval/eval_oracle.py --model <model_path> --run-name <name> --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from agentenv.envs import TextCraftEnvClient

# Réutilise les briques d'eval_vllm.py (même dossier, ajouté à sys.path par Python).
from eval_vllm import (
    generate_reply_vllm,
    build_initial_messages,
    item_id_to_idx,
    check_vllm_server,
    ENV_SERVER_URL,
    MAX_ROUNDS,
    DEFAULT_SYSTEM_PROMPT,
    DATASET_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimateur non biaisé pass@k (Chen et al. 2021, HumanEval) : probabilité qu'un
    sous-ensemble aléatoire de k tirages parmi n contienne >=1 réussite, sachant c
    réussites. Plus stable que de prendre "les k premiers"."""
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_one_pass(items: list[dict], model: str, temperature: float,
                 max_workers: int = 64) -> list[int]:
    """Une passe stochastique : 1 trajectoire par item. Toutes les trajectoires sont
    avancées en parallèle tour par tour (le serveur vLLM batche les générations, le pool
    de threads parallélise les appels HTTP de génération et de step env). Retourne la
    liste des succès (1/0) par item, dans l'ordre de `items`."""
    n = len(items)
    clients: list[TextCraftEnvClient] = []
    states: list[list[dict]] = []
    done = [False] * n
    reward = [0.0] * n

    for it in items:
        c = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        c.reset(item_id_to_idx(it["item_id"]))
        clients.append(c)
        states.append(build_initial_messages(c, system_prompt=DEFAULT_SYSTEM_PROMPT))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in range(MAX_ROUNDS):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            # 1) générations concurrentes (vLLM batche côté serveur)
            replies = list(ex.map(
                lambda i: generate_reply_vllm(model, states[i], temperature=temperature),
                active,
            ))

            # 2) step env concurrents (1 client distinct par épisode => pas de partage d'état)
            def _step(pair: tuple[int, str]) -> tuple[int, float, bool]:
                i, txt = pair
                states[i].append({"role": "assistant", "content": txt})
                out = clients[i].step(txt)
                states[i].append({"role": "user", "content": out.state})
                return i, float(out.reward), bool(out.done)

            for i, r, d in ex.map(_step, zip(active, replies)):
                reward[i] = r
                if d:
                    done[i] = True

    for c in clients:
        try:
            c.close()
        except Exception:
            pass

    return [1 if reward[i] >= 1.0 else 0 for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="chemin/nom du modèle tel que servi par vLLM sur le port 8001")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-samples", type=int, default=20, help="N trajectoires par item")
    ap.add_argument("--max-items", type=int, default=0, help="0 = les 100 items du test")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="DOIT être > 0 pour diversifier les tirages (sinon pass@k = pass@1)")
    ap.add_argument("--max-workers", type=int, default=64)
    args = ap.parse_args()

    if args.temperature <= 0:
        raise SystemExit("--temperature doit être > 0 (sinon les N tirages sont identiques).")
    check_vllm_server()

    with DATASET_PATH.open() as f:
        items = json.load(f)
    if args.max_items > 0:
        items = items[:args.max_items]
    with DEPTH_PATH.open() as f:
        depth_map = json.load(f)

    n_items = len(items)
    N = args.n_samples
    succ = [[0] * N for _ in range(n_items)]  # matrice succès [item][tirage]

    print(f"[oracle] {n_items} items × {N} tirages (T={args.temperature}) sur {args.model}", flush=True)
    t0 = time.time()
    for s in range(N):
        ts = time.time()
        col = run_one_pass(items, args.model, args.temperature, args.max_workers)
        for i in range(n_items):
            succ[i][s] = col[i]
        print(f"[oracle] passe {s + 1}/{N} : {sum(col)}/{n_items} résolus "
              f"({time.time() - ts:.0f}s)", flush=True)

    # c_i = nb de réussites parmi N pour l'item i
    c = [sum(succ[i]) for i in range(n_items)]
    ks = list(range(1, N + 1))

    def passk_over(indices: list[int]) -> dict[int, float]:
        return {k: sum(pass_at_k(N, c[i], k) for i in indices) / len(indices) for k in ks}

    overall = passk_over(list(range(n_items)))
    depth_groups: dict = defaultdict(list)
    for i, it in enumerate(items):
        depth_groups[depth_map.get(it["item_id"], "?")].append(i)
    by_depth = {d: passk_over(idxs) for d, idxs in depth_groups.items()}

    # ── Affichage ───────────────────────────────────────────────────────────
    show_ks = [k for k in (1, 2, 5, 10, N) if k <= N]
    print(f"\n=== pass@k GLOBAL ({n_items} items, {N} tirages) ===")
    for k in show_ks:
        print(f"  pass@{k:<2d} = {100 * overall[k]:.1f}%")
    print(f"\n=== pass@k PAR DEPTH ===")
    print("depth | items | " + " | ".join(f"@{k}".rjust(5) for k in show_ks))
    for d in sorted(depth_groups, key=str):
        idxs = depth_groups[d]
        cells = " | ".join(f"{100 * by_depth[d][k]:.0f}".rjust(5) for k in show_ks)
        print(f"  {str(d):<3} | {len(idxs):4d}  | {cells}")

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "n_samples": N,
        "n_items": n_items,
        "temperature": args.temperature,
        "successes_per_item": {items[i]["item_id"]: c[i] for i in range(n_items)},
        "pass_at_k_overall": {str(k): overall[k] for k in ks},
        "pass_at_k_by_depth": {str(d): {str(k): by_depth[d][k] for k in ks} for d in by_depth},
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_fp = out_dir / "oracle_passk.json"
    with out_fp.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[oracle] résultats -> {out_fp}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
