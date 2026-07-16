"""Expérience "oracle" — pass@k sur TextCraft pour estimer la marge exploitable par le RL.

Idée (discussion 2026-06-16, "trick best-of-K") : pour chaque item de test on tire N
trajectoires indépendantes (température 1.0) et on regarde si AU MOINS UNE réussit. Le
pass@k (estimateur non biaisé de Chen et al. 2021) trace, en fonction de k=1..N, le taux
de réussite "avec oracle" — un oracle qui choisirait toujours la meilleure des k tentatives.

Lecture :
  - pass@1   ≈ taux de réussite actuel (une seule trajectoire par item).
  - pass@N >> pass@1  =>  le modèle SAIT parfois résoudre l'item mais pas de façon fiable
                          =>  marge pour le RL (rendre fiable ce qui est sporadique).
  - pass@N ≈ pass@1   =>  plafond de capacité  =>  le RL seul n'y peut pas grand-chose.

Le breakdown PAR DEPTH est le point clé pour TextCraft : il dit si le mur depth 3-4 est
un plafond de capacité (pass@N ≈ 0) ou un problème de fiabilité (pass@N élevé).

Fusion d'eval_oracle.py + eval_oracle_hf.py (refacto 2026-07-16) — deux backends :
  vllm  passes parallélisées (ThreadPool sur le serveur vLLM, ~35 s/passe pour 100 items)
  hf    passes séquentielles in-process (archis non servables par vLLM 0.9.1, ex. Qwen3.5)
RÉSUMABLE dans les deux cas (crucial vu les coupures infra) : chaque passe complète est
écrite dans runs/<run>/passes.jsonl immédiatement ; relancer la MÊME commande reprend aux
passes manquantes → une coupure ne coûte qu'une passe, pas tout le run.

Usage :
    # backend vllm : serveur sur le modèle à tester (port 8001) + serveur TextCraft
    bash src/utils/start_vllm_server.sh <model_path>
    python src/eval/eval_oracle.py --model <model_path> --run-name 7_oracle/<name> --n-samples 20

    # backend hf (ex. Qwen3.5-4B)
    python src/eval/eval_oracle.py --model models/Qwen3.5-4B --backend hf --no-thinking \\
        --run-name 7_oracle/oracle_qwen35_4b --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo → imports src.*

from agentenv.envs import TextCraftEnvClient

from src.eval.llm_chat import ChatGenerator, VLLM_SERVER_URL
from src.eval.textcraft_common import (
    DEFAULT_SYSTEM_PROMPT,
    ENV_SERVER_URL,
    MAX_ROUNDS,
    RUNS_DIR,
    build_initial_messages,
    item_id_to_idx,
    load_depth_map,
    load_items,
    pass_at_k,
    run_episode,
)

ORACLE_MAX_TOKENS = 512


def run_one_pass_vllm(items: list[dict], gen: ChatGenerator, temperature: float,
                      system_prompt: str, max_workers: int = 64) -> list[int]:
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
        states.append(build_initial_messages(c, system_prompt=system_prompt))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in range(MAX_ROUNDS):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            # 1) générations concurrentes (vLLM batche côté serveur)
            replies = list(ex.map(
                lambda i: gen.generate(states[i], max_tokens=ORACLE_MAX_TOKENS,
                                       temperature=temperature),
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


def run_one_pass_hf(items: list[dict], gen: ChatGenerator, temperature: float,
                    system_prompt: str, client: TextCraftEnvClient) -> list[int]:
    """Une passe séquentielle via HF generate (le modèle in-process n'est pas thread-safe)."""
    generate_fn = lambda msgs: gen.generate(msgs, max_tokens=ORACLE_MAX_TOKENS,  # noqa: E731
                                            temperature=temperature)
    succ = []
    for it in items:
        r = run_episode(generate_fn, client, it["item_id"], system_prompt=system_prompt)
        succ.append(1 if r.reward >= 1.0 else 0)
    return succ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="chemin local ou id HF Hub")
    ap.add_argument("--run-name", required=True,
                    help="sortie dans runs/<run-name>/ (ex. 7_oracle/oracle_<modele>)")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--n-samples", type=int, default=20, help="N trajectoires par item")
    ap.add_argument("--max-items", type=int, default=0, help="0 = les 100 items du test")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="DOIT être > 0 pour diversifier les tirages (sinon pass@k = pass@1)")
    ap.add_argument("--max-workers", type=int, default=64, help="parallélisme (backend vllm)")
    ap.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Qwen3/3.5 : enable_thinking=False (backend hf)")
    ap.add_argument("--vllm-url", type=str, default=VLLM_SERVER_URL)
    args = ap.parse_args()

    if args.temperature <= 0:
        raise SystemExit("--temperature doit être > 0 (sinon les N tirages sont identiques).")

    gen = ChatGenerator(model_ref=args.model, backend=args.backend,
                        vllm_url=args.vllm_url, no_thinking=args.no_thinking)

    items = load_items(args.max_items)
    item_ids = [it["item_id"] for it in items]
    depth_map = load_depth_map()
    n_items = len(items)
    N = args.n_samples

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    passes_fp = out_dir / "passes.jsonl"

    # ── Reprise : recharge les passes déjà sauvées ─────────────────────────
    done: dict[int, list[int]] = {}
    if passes_fp.exists():
        for line in passes_fp.open():
            line = line.strip()
            if line:
                d = json.loads(line)
                done[d["pass"]] = d["successes"]
    if done:
        print(f"[oracle] {len(done)}/{N} passes déjà faites — reprise.", flush=True)

    todo_passes = [s for s in range(N) if s not in done]
    print(f"[oracle] backend={gen.backend_name} — {n_items} items × {N} tirages "
          f"(T={args.temperature}), {len(todo_passes)} passes à jouer.", flush=True)

    t0 = time.time()
    hf_client = None
    if todo_passes and gen.backend_name == "hf":
        hf_client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
    for s in todo_passes:
        ts = time.time()
        if gen.backend_name == "vllm":
            col = run_one_pass_vllm(items, gen, args.temperature,
                                    args.system_prompt, args.max_workers)
        else:
            col = run_one_pass_hf(items, gen, args.temperature, args.system_prompt, hf_client)
        done[s] = col
        with passes_fp.open("a") as f:  # checkpoint immédiat de la passe
            f.write(json.dumps({"pass": s, "successes": col, "items": item_ids}) + "\n")
        print(f"[oracle] passe {s + 1}/{N} : {sum(col)}/{n_items} résolus "
              f"({time.time() - ts:.0f}s) [sauvée]", flush=True)

    # ── Agrégation pass@k ───────────────────────────────────────────────────
    # c_i = nb de réussites parmi N pour l'item i
    c = [sum(done[s][i] for s in range(N)) for i in range(n_items)]
    ks = list(range(1, N + 1))

    def passk_over(indices: list[int]) -> dict[int, float]:
        return {k: sum(pass_at_k(N, c[i], k) for i in indices) / len(indices) for k in ks}

    overall = passk_over(list(range(n_items)))
    depth_groups: dict = defaultdict(list)
    for i, iid in enumerate(item_ids):
        depth_groups[depth_map.get(iid, "?")].append(i)
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
    payload = {
        "model": args.model,
        "engine": gen.backend_name,
        "n_samples": N,
        "n_items": n_items,
        "temperature": args.temperature,
        "successes_per_item": {item_ids[i]: c[i] for i in range(n_items)},
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
