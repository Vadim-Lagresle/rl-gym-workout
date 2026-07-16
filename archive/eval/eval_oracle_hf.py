"""Oracle pass@k via HuggingFace generate — pour les modèles que vLLM 0.9.1 ne sait PAS
servir (ex. Qwen3.5-4B, archi qwen3_5). Même sortie que eval_oracle.py (`oracle_passk.json`),
mais génération HF (séquentielle, lente) au lieu du serveur vLLM.

RÉSUMABLE (crucial vu les coupures infra) : chaque passe complète est écrite dans
`runs/<run>/passes.jsonl` immédiatement. Relancer la MÊME commande reprend aux passes
manquantes → une coupure ne coûte qu'une passe (~35 min), pas tout le run.

Réutilise `run_episode`/constantes d'eval_fullft.py (chemin HF) et `pass_at_k` d'eval_oracle.py
(estimateur). Le modèle est chargé une seule fois et réutilisé sur les N passes.

Pré-requis : serveur TextCraft (port 36005). PAS de serveur vLLM (génération in-process HF).

Usage :
    python src/eval/eval_oracle_hf.py --model models/Qwen3.5-4B --run-name oracle_qwen35_4b \
        --n-samples 20 --no-thinking
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agentenv.envs import TextCraftEnvClient

from eval_fullft import (
    run_episode, ENV_SERVER_URL, DEFAULT_SYSTEM_PROMPT, DATASET_PATH,
)
from eval_oracle import pass_at_k  # estimateur non biaisé partagé

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="chemin local ou id HF Hub")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-items", type=int, default=0, help="0 = les 100 items")
    ap.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Qwen3/3.5 : enable_thinking=False (actions directes, pas de <think>)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    passes_fp = out_dir / "passes.jsonl"

    with DATASET_PATH.open() as f:
        items = json.load(f)
    if args.max_items > 0:
        items = items[:args.max_items]
    item_ids = [it["item_id"] for it in items]
    with DEPTH_PATH.open() as f:
        depth_map = json.load(f)
    n_items = len(items)
    N = args.n_samples

    # ── Reprise : recharge les passes déjà sauvées ─────────────────────────
    done: dict[int, list[int]] = {}
    if passes_fp.exists():
        for line in passes_fp.open():
            line = line.strip()
            if line:
                d = json.loads(line)
                done[d["pass"]] = d["successes"]
    print(f"[oracle_hf] {len(done)}/{N} passes déjà faites — reprise.", flush=True)

    todo_passes = [s for s in range(N) if s not in done]
    if todo_passes:
        # Résolution du modèle : chemin local s'il existe, sinon id HF Hub.
        local = Path(args.model)
        if not local.is_absolute():
            local = REPO_ROOT / args.model
        model_ref = str(local) if local.exists() else args.model
        print(f"[oracle_hf] chargement {model_ref} (HF generate) ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        print("[oracle_hf] modèle chargé.", flush=True)

        for s in todo_passes:
            t0 = time.time()
            succ = []
            for it in items:
                r = run_episode(model, tok, client, it["item_id"],
                                system_prompt=args.system_prompt,
                                enable_thinking=not args.no_thinking)
                succ.append(1 if r.reward >= 1.0 else 0)
            done[s] = succ
            with passes_fp.open("a") as f:  # checkpoint immédiat de la passe
                f.write(json.dumps({"pass": s, "successes": succ, "items": item_ids}) + "\n")
            print(f"[oracle_hf] passe {s + 1}/{N} : {sum(succ)}/{n_items} résolus "
                  f"({time.time() - t0:.0f}s) [sauvée]", flush=True)

    # ── Agrégation pass@k (identique à eval_oracle) ────────────────────────
    c = [sum(done[s][i] for s in range(N)) for i in range(n_items)]
    ks = list(range(1, N + 1))

    def passk_over(indices: list[int]) -> dict[int, float]:
        return {k: sum(pass_at_k(N, c[i], k) for i in indices) / len(indices) for k in ks}

    overall = passk_over(list(range(n_items)))
    groups: dict = defaultdict(list)
    for i, iid in enumerate(item_ids):
        groups[depth_map.get(iid, "?")].append(i)
    by_depth = {d: passk_over(idx) for d, idx in groups.items()}

    show_ks = [k for k in (1, 2, 5, 10, N) if k <= N]
    print(f"\n=== pass@k GLOBAL ({n_items} items, {N} tirages) ===")
    for k in show_ks:
        print(f"  pass@{k:<2d} = {100 * overall[k]:.1f}%")
    print("\n=== pass@k PAR DEPTH ===")
    print("depth | items | " + " | ".join(f"@{k}".rjust(5) for k in show_ks))
    for d in sorted(groups, key=str):
        idx = groups[d]
        cells = " | ".join(f"{100 * by_depth[d][k]:.0f}".rjust(5) for k in show_ks)
        print(f"  {str(d):<3} | {len(idx):4d}  | {cells}")

    payload = {
        "model": args.model,
        "engine": "hf_generate",
        "n_samples": N,
        "n_items": n_items,
        "successes_per_item": {item_ids[i]: c[i] for i in range(n_items)},
        "pass_at_k_overall": {str(k): overall[k] for k in ks},
        "pass_at_k_by_depth": {str(d): {str(k): by_depth[d][k] for k in ks} for d in by_depth},
    }
    (out_dir / "oracle_passk.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[oracle_hf] résultats -> {out_dir / 'oracle_passk.json'}")


if __name__ == "__main__":
    main()
