"""Évaluation multi-tour TextCraft — LE point d'entrée (2 backends).

Fusion d'eval_vllm.py et eval_fullft.py (refacto 2026-07-16) : la boucle
d'épisode vit dans textcraft_common.run_episode, la génération dans
llm_chat.ChatGenerator. Le backend est choisi par --backend :

  vllm  serveur vLLM OpenAI-compatible (port 8001) — KV cache entre les rounds,
        ×7-10 plus rapide. Pré-requis : bash src/utils/start_vllm_server.sh <ckpt>
  hf    HuggingFace generate() in-process — LENT mais universel : indispensable
        pour les archis que vLLM 0.9.1 (contrainte glibc 2.28) ne sert pas,
        ex. Qwen3.5 (--no-thinking recommandé)
  auto  (défaut) vllm si l'archi est servable par la vLLM installée, sinon hf

Usage :
    # stack rapide (checkpoint 3B standard)
    bash src/utils/start_vllm_server.sh saves/trl_grpo/<run>/checkpoint-<N>
    python src/eval/eval_textcraft.py --model saves/trl_grpo/<run>/checkpoint-<N> \\
        --run-name <famille>/<run_name>

    # archi non servable par vLLM (ex. Qwen3.5-4B)
    python src/eval/eval_textcraft.py --model models/Qwen3.5-4B --backend hf \\
        --no-thinking --run-name 0_baselines/exp15_qwen35_4b

    # diagnostic pass@K par depth (marge d'exploration ; pour N grand, eval_oracle.py)
    python src/eval/eval_textcraft.py --model <ckpt> --run-name <name> --passk 8
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo → imports src.*

from agentenv.envs import TextCraftEnvClient

from src.eval.llm_chat import ChatGenerator, VLLM_SERVER_URL
from src.eval.textcraft_common import (
    DEFAULT_SYSTEM_PROMPT,
    ENV_SERVER_URL,
    EpisodeResult,
    REPO_ROOT,
    RUNS_DIR,
    load_depth_map,
    load_existing_log,
    load_items,
    run_episode,
    write_episode_log,
)

EVAL_MAX_TOKENS = 512  # budget par tour du protocole d'éval (aligné training)


# ---------------------------------------------------------------------------
# Diagnostic pass@k par profondeur (depth)
#
# But : mesurer la capacité d'EXPLORATION du modèle aux profondeurs difficiles.
#   pass@k(item) = 1 si AU MOINS un des k rollouts atteint reward > 0, sinon 0.
# C'est l'indicateur qui dit si le RL a une graine à amplifier : si aucune
# trajectoire correcte n'est jamais échantillonnée (pass@k = 0), le RL n'a aucun
# signal de récompense à exploiter (cf. Dr. GRPO, arXiv:2503.20783).
# Pour un N grand et l'estimateur non biaisé, préférer eval_oracle.py.
# ---------------------------------------------------------------------------

@dataclass
class PassKItemResult:
    item_id: str
    depth: int
    rewards: list[float]      # k récompenses finales (une par échantillon réussi à tourner)
    n_success: int            # nb d'échantillons avec reward > 0
    passed: bool              # au moins un succès sur les k


def write_passk_log(result: EpisodeResult, sample_idx: int, depth: int,
                    log_dir: Path) -> None:
    import json
    fp = log_dir / f"{result.item_id}_d{depth}_s{sample_idx}.json"
    payload = {
        "item_id": result.item_id,
        "item_idx": result.item_idx,
        "depth": depth,
        "sample_idx": sample_idx,
        "reward": result.reward,
        "done": result.done,
        "rounds": result.rounds,
        "duration_s": round(result.duration_s, 2),
        "transcript": result.transcript,
    }
    with fp.open("w") as f:
        json.dump(payload, f, indent=2)


def run_pass_at_k_by_depth(
    gen: ChatGenerator,
    items: list[dict],
    depth_map: dict[str, int],
    k: int = 8,
    depths: tuple[int, ...] = (3, 4),
    temperature: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    log_dir: Path | None = None,
) -> list[PassKItemResult]:
    """Lance k rollouts par item (depths ciblées) et calcule pass@k par item.

    Réutilise run_episode() tel quel ; seule la température (>0) garantit la
    diversité entre les k échantillons d'un même item.
    """
    target = [it for it in items if depth_map.get(it["item_id"]) in depths]
    if not target:
        print(f"[passk] Aucun item aux depths {depths} dans le dataset.")
        return []

    print(f"[passk] {len(target)} items aux depths {depths}, k={k}, "
          f"temperature={temperature} → {len(target) * k} rollouts.")

    client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL,
                                data_len=len(target), timeout=60)
    generate_fn = lambda msgs: gen.generate(msgs, max_tokens=EVAL_MAX_TOKENS,  # noqa: E731
                                            temperature=temperature)

    results: list[PassKItemResult] = []
    for i, item in enumerate(target):
        item_id = item["item_id"]
        depth = depth_map[item_id]
        rewards: list[float] = []
        for j in range(k):
            try:
                r = run_episode(generate_fn, client, item_id, system_prompt=system_prompt)
            except Exception as e:
                print(f"[passk][{item_id} d{depth} s{j+1}/{k}] CRASHED: {e}")
                continue
            rewards.append(r.reward)
            if log_dir is not None:
                write_passk_log(r, j, depth, log_dir)
        n_success = sum(1 for rw in rewards if rw > 0)
        passed = n_success > 0
        results.append(PassKItemResult(item_id, depth, rewards, n_success, passed))
        print(f"[passk][{i+1}/{len(target)}] {item_id} d{depth} "
              f"success={n_success}/{len(rewards)} pass@{k}={'Y' if passed else 'N'}")

    return results


def report_pass_at_k(results: list[PassKItemResult], k: int) -> None:
    """Affiche pass@k moyen par depth (fraction d'items résolus ≥ 1 fois sur k)."""
    if not results:
        print("[passk] Aucun résultat.")
        return
    depths = sorted({r.depth for r in results})
    print("\n" + "=" * 52)
    print(f"PASS@{k} PAR DEPTH")
    print(f"{'depth':>5} | {'items':>5} | {'pass@'+str(k):>8} | {'succ/sample':>11}")
    print("-" * 52)
    for d in depths:
        rs = [r for r in results if r.depth == d]
        n_items = len(rs)
        n_passed = sum(1 for r in rs if r.passed)
        total_samples = sum(len(r.rewards) for r in rs)
        total_success = sum(r.n_success for r in rs)
        passk = n_passed / n_items if n_items else 0.0
        per_sample = total_success / total_samples if total_samples else 0.0
        print(f"{d:>5} | {n_items:>5} | {passk:>8.2f} | {per_sample:>11.3f}")
    print("=" * 52)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Chemin local ou id HF Hub (backend vllm : le modèle chargé dans le serveur)")
    parser.add_argument("--run-name", required=True,
                        help="Nom du run — logs dans runs/<run-name>/eval_logs/ "
                             "(peut contenir une famille : '0_baselines/exp…')")
    parser.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                        help="auto = vllm si l'archi est servable par la vLLM installée, sinon hf")
    parser.add_argument("--max-items", type=int, default=0, help="0 = tous les 100 items")
    parser.add_argument("--force-redo", action="store_true", help="Ignore le cache eval_logs")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt. Passer '' pour les modèles sans rôle system (ex. Gemma-3).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Température de génération (protocole de référence : 1.0)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Qwen3/3.5 : enable_thinking=False (actions directes, pas de <think>) — backend hf")
    parser.add_argument("--vllm-url", type=str, default=VLLM_SERVER_URL,
                        help="URL du serveur vLLM (défaut: http://127.0.0.1:8001)")
    parser.add_argument("--fewshot", type=int, default=0,
                        help="Nb d'exemples résolus injectés dans le message de règles "
                             "(0 = zero-shot, protocole historique). Exemples issus des 70 "
                             "recettes HORS train/test (voir build_fewshot_examples.py). "
                             "NB : k>=20 nécessite un serveur vLLM à 32768 "
                             "(start_vllm_server.sh <model> 32768).")
    parser.add_argument("--fewshot-file", type=str,
                        default=str(REPO_ROOT / "data" / "eval" / "textcraft_fewshot_examples.json"),
                        help="Fichier JSON des exemples (défaut : data/eval/textcraft_fewshot_examples.json)")
    parser.add_argument("--fewshot-format", choices=["dialogue", "bloc"], default="dialogue",
                        help="dialogue (défaut) : exemples injectés comme de VRAIS tours "
                             "user/assistant (1 action par tour, format ReAct). "
                             "bloc : texte monolithique dans le message de règles — "
                             "⚠ fait halluciner les observations au modèle (5-8/100 vs "
                             "18 zero-shot, cf. exp18/format_bloc_*), gardé pour l'ablation.")
    parser.add_argument("--passk", type=int, default=0,
                        help="Si >0 : lance le diagnostic pass@K par depth (au lieu du pass@1).")
    parser.add_argument("--passk-depths", type=str, default="3,4",
                        help="Depths ciblées par le diagnostic pass@K (défaut: '3,4').")
    parser.add_argument("--passk-temp", type=float, default=1.0,
                        help="Température d'échantillonnage pour pass@K (>0 pour la diversité).")
    args = parser.parse_args()

    gen = ChatGenerator(model_ref=args.model, backend=args.backend,
                        vllm_url=args.vllm_url, no_thinking=args.no_thinking)

    fewshot_block = None
    fewshot_messages = None
    if args.fewshot > 0:
        import json
        with open(args.fewshot_file) as f:
            examples = json.load(f)
        if args.fewshot > len(examples):
            raise SystemExit(f"--fewshot {args.fewshot} > {len(examples)} exemples disponibles "
                             f"dans {args.fewshot_file}")
        picked = examples[:args.fewshot]

        if args.fewshot_format == "bloc":
            # Format monolithique historique (ablation) — fait halluciner les
            # observations : voir runs/0_baselines/exp18_fewshot/format_bloc_*.
            parts = [f"Here are {args.fewshot} solved example task(s). Follow the same "
                     f"reasoning and action format:"]
            for i, e in enumerate(picked, start=1):
                parts.append(f"=== EXAMPLE {i} ===\n{e['block']}")
            parts.append("=== END OF EXAMPLES ===\nNow solve the new task I will give you.")
            fewshot_block = "\n\n".join(parts)
        else:
            # Format dialogue (ReAct) : chaque exemple devient de vrais tours
            # user (observation) / assistant (Thought + UNE action). L'obs finale
            # de chaque exemple ouvre le message user suivant ; la toute dernière
            # est fusionnée avec la vraie tâche par build_initial_messages.
            fewshot_block = (f"I will first show you {args.fewshot} solved example task(s), "
                             f"then give you a new task to solve.")
            fewshot_messages = []
            carry = None
            for e in picked:
                task = ("Crafting commands:\n" + "\n".join(e["commands"])
                        + f"\n\nGoal: craft {e['goal_str']}.")
                if carry:
                    task = carry + "\n\n" + task
                fewshot_messages.append({"role": "user", "content": task})
                steps = e["steps"]
                for j, st in enumerate(steps):
                    if j == 0:
                        content = f"Thought: {e['thought']}\n\nAction: {st['action']}"
                    else:
                        content = f"Action: {st['action']}"
                    fewshot_messages.append({"role": "assistant", "content": content})
                    if j < len(steps) - 1:
                        fewshot_messages.append({"role": "user", "content": st["observation"]})
                    else:
                        carry = st["observation"]
            if carry:
                fewshot_messages.append({"role": "user", "content": carry})

        from collections import Counter
        n_turns = len(fewshot_messages) if fewshot_messages else 0
        print(f"[fewshot] k={args.fewshot} exemples injectés, format={args.fewshot_format} "
              f"(depths {dict(Counter(e['depth'] for e in picked))}, "
              f"{n_turns} tours)" , flush=True)

    items = load_items(args.max_items)

    # --- Mode diagnostic pass@K par depth (n'exécute PAS le pass@1 standard) ---
    if args.passk > 0:
        if fewshot_block or fewshot_messages:
            raise SystemExit("--passk + --fewshot non combinés pour l'instant "
                             "(le mode passk reste zero-shot).")
        depths = tuple(int(x) for x in args.passk_depths.split(","))
        depth_map = load_depth_map()
        passk_log_dir = RUNS_DIR / args.run_name / "eval_logs_passk"
        passk_log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results = run_pass_at_k_by_depth(
            gen, items, depth_map,
            k=args.passk, depths=depths, temperature=args.passk_temp,
            system_prompt=args.system_prompt, log_dir=passk_log_dir,
        )
        report_pass_at_k(results, args.passk)
        print(f"[passk] Wall time: {time.time() - t0:.1f}s  Logs: {passk_log_dir}/")
        return

    log_dir = RUNS_DIR / args.run_name / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cached, todo = [], []
    for it in items:
        existing = None if args.force_redo else load_existing_log(it["item_id"], log_dir)
        if existing is not None:
            cached.append(existing)
        else:
            todo.append(it)

    print(f"[eval] backend={gen.backend_name} — {len(items)} items : "
          f"{len(cached)} cached, {len(todo)} à évaluer.")

    new_results: list[EpisodeResult] = []
    if todo:
        client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(todo), timeout=60)
        generate_fn = lambda msgs: gen.generate(msgs, max_tokens=EVAL_MAX_TOKENS,  # noqa: E731
                                                temperature=args.temperature)
        overall_t0 = time.time()
        for i, item in enumerate(todo):
            item_id = item["item_id"]
            try:
                r = run_episode(generate_fn, client, item_id, system_prompt=args.system_prompt,
                                fewshot_block=fewshot_block, fewshot_messages=fewshot_messages)
            except Exception as e:
                print(f"[eval][{i+1}/{len(todo)}] {item_id} CRASHED: {e}")
                continue
            write_episode_log(r, log_dir)
            status = "DONE" if r.done else "TIMEOUT"
            print(f"[eval][{i+1}/{len(todo)}] {item_id} reward={r.reward:.0f} "
                  f"rounds={r.rounds} dur={r.duration_s:.1f}s [{status}]")
            new_results.append(r)
        print(f"\n[eval] Wall time: {time.time() - overall_t0:.1f}s")

    all_results = cached + new_results
    if not all_results:
        print("[eval] Aucun résultat.")
        return

    n_pass = sum(1 for r in all_results if r.reward > 0)
    print("=" * 50)
    print(f"RESULTS — {len(all_results)} items ({len(cached)} cached + {len(new_results)} fresh)")
    print(f"Pass@1  = {n_pass}/{len(all_results)}  ({100*n_pass/len(all_results):.1f}%)")
    print(f"Mean rounds   = {sum(r.rounds for r in all_results)/len(all_results):.1f}")
    print(f"Mean duration = {sum(r.duration_s for r in all_results)/len(all_results):.1f}s")
    print(f"Logs: {log_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
