"""Rejeu hors entraînement de l'estimateur KL k3, token par token (étape 2, 15/09/2026).

Question : quand la KL loguée par TRL explose (exp36/40/34), est-ce une dérive
uniforme de la politique ou une poignée de tokens de queue qui portent tout ?
Et l'écart vient-il d'un vrai tirage de queue, ou d'un désaccord numérique entre
vLLM (qui tire les tokens) et HF (qui recalcule les log-probs pour la perte) ?

Protocole, pour UN checkpoint (politique) et SA référence KL exacte :
  1. génération d'épisodes TextCraft avec la politique, par LA boucle d'entraînement
     (`rollout.collect_episodes`, moteur vLLM en process, température 1, 30 tours,
     512 tokens/tour) — on lui passe un « faux trainer » qui expose juste ce qu'elle lit ;
  2. mise à plat au contrat TRL (`episodes_to_trl_batch`) : flux entrelacé
     actions/observations + env_mask, donc le même contexte que le backward ;
  3. recalcul HF de log π (politique) et log π_ref (référence) sur les tokens d'action ;
  4. par token : ρ = log π_ref − log π, k3 = exp(ρ) − ρ − 1, écart |log π_HF − log π_vLLM|.

Sorties : un JSON (agrégats + les tokens les plus lourds avec leur contexte) et un résumé.
Les modèles complets (référence = Qwen ⊕ cycles d'ancre, politique = référence ⊕ adapter)
sont préparés par src/utils/merge_anchor_chain.py.

Usage (env v2, serveur TextCraft sur 36005) :
    python src/analysis/replay_k3_tokens.py --policy /tmp/models/replay/exp40_s705_policy \
        --ref /tmp/models/replay/exp40_s705_ref --tag exp40_s705 --n-items 32 --episodes-per-item 2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.train import data, rollout, schedules  # noqa: E402

LOG_1E4 = math.log(1e-4)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="modèle HF complet de la politique (référence ⊕ adapter)")
    ap.add_argument("--ref", required=True, help="modèle HF complet de la référence KL (Qwen ⊕ cycles d'ancre)")
    ap.add_argument("--tag", required=True, help="nom court du checkpoint (fichier de sortie)")
    ap.add_argument("--n-items", type=int, default=32, help="tâches d'entraînement tirées au sort")
    ap.add_argument("--episodes-per-item", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="même tirage de tâches pour tous les checkpoints")
    ap.add_argument("--max-rounds", type=int, default=30, help="cap de tours (= '30:0' des runs)")
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--batch-episodes", type=int, default=64, help="épisodes joués en parallèle")
    ap.add_argument("--gpu-util", type=float, default=0.3, help="fraction GPU réservée au moteur vLLM")
    ap.add_argument("--top-k", type=int, default=30, help="tokens les plus lourds détaillés dans le JSON")
    ap.add_argument("--out-dir", default="runs/15_replay_k3")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 1-2. génération par la boucle d'entraînement + mise à plat TRL
# ---------------------------------------------------------------------------

def make_fake_trainer(tokenizer, llm, max_completion_length: int) -> SimpleNamespace:
    """Le strict nécessaire lu par rollout.collect_episodes / vllm_engine.generate_round :
    processing_class, args.max_completion_length, vllm_generation.llm, state (None =
    pas de schedule, cap fixe)."""
    return SimpleNamespace(processing_class=tokenizer,
                           args=SimpleNamespace(max_completion_length=max_completion_length),
                           vllm_generation=SimpleNamespace(llm=llm),
                           state=None)


def stream_with_meta(ep: rollout.Episode, tokenizer, max_completion_length: int
                     ) -> tuple[list[int], list[float], list[int], list[tuple]]:
    """Même mise à plat que rollout.extend_stream, plus une méta par position :
    (turn_idx 1-based, pos_in_turn, turn_len, forced) pour les tokens d'action, None sinon.
    forced = le <|im_end|> ajouté par collect_episodes après une troncature à max_tokens
    (tour de max_completion_length + 1 tokens) : il n'a PAS été tiré par vLLM."""
    prefix = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)
    ids: list[int] = []
    lps: list[float] = []
    mask: list[int] = []
    meta: list[tuple | None] = []
    for k, turn in enumerate(ep.turns, start=1):
        n_before = len(ids)
        rollout.extend_stream(ids, lps, mask, turn, k, turn.logprobs, prefix)
        added = len(ids) - n_before
        n_prefix = len(prefix) if k > 1 else 0
        meta.extend([None] * n_prefix)
        truncated = len(turn.ids) > max_completion_length
        meta.extend([(k, j, len(turn.ids), truncated and j == len(turn.ids) - 1) for j in range(len(turn.ids))])
        meta.extend([None] * (added - n_prefix - len(turn.ids)))
    assert len(meta) == len(ids) == len(mask)
    return ids, lps, mask, meta


# ---------------------------------------------------------------------------
# 3. recalcul HF des log-probs (politique et référence) sur le flux complet
# ---------------------------------------------------------------------------

@torch.no_grad()
def token_logps(model, prompt_ids: list[int], completion_ids: list[int], want_entropy: bool,
                chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor | None]:
    """log p(token_t | contexte) pour chaque token de completion_ids, contexte =
    prompt + tokens précédents (exactement ce que TRL calcule, température 1).
    log_softmax par tranches de positions pour ne pas matérialiser 10k × 151k en float32."""
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    logits = model(input_ids=ids).logits[0]                     # [L, V] bf16
    P = len(prompt_ids)
    pred = logits[P - 1: P - 1 + len(completion_ids)]            # prédit completion_ids[t]
    tgt = torch.tensor(completion_ids, device=model.device)
    out_lp, out_ent = [], []
    for s in range(0, pred.shape[0], chunk):
        lsm = torch.log_softmax(pred[s:s + chunk].float(), dim=-1)
        out_lp.append(lsm.gather(1, tgt[s:s + chunk, None])[:, 0])
        if want_entropy:
            out_ent.append(-(lsm.exp() * lsm).sum(-1))
    return torch.cat(out_lp), (torch.cat(out_ent) if want_entropy else None)


# ---------------------------------------------------------------------------
# 4. agrégats
# ---------------------------------------------------------------------------

def quantiles(xs: list[float], qs=(0.5, 0.9, 0.99, 0.999)) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{int(q * 1000) / 10:g}": s[min(len(s) - 1, int(q * len(s)))] for q in qs} | {"max": s[-1]}


def main() -> None:
    args = parse_args()
    import os
    os.environ.setdefault("PATH", "")
    schedules.MAX_SIM_ROUNDS = args.max_rounds            # équivalent de --max-rounds-schedule '30:0'
    data.check_server()

    tokenizer = AutoTokenizer.from_pretrained(args.policy)
    rows = data.build_prompt_rows(max_items=0)
    rng = random.Random(args.seed)
    picked = rng.sample(rows, args.n_items)
    # copie PROFONDE : collect_episodes remplace le marqueur <ITEM_IDX:n> en place dans le dict du message
    prompts = [copy.deepcopy(r["prompt"]) for r in picked for _ in range(args.episodes_per_item)]
    print(f"[replay] {len(picked)} tâches × {args.episodes_per_item} = {len(prompts)} épisodes, "
          f"tâches : {[r['item_id'] for r in picked]}", flush=True)

    # --- génération (moteur vLLM en process, comme le colocate de TRL)
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=args.policy, dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
              max_model_len=32768, enable_prefix_caching=True, seed=args.seed)
    trainer = make_fake_trainer(tokenizer, llm, args.max_completion_length)
    episodes: list[rollout.Episode] = []
    for s in range(0, len(prompts), args.batch_episodes):
        episodes += rollout.collect_episodes(prompts[s:s + args.batch_episodes], trainer)
        print(f"[replay] {len(episodes)}/{len(prompts)} épisodes joués ({time.time() - t0:.0f} s)", flush=True)
    rewards = [ep.reward for ep in episodes]
    print(f"[replay] reward moyen {sum(rewards) / len(rewards):.3f}, "
          f"tours moyens {sum(len(ep.turns) for ep in episodes) / len(episodes):.1f}", flush=True)
    del llm, trainer
    torch.cuda.empty_cache()

    # --- recalcul HF
    policy = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16, device_map="cuda").eval()
    ref = AutoModelForCausalLM.from_pretrained(args.ref, dtype=torch.bfloat16, device_map="cuda").eval()
    tokens: list[dict] = []          # un enregistrement par token d'action
    ep_summaries: list[dict] = []
    for e_idx, ep in enumerate(episodes):
        ids, lps_vllm, mask, meta = stream_with_meta(ep, tokenizer, args.max_completion_length)
        lp_pi, ent = token_logps(policy, ep.prompt_ids, ids, want_entropy=True)
        lp_ref, _ = token_logps(ref, ep.prompt_ids, ids, want_entropy=False)
        lp_pi, lp_ref, ent = lp_pi.tolist(), lp_ref.tolist(), ent.tolist()
        ep_k3 = []
        for t in range(len(ids)):
            if mask[t] != 1:
                continue
            rho = lp_ref[t] - lp_pi[t]
            k3 = math.exp(min(rho, 80.0)) - rho - 1.0
            turn_idx, pos, tlen, forced = meta[t]
            tokens.append(dict(ep=e_idx, t=t, tok=ids[t], turn=turn_idx, pos=pos, turn_len=tlen,
                               lp_pi=lp_pi[t], lp_ref=lp_ref[t], lp_vllm=lps_vllm[t], rho=rho, k3=k3,
                               ent=ent[t], forced=forced, reward=ep.reward))
            ep_k3.append(k3)
        ep_summaries.append(dict(ep=e_idx, item_idx=ep.item_idx, reward=ep.reward, turns=len(ep.turns),
                                 n_action_tokens=len(ep_k3), k3_mean=sum(ep_k3) / max(1, len(ep_k3)),
                                 k3_max=max(ep_k3) if ep_k3 else 0.0))
        if e_idx % 8 == 0:
            print(f"[replay] logps {e_idx + 1}/{len(episodes)} ({time.time() - t0:.0f} s)", flush=True)

    # --- agrégats
    n = len(tokens)
    k3s = [x["k3"] for x in tokens]
    total = sum(k3s)
    srt = sorted(tokens, key=lambda x: -x["k3"])
    share = lambda m: sum(x["k3"] for x in srt[:m]) / total if total > 0 else 0.0
    sampled = [x for x in tokens if not x["forced"]]
    k3_pos = sum(x["k3"] for x in tokens if x["rho"] > 0)      # côté exponentiel : référence > politique (tirage de queue)
    k3_neg = sum(x["k3"] for x in tokens if x["rho"] < 0)      # côté linéaire : politique sûre, référence en désaccord (dérive)
    n_turns = sum(len(ep.turns) for ep in episodes)
    n_trunc = sum(len(t.ids) > args.max_completion_length for ep in episodes for t in ep.turns)
    gaps = [abs(x["lp_pi"] - x["lp_vllm"]) for x in sampled]
    summary = dict(
        tag=args.tag, policy=args.policy, ref=args.ref, n_episodes=len(episodes), n_items=args.n_items,
        reward_mean=sum(rewards) / len(rewards), n_action_tokens=n,
        kl_k3_mean=total / max(1, n),                       # = la 'kl' loguée par TRL
        entropy_mean=sum(x["ent"] for x in tokens) / max(1, n),   # = l''entropy' loguée par TRL
        k3_share_top1=share(1), k3_share_top10=share(10), k3_share_top1pct=share(max(1, n // 100)),
        k3_share_rho_pos=k3_pos / total if total > 0 else 0.0, k3_share_rho_neg=k3_neg / total if total > 0 else 0.0,
        n_turns=n_turns, n_truncated_turns=n_trunc,
        n_rho_gt5=sum(x["rho"] > 5 for x in tokens), n_rho_gt10=sum(x["rho"] > 10 for x in tokens),
        n_rho_lt_m5=sum(x["rho"] < -5 for x in tokens),
        frac_lp_pi_below_1e4=sum(x["lp_pi"] < LOG_1E4 for x in tokens) / max(1, n),
        frac_lp_ref_below_1e4=sum(x["lp_ref"] < LOG_1E4 for x in tokens) / max(1, n),
        rho_quantiles=quantiles([x["rho"] for x in tokens]),
        vllm_hf_gap_quantiles=quantiles(gaps),
        n_vllm_hf_gap_gt2=sum(g > 2 for g in gaps), n_sampled=len(sampled),
        n_forced_im_end=n - len(sampled),
    )
    def describe(x: dict) -> dict:
        ep = episodes[x["ep"]]
        ids, _, _, _ = stream_with_meta(ep, tokenizer, args.max_completion_length)
        ctx = tokenizer.decode(ids[max(0, x["t"] - 24):x["t"]])
        nxt = tokenizer.decode(ids[x["t"] + 1:x["t"] + 6])
        return x | dict(text=tokenizer.decode([x["tok"]]), context_before=ctx, context_after=nxt,
                        p_pi=math.exp(x["lp_pi"]), p_ref=math.exp(x["lp_ref"]), p_vllm=math.exp(x["lp_vllm"]))
    top = [describe(x) for x in srt[:args.top_k]]
    # tokens où vLLM (tirage) et HF (perte) divergent le plus : candidats « désaccord numérique »
    top_gap = [describe(x) for x in sorted(sampled, key=lambda x: -abs(x["lp_pi"] - x["lp_vllm"]))[:args.top_k]]

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}.json"
    with out.open("w") as f:
        json.dump(dict(summary=summary, episodes=ep_summaries, top_tokens=top, top_gap_tokens=top_gap,
                       # histogramme compact de tous les tokens pour les figures (pas de texte)
                       tokens=[[x["lp_pi"], x["lp_ref"], x["lp_vllm"], x["ent"], x["turn"], x["pos"], x["turn_len"],
                                int(x["forced"]), x["reward"], x["tok"], x["ep"]] for x in tokens],
                       tokens_cols=["lp_pi", "lp_ref", "lp_vllm", "ent", "turn", "pos", "turn_len", "forced", "reward",
                                    "tok", "ep"]),
                  f)

    print("\n[replay] ===== résumé", args.tag)
    for k in ("n_episodes", "reward_mean", "n_action_tokens", "kl_k3_mean", "entropy_mean",
              "k3_share_top1", "k3_share_top10", "k3_share_top1pct", "k3_share_rho_pos", "k3_share_rho_neg",
              "n_turns", "n_truncated_turns", "n_rho_gt5", "n_rho_gt10", "n_rho_lt_m5",
              "frac_lp_pi_below_1e4", "frac_lp_ref_below_1e4", "n_vllm_hf_gap_gt2", "n_forced_im_end"):
        v = summary[k]
        print(f"  {k:24s} {v:.4g}" if isinstance(v, float) else f"  {k:24s} {v}")
    print(f"  rho quantiles           {summary['rho_quantiles']}")
    print(f"  |lp_HF - lp_vLLM|       {summary['vllm_hf_gap_quantiles']}")
    print("[replay] top tokens par k3 :")
    for x in top[:12]:
        print(f"  k3={x['k3']:.3g} rho={x['rho']:.2f} p_pi={x['p_pi']:.2e} p_ref={x['p_ref']:.2e} "
              f"p_vllm={x['p_vllm']:.2e} tour {x['turn']} pos {x['pos']}/{x['turn_len']} "
              f"reward={x['reward']:.0f} tok={x['text']!r} | …{x['context_before'][-60:]!r}")
    print("[replay] top tokens par écart |lp_HF - lp_vLLM| :")
    for x in top_gap[:8]:
        print(f"  gap={abs(x['lp_pi'] - x['lp_vllm']):.2f} p_pi={x['p_pi']:.2e} p_ref={x['p_ref']:.2e} p_vllm={x['p_vllm']:.2e} "
              f"rho={x['rho']:.2f} tour {x['turn']} pos {x['pos']}/{x['turn_len']} tok={x['text']!r} | "
              f"…{x['context_before'][-50:]!r} -> {x['context_after'][:20]!r}")
    print(f"[replay] écrit : {out}")


if __name__ == "__main__":
    main()
