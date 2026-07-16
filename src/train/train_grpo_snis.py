"""GRPO multi-tour sur TextCraft + recombinaison SNIS des tours (v1).

Idée (cf. note LaTeX "L'inversion entre a et z n'est pas utile", section 3) :
à partir des G trajectoires générées par GRPO pour un prompt x, on fabrique de
nouvelles trajectoires ARTIFICIELLES en recombinant les tours entre trajectoires,
et on quantifie par Self-Normalized Importance Sampling (SNIS) la probabilité que
la politique ait produit chaque tour candidat dans le contexte mixé. L'estimateur
du gradient (formule forward) :

    grad_V ≈ (1/G) Σ_{s1} Σ_{s2} ω(s2|s1) ... Σ_{sN} ω(sN|s1..sN-1)
             · ∇log π((z^k_{sk})_k | x) · r(x, z^N_{sN})

    avec ω(sk | prefixe) = π(z^k_{sk} | x, prefixe) / Σ_l π(z^k_l | x, prefixe)

Implémentation : la somme complète a G^N termes -> intraitable. On l'estime par
ÉCHANTILLONNAGE ANCESTRAL : s1 est énuméré exactement (le combo m démarre sur le
tour 1 de la trajectoire m, ce qui couvre la somme externe (1/G) Σ_{s1}), puis à
chaque tour k >= 2 on tire s_k ~ ω(·|prefixe). Comme on échantillonne PROPORTION-
NELLEMENT aux poids SNIS, ceux-ci sont absorbés par l'échantillonnage : chaque
trajectoire combinée entre dans la loss avec un poids uniforme, et la loss GRPO
de TRL (ratio ≡ 1 en on-policy aligné => gradient = ∇log π × avantage sur les
tokens env_mask=1) reste STRICTEMENT INCHANGÉE. Seule la rollout_func change.

Approximations assumées de cette v1 (à discuter/critiquer) :
  1. Observations : quand on greffe le tour k de la trajectoire j sur un préfixe
     mixé, on garde l'observation env QUE j AVAIT REÇUE (pas de re-simulation).
     Le contexte mixé peut donc être incohérent avec l'état réel de l'env ; le
     poids SNIS pénalise naturellement les greffes improbables, mais l'obs reste
     contrefactuelle. (v2 possible : rejouer les actions dans TextCraft, qui est
     déterministe et peu coûteux, pour obtenir obs + reward VRAIS.)
  2. Reward : r(x, z^N_{sN}) = reward de la trajectoire SOURCE du dernier tour
     consommé (cohérent avec la formule forward de la note).
  3. Terminaison : un combo se termine quand il consomme le DERNIER tour de sa
     trajectoire source (done env ou cap de rounds), ou quand le contexte dépasse
     --snis-max-ctx-tokens.
  4. Biais longueur : ω est un ratio de probas de SÉQUENCES -> favorise les tours
     courts. L'exposant α de la note (--snis-alpha, ω ∝ π^α renormalisé) permet
     d'aplatir (α<1) ou de durcir (α>1) les poids. Diagnostics ESS + Δlen loggés
     à chaque step pour surveiller un éventuel effondrement des poids.

Découplage M/G (efficience en données, le cœur de la méthode) : --num-generations
est M = nb de trajectoires COMBINÉES par prompt entrant dans la loss (taille de
groupe TRL pour les avantages), --snis-real-rollouts est G = nb d'épisodes RÉELS
joués dans l'env par prompt. Avec M > G (multiple), on met M tirages de la somme
interne dans le gradient pour le même coût d'environnement : variance divisée par
~M/G. s1 cycle sur m mod G pour garder la somme externe (1/G) Σ_{s1} équilibrée.

Usage : mêmes arguments que train_grpo.py (le main() délègue à train_grpo.main()
après avoir substitué la rollout_func), plus :
    --snis-alpha 1.0             exposant α sur les poids SNIS
    --snis-real-rollouts 0       G = épisodes réels joués par prompt ; 0 = M
                                 (pas de découplage, mode v1 : autant de combos
                                 que de rollouts)
    --snis-keep-originals 0      nb de trajectoires originales gardées telles
                                 quelles parmi les M sorties (0 = SNIS pur,
                                 doit être <= G)
    --snis-score-batch-size 4    batch du forward HF de scoring
    --snis-max-ctx-tokens 15000  budget tokens du contexte d'un combo
    --snis-seed 0                graine du tirage ancestral (déterministe par step)
    --selftest                   test CPU de la logique de recombinaison (sans GPU,
                                 sans serveur TextCraft) puis exit

Exemple (M=32 combos par prompt, G=8 rollouts réels, soit 4x plus de données de
gradient que GRPO classique pour le même coût d'environnement) :
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_grpo_snis.py --full-ft --num-generations 32 \
        --snis-real-rollouts 8 --gradient-accumulation-steps 256 \
        --max-completion-length 512 --max-items 0 --max-steps 200 \
        --use-vllm-inprocess --snis-alpha 1.0 --run-name exp17_snis_v1
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

# Réutilisation directe de train_grpo.py (helpers, globals vLLM, main).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_grpo as tg

from trl.trainer.utils import selective_log_softmax

try:
    from trl.models.utils import disable_gradient_checkpointing
except ImportError:  # version TRL sans ce helper : warning cosmétique possible
    def disable_gradient_checkpointing(model, kwargs=None):
        return nullcontext()


# Hyperparamètres SNIS (module-level, fixés par main() depuis la CLI).
SNIS_ALPHA = 1.0
SNIS_REAL_ROLLOUTS = 0   # G = épisodes réels par prompt ; 0 = num_generations (M=G)
SNIS_KEEP_ORIGINALS = 0
SNIS_SCORE_BATCH = 4
SNIS_MAX_CTX_TOKENS = 15000
SNIS_SEED = 0


# ---------------------------------------------------------------------------
# 1) Collecte de rollouts STRUCTURÉS PAR TOUR
#    (copie adaptée de tg.textcraft_rollout_func : même génération, même env,
#     mais on garde la granularité tour-par-tour au lieu d'un flux aplati)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    ids: list[int]            # tokens assistant du tour (+ <|im_end|> forcé si tronqué)
    logprobs: list[float]     # logprobs de génération (0.0 sur l'im_end forcé)
    obs_ids: list[int]        # tokens du bloc obs "\n<|im_start|>user\n{obs}<|im_end|>"
    action_count: int         # nb d'actions détectées dans le texte du tour
    invalid: bool             # l'obs signale une action invalide


@dataclass
class Episode:
    prompt_ids: list[int]     # prompt rendu au round 0 (finit par le generation prompt)
    turns: list[Turn]
    reward: float


def collect_episodes(prompts: list[list[dict[str, str]]], trainer: Any) -> list[Episode]:
    """Joue les épisodes comme tg.textcraft_rollout_func mais retourne la structure par tour."""
    from concurrent.futures import ThreadPoolExecutor

    tokenizer = trainer.processing_class
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    n = len(prompts)
    states = [list(p) for p in prompts]
    done = [False] * n
    episodes = [Episode(prompt_ids=[], turns=[], reward=0.0) for _ in range(n)]
    env_clients = []

    for i in range(n):
        marker = str(states[i][-1]["content"]).strip()
        m = tg.ITEM_TAG_RE.match(marker)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}] last user message: {marker!r}")
        env = tg.TextCraftEnvClient(env_server_base=tg.ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(int(m.group(1)))
        states[i][-1]["content"] = env.observe()
        env_clients.append(env)

    cap = tg.current_max_rounds(trainer)

    with ThreadPoolExecutor(max_workers=n) as executor:
        for round_idx in range(cap):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            round_prompt_ids = []
            for i in active:
                rendered = tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
                pids = tokenizer.encode(rendered, add_special_tokens=False)
                round_prompt_ids.append(pids)
                if round_idx == 0:
                    episodes[i].prompt_ids = pids

            if tg.VLLM_LLM is not None:
                round_completion_ids, round_logprobs = tg._vllm_generate_round(
                    tokenizer, states, active, max_tokens=trainer.args.max_completion_length
                )
            else:
                round_completion_ids, round_logprobs = trainer._generate_single_turn(
                    round_prompt_ids, images=None, multimodal_fields={}
                )

            actions = {}
            for j, i in enumerate(active):
                ids = list(round_completion_ids[j])
                lps = list(round_logprobs[j][: len(ids)]) if round_logprobs is not None else [0.0] * len(ids)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                states[i].append({"role": "assistant", "content": text})
                actions[i] = text
                if not ids or ids[-1] != tokenizer.eos_token_id:
                    # Tour tronqué par max_tokens : on ferme quand même le tour
                    # (même convention que l'original : <|im_end|> avec env_mask=1, lp=0).
                    ids = ids + im_end_ids
                    lps = lps + [0.0] * len(im_end_ids)
                episodes[i].turns.append(Turn(ids=ids, logprobs=lps, obs_ids=[],
                                              action_count=tg.count_actions(text), invalid=False))

            futures = [(executor.submit(env_clients[i].step, actions[i]), i) for i in active]
            step_results = {i: fut.result() for fut, i in futures}

            for i in active:
                out = step_results[i]
                obs = out.state
                states[i].append({"role": "user", "content": obs})
                episodes[i].reward = max(episodes[i].reward, float(out.reward))
                low = obs.lower()
                turn = episodes[i].turns[-1]
                turn.invalid = ("could not" in low or "error:" in low or "wrong item format" in low)
                obs_block = f"\n<|im_start|>user\n{obs}<|im_end|>"
                turn.obs_ids = tokenizer.encode(obs_block, add_special_tokens=False)
                done[i] = bool(out.done)

    for env in env_clients:
        try:
            env.close()
        except Exception:
            pass

    return episodes


# ---------------------------------------------------------------------------
# 2) Scoring des tours candidats : log π_θ(tour | contexte mixé)
#    Forward HF batché sous no_grad, left padding (aligne les suffixes à droite),
#    logits_to_keep pour ne matérialiser les logits que sur le suffixe scoré.
# ---------------------------------------------------------------------------

def score_candidate_turns(model: Any, jobs: list[tuple[list[int], list[int]]],
                          batch_size: int, pad_id: int) -> list[list[float]]:
    """Pour chaque job (ctx_ids, cand_ids), retourne les logprobs par token de
    cand_ids sous le contexte ctx_ids (politique courante θ). ctx_ids doit finir
    par le generation prompt ("\\n<|im_start|>assistant\\n")."""
    if not jobs:
        return []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results: list[list[float] | None] = [None] * len(jobs)
    # Tri par longueur totale pour limiter le padding intra-batch.
    order = sorted(range(len(jobs)), key=lambda t: len(jobs[t][0]) + len(jobs[t][1]))

    try:
        gc_ctx = disable_gradient_checkpointing(model)
    except Exception:
        gc_ctx = nullcontext()

    with torch.no_grad(), gc_ctx:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            seqs = [jobs[t][0] + jobs[t][1] for t in chunk]
            cand_lens = [len(jobs[t][1]) for t in chunk]
            L = max(len(s) for s in seqs)
            K = max(cand_lens) + 1  # positions de logits nécessaires (suffixe)

            input_ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            for b, s in enumerate(seqs):
                input_ids[b, L - len(s):] = torch.tensor(s, dtype=torch.long)
                attn[b, L - len(s):] = 1
            # position_ids explicites : avec left padding, ne pas laisser le
            # default arange compter les tokens de padding.
            pos = (attn.cumsum(-1) - 1).clamp(min=0)

            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        position_ids=pos.to(device), logits_to_keep=K, use_cache=False)
            logits = out.logits  # [B, K, V] — K dernières positions de chaque ligne

            for b, t in enumerate(chunk):
                c = cand_lens[b]
                # Le token à la position paddée p est prédit par les logits en p-1.
                # cand occupe les positions [L-c, L) ; ses prédicteurs [L-c-1, L-2],
                # soit, dans la fenêtre des K dernières positions : [K-1-c, K-1).
                sl = logits[b, K - 1 - c: K - 1, :].float()
                idx = torch.tensor(jobs[t][1], dtype=torch.long, device=sl.device)
                results[t] = selective_log_softmax(sl, idx).tolist()

    model.train(was_training)
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3) Recombinaison ancestrale SNIS
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    group: int
    prompt_ids: list[int]
    stream_ids: list[int] = field(default_factory=list)
    stream_lps: list[float] = field(default_factory=list)
    stream_mask: list[int] = field(default_factory=list)
    chosen: list[tuple[int, int]] = field(default_factory=list)  # (source j, tour k 1-based)
    reward: float = 0.0
    finished: bool = False
    end_reason: str = ""       # "source_end" | "budget" | "original"
    invalid_steps: int = 0
    action_counts: list[int] = field(default_factory=list)
    n_switches: int = 0


def _append_turn(combo: Combo, ep_turn: Turn, j: int, k: int,
                 lps: list[float], assistant_prefix_ids: list[int]) -> None:
    """Ajoute un tour (tokens assistant + bloc obs) au flux du combo, avec les
    mêmes conventions de masque que tg.textcraft_rollout_func : generation prompt
    et obs masqués (env_mask=0), tokens d'action actifs (env_mask=1)."""
    if k > 1:
        combo.stream_ids.extend(assistant_prefix_ids)
        combo.stream_lps.extend([0.0] * len(assistant_prefix_ids))
        combo.stream_mask.extend([0] * len(assistant_prefix_ids))
    combo.stream_ids.extend(ep_turn.ids)
    combo.stream_lps.extend(lps[: len(ep_turn.ids)] + [0.0] * max(0, len(ep_turn.ids) - len(lps)))
    combo.stream_mask.extend([1] * len(ep_turn.ids))
    combo.stream_ids.extend(ep_turn.obs_ids)
    combo.stream_lps.extend([0.0] * len(ep_turn.obs_ids))
    combo.stream_mask.extend([0] * len(ep_turn.obs_ids))
    if combo.chosen and combo.chosen[-1][0] != j:
        combo.n_switches += 1
    combo.chosen.append((j, k))
    combo.action_counts.append(ep_turn.action_count)
    combo.invalid_steps += int(ep_turn.invalid)


def _snis_weights(logps: list[float], alpha: float) -> list[float]:
    """ω_l ∝ exp(α · log π_l), renormalisé — SNIS avec exposant α de la note
    ((π/Σπ)^α renormalisé ≡ softmax(α·logπ))."""
    scaled = [alpha * lp for lp in logps]
    mx = max(scaled)
    exps = [math.exp(s - mx) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def snis_recombine(groups: list[list[Episode]],
                   score_fn: Callable[[list[tuple[list[int], list[int]]]], list[list[float]]],
                   rng: random.Random, alpha: float, keep_originals: int,
                   max_ctx_tokens: int, assistant_prefix_ids: list[int],
                   combos_per_group: int | None = None,
                   record_weights: bool = False) -> tuple[list[Combo], dict[str, Any]]:
    """Construit M = combos_per_group combos par groupe par échantillonnage
    ancestral SNIS (None = autant que d'épisodes réels : mode v1, M=G).

    - combos [0, keep_originals) : trajectoires originales rejouées telles quelles ;
    - combos [keep_originals, M) : tour 1 = celui de l'épisode s1 = m mod G
      (chaque valeur de s1 utilisée M/G fois -> somme externe (1/G) Σ_{s1}
      équilibrée), tours suivants tirés ~ ω(·|préfixe mixé). Deux combos partant
      du même s1 divergent par leurs tirages aux tours >= 2.
    Tous les combos d'un round k sont scorés dans UN SEUL appel à score_fn
    (batching global sur les groupes), ce qui borne le nb de forwards à
    max_tours × ceil(jobs/batch).
    """
    diag: dict[str, Any] = {"n_jobs": 0, "tok_scored": 0, "ess": [], "w_max": [],
                            "dlen": [], "weights": [] if record_weights else None}
    combos: list[Combo] = []

    for g, eps in enumerate(groups):
        n_real = len(eps)
        n_combos = combos_per_group or n_real
        for m in range(n_combos):
            s1 = m % n_real  # cycle sur les G épisodes réels (m < keep_originals => s1 = m)
            c = Combo(group=g, prompt_ids=eps[s1].prompt_ids)
            if m < keep_originals:
                # Trajectoire originale intacte (ancre on-policy).
                for k, turn in enumerate(eps[m].turns, start=1):
                    _append_turn(c, turn, m, k, turn.logprobs, assistant_prefix_ids)
                c.reward, c.finished, c.end_reason = eps[m].reward, True, "original"
            else:
                turn1 = eps[s1].turns[0]
                _append_turn(c, turn1, s1, 1, turn1.logprobs, assistant_prefix_ids)
                if len(eps[s1].turns) == 1:
                    # Le tour 1 était le dernier de sa source -> combo terminé.
                    c.reward, c.finished, c.end_reason = eps[s1].reward, True, "source_end"
            combos.append(c)

    k = 2
    while True:
        active = [c for c in combos if not c.finished]
        if not active:
            break

        # Budget contexte : on coupe AVANT de scorer un round de plus.
        for c in active:
            if len(c.prompt_ids) + len(c.stream_ids) > max_ctx_tokens:
                j_last = c.chosen[-1][0]
                c.reward = groups[c.group][j_last].reward
                c.finished, c.end_reason = True, "budget"
        active = [c for c in combos if not c.finished]
        if not active:
            break

        # Un job de scoring par (combo actif, tour candidat).
        jobs: list[tuple[list[int], list[int]]] = []
        meta: list[tuple[int, int]] = []  # (index dans active, source j)
        cand_sets: list[list[int]] = []
        for a, c in enumerate(active):
            eps = groups[c.group]
            cands = [j for j in range(len(eps)) if len(eps[j].turns) >= k]
            # Invariant : le combo n'est actif que si sa source du tour k-1 avait
            # encore des tours après k-1 -> au moins elle est candidate.
            assert cands, f"combo {a} actif au tour {k} sans candidat"
            cand_sets.append(cands)
            ctx = c.prompt_ids + c.stream_ids + assistant_prefix_ids
            for j in cands:
                jobs.append((ctx, eps[j].turns[k - 1].ids))
                meta.append((a, j))

        results = score_fn(jobs)
        diag["n_jobs"] += len(jobs)
        diag["tok_scored"] += sum(len(ctx) + len(cd) for ctx, cd in jobs)

        # Regroupe les logprobs par combo actif.
        per_combo: list[dict[int, list[float]]] = [{} for _ in active]
        for (a, j), lps in zip(meta, results):
            per_combo[a][j] = lps

        for a, c in enumerate(active):
            eps = groups[c.group]
            cands = cand_sets[a]
            sum_lps = [sum(per_combo[a][j]) for j in cands]
            w = _snis_weights(sum_lps, alpha)
            if diag["weights"] is not None:
                diag["weights"].append(list(w))
            ess = 1.0 / sum(x * x for x in w)
            diag["ess"].append(ess / len(cands))
            diag["w_max"].append(max(w))
            pick = rng.choices(range(len(cands)), weights=w, k=1)[0]
            j_star = cands[pick]
            lens = [len(eps[j].turns[k - 1].ids) for j in cands]
            diag["dlen"].append(lens[pick] - sum(lens) / len(lens))

            _append_turn(c, eps[j_star].turns[k - 1], j_star, k,
                         per_combo[a][j_star], assistant_prefix_ids)
            if len(eps[j_star].turns) == k:
                # Dernier tour de la source consommé -> fin, reward hérité.
                c.reward, c.finished, c.end_reason = eps[j_star].reward, True, "source_end"
        k += 1

    return combos, diag


# ---------------------------------------------------------------------------
# 4) rollout_func SNIS : collecte -> recombinaison -> contrat TRL
# ---------------------------------------------------------------------------

def textcraft_snis_rollout_func(prompts: list[list[dict[str, str]]], trainer: Any) -> dict[str, Any]:
    tokenizer = trainer.processing_class
    assistant_prefix_ids = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)
    # M = combos par prompt entrant dans la loss (taille de groupe TRL pour les
    # avantages) ; G = épisodes réellement joués dans l'env par prompt.
    M = trainer.num_generations
    G = SNIS_REAL_ROLLOUTS if SNIS_REAL_ROLLOUTS > 0 else M
    if M % G != 0:
        raise ValueError(f"num_generations (M={M}) doit être multiple de "
                         f"--snis-real-rollouts (G={G})")
    if SNIS_KEEP_ORIGINALS > G:
        raise ValueError(f"--snis-keep-originals ({SNIS_KEEP_ORIGINALS}) > G ({G}) : "
                         f"on ne peut pas garder plus d'originaux que de rollouts réels")
    n = len(prompts)
    if n % M != 0:
        raise ValueError(f"len(prompts)={n} non divisible par num_generations={M}")
    n_groups = n // M

    # TRL envoie M copies consécutives de chaque prompt (RepeatSampler) ; on ne
    # joue réellement que les G premières de chaque groupe — c'est LE point où
    # M combos coûtent moins cher que M rollouts.
    real_rows = [prompts[g * M + i] for g in range(n_groups) for i in range(G)]

    t0 = time.time()
    episodes = collect_episodes(real_rows, trainer)
    t_rollout = time.time() - t0

    groups = [episodes[g * G:(g + 1) * G] for g in range(n_groups)]
    for g, eps in enumerate(groups):
        if any(e.prompt_ids != eps[0].prompt_ids for e in eps):
            raise ValueError(f"Groupe {g}: prompts non identiques — l'ordre TRL "
                             f"(RepeatSampler, {M} copies consécutives) est requis.")

    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    rng = random.Random(SNIS_SEED * 1_000_003 + step)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def score_fn(jobs):
        return score_candidate_turns(trainer.model, jobs, SNIS_SCORE_BATCH, pad_id)

    t1 = time.time()
    combos, diag = snis_recombine(groups, score_fn, rng, SNIS_ALPHA, SNIS_KEEP_ORIGINALS,
                                  SNIS_MAX_CTX_TOKENS, assistant_prefix_ids,
                                  combos_per_group=M)
    t_snis = time.time() - t1

    # Assemblage au contrat TRL (mêmes clés que tg.textcraft_rollout_func).
    prompt_ids_out, completion_ids_out, logprobs_out, env_mask_out = [], [], [], []
    rewards_out, invalid_out, mean_actions_out = [], [], []
    for c in combos:
        if not c.stream_ids:  # garde-fou (ne devrait jamais arriver : tour 1 toujours présent)
            c.stream_ids, c.stream_lps, c.stream_mask = [tokenizer.eos_token_id], [0.0], [1]
        prompt_ids_out.append(c.prompt_ids or [tokenizer.eos_token_id])
        completion_ids_out.append(c.stream_ids)
        logprobs_out.append(c.stream_lps)
        env_mask_out.append(c.stream_mask)
        rewards_out.append(c.reward)
        invalid_out.append(c.invalid_steps)
        mean_actions_out.append(sum(c.action_counts) / len(c.action_counts) if c.action_counts else 1.0)

    # Diagnostics SNIS (ESS ~1 => poids uniformes ; ESS ~1/G => effondrement).
    n_dec = max(1, len(diag["ess"]))
    ends = {r: sum(1 for c in combos if c.end_reason == r) for r in ("original", "source_end", "budget")}
    orig_rewards = [round(e.reward, 2) for eps in groups for e in eps]
    combo_turns = [len(c.chosen) for c in combos]
    print(
        f"[snis] step={step} groups={len(groups)} G_reel={G} M={M} "
        f"α={SNIS_ALPHA} keep={SNIS_KEEP_ORIGINALS} | "
        f"tours/combo={sum(combo_turns)/len(combos):.1f} switches/combo="
        f"{sum(c.n_switches for c in combos)/len(combos):.1f} | "
        f"ESS_norm={sum(diag['ess'])/n_dec:.2f} w_max={sum(diag['w_max'])/n_dec:.2f} "
        f"Δlen_choisi={sum(diag['dlen'])/n_dec:+.0f}tok | "
        f"fins: orig={ends['original']} source_end={ends['source_end']} budget={ends['budget']} | "
        f"scoring: {diag['n_jobs']} jobs ~{diag['tok_scored']/1e6:.1f}Mtok en {t_snis:.0f}s "
        f"(rollout {t_rollout:.0f}s)",
        flush=True,
    )
    print(f"[snis] rewards originaux={orig_rewards} -> combos={[round(r, 2) for r in rewards_out]}", flush=True)

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        "env_mask": env_mask_out,
        "episode_reward": rewards_out,
        "invalid_steps": invalid_out,
        "mean_n_actions_per_turn": mean_actions_out,
    }


# ---------------------------------------------------------------------------
# 5) Selftest CPU : valide la logique de recombinaison sans GPU ni serveur env
# ---------------------------------------------------------------------------

def run_selftest() -> None:
    AP = [901, 902]  # faux "\n<|im_start|>assistant\n"

    def mk_episode(src: int, n_turns: int, reward: float) -> Episode:
        turns = []
        for k in range(n_turns):
            ids = [src * 100 + k * 10 + t for t in range(3 + (src + k) % 3)]
            turns.append(Turn(ids=ids, logprobs=[-0.5] * len(ids),
                              obs_ids=[8000 + src * 10 + k], action_count=1, invalid=(k == 0 and src == 2)))
        return Episode(prompt_ids=[1, 2, 3], turns=turns, reward=reward)

    eps = [mk_episode(0, 1, 1.0), mk_episode(1, 2, 0.0), mk_episode(2, 3, 0.0), mk_episode(3, 3, 1.0)]

    def fake_score(jobs):
        # logp par token déterministe, dépendant du candidat (pour le test argmax).
        return [[-0.001 * (sum(cand) % 97) - 0.1] * len(cand) for _, cand in jobs]

    def expected_original_stream(ep: Episode):
        ids, mask = [], []
        for k, t in enumerate(ep.turns, start=1):
            if k > 1:
                ids += AP; mask += [0] * len(AP)
            ids += t.ids; mask += [1] * len(t.ids)
            ids += t.obs_ids; mask += [0] * len(t.obs_ids)
        return ids, mask

    # Test 1 — keep_originals=G : les combos reproduisent exactement les originaux.
    combos, _ = snis_recombine([eps], fake_score, random.Random(0), alpha=1.0,
                               keep_originals=4, max_ctx_tokens=10_000, assistant_prefix_ids=AP)
    for m, (c, e) in enumerate(zip(combos, eps)):
        exp_ids, exp_mask = expected_original_stream(e)
        assert c.stream_ids == exp_ids and c.stream_mask == exp_mask, f"original {m} non reproduit"
        assert c.reward == e.reward and c.end_reason == "original"
        assert len(c.stream_ids) == len(c.stream_lps) == len(c.stream_mask)
    print("PASS 1 — keep_originals=G reproduit les trajectoires originales à l'identique")

    # Test 2 — α=0 : poids uniformes sur les candidats à chaque décision.
    _, diag = snis_recombine([eps], fake_score, random.Random(1), alpha=0.0,
                             keep_originals=0, max_ctx_tokens=10_000,
                             assistant_prefix_ids=AP, record_weights=True)
    for w in diag["weights"]:
        assert all(abs(x - 1.0 / len(w)) < 1e-12 for x in w), f"poids non uniformes: {w}"
    assert all(abs(e - 1.0) < 1e-9 for e in diag["ess"]), "ESS != 1 avec α=0"
    print(f"PASS 2 — α=0 -> poids uniformes ({len(diag['weights'])} décisions vérifiées, ESS=1)")

    # Test 3 — α très grand : choix déterministe du candidat argmax + invariants.
    combos_a, diag_a = snis_recombine([eps], fake_score, random.Random(2), alpha=1e6,
                                      keep_originals=0, max_ctx_tokens=10_000,
                                      assistant_prefix_ids=AP, record_weights=True)
    combos_b, _ = snis_recombine([eps], fake_score, random.Random(999), alpha=1e6,
                                 keep_originals=0, max_ctx_tokens=10_000, assistant_prefix_ids=AP)
    assert [c.chosen for c in combos_a] == [c.chosen for c in combos_b], "argmax non déterministe"
    assert all(max(w) > 0.999 for w in diag_a["weights"]), "α grand -> poids non concentrés"
    print("PASS 3 — α→∞ : sélection argmax déterministe (indépendante de la graine)")

    # Test 4 — invariants structurels + héritage du reward et de la terminaison.
    for c in combos_a:
        assert c.finished and c.chosen[0][1] == 1
        ks = [k for _, k in c.chosen]
        assert ks == list(range(1, len(ks) + 1)), f"tours non consécutifs: {ks}"
        j_last, k_last = c.chosen[-1]
        assert k_last == len(eps[j_last].turns), "fin sans consommer le dernier tour source"
        assert c.reward == eps[j_last].reward, "reward non hérité de la source du dernier tour"
        assert len(c.stream_ids) == len(c.stream_lps) == len(c.stream_mask)
        assert set(c.stream_mask) <= {0, 1}
    print("PASS 4 — invariants: tours consécutifs, terminaison source_end, reward hérité, masques cohérents")

    # Test 5 — budget contexte minuscule : les combos se coupent en 'budget'.
    combos_c, _ = snis_recombine([eps], fake_score, random.Random(3), alpha=0.0,
                                 keep_originals=0, max_ctx_tokens=10, assistant_prefix_ids=AP)
    long_combos = [c for c in combos_c if len(c.chosen) >= 1 and c.end_reason == "budget"]
    assert long_combos, "aucun combo coupé par le budget alors que max_ctx_tokens=10"
    for c in long_combos:
        j_last = c.chosen[-1][0]
        assert c.reward == eps[j_last].reward
    print(f"PASS 5 — budget contexte respecté ({len(long_combos)}/4 combos coupés, reward hérité)")

    # Test 6 — reproductibilité : même graine => mêmes choix.
    r1, _ = snis_recombine([eps], fake_score, random.Random(42), 1.0, 0, 10_000, AP)
    r2, _ = snis_recombine([eps], fake_score, random.Random(42), 1.0, 0, 10_000, AP)
    assert [c.chosen for c in r1] == [c.chosen for c in r2]
    print("PASS 6 — reproductibilité à graine fixée")

    # Test 7 — découplage M/G : M=8 combos depuis G=4 épisodes réels.
    combos_m, _ = snis_recombine([eps], fake_score, random.Random(5), alpha=1.0,
                                 keep_originals=2, max_ctx_tokens=10_000,
                                 assistant_prefix_ids=AP, combos_per_group=8)
    assert len(combos_m) == 8, f"attendu 8 combos, obtenu {len(combos_m)}"
    for m, c in enumerate(combos_m):
        assert c.chosen[0] == (m % 4, 1), f"combo {m}: s1={c.chosen[0]} != {(m % 4, 1)}"
        assert c.finished and len(c.stream_ids) == len(c.stream_lps) == len(c.stream_mask)
        j_last, k_last = c.chosen[-1]
        assert c.reward == eps[j_last].reward
    assert combos_m[0].end_reason == combos_m[1].end_reason == "original"
    s1_counts = [sum(1 for c in combos_m if c.chosen[0][0] == j) for j in range(4)]
    assert s1_counts == [2, 2, 2, 2], f"s1 non équilibré: {s1_counts}"
    print("PASS 7 — M/G découplés : 8 combos depuis 4 épisodes, s1 cyclé 2x par source, "
          "originaux préservés, invariants OK")

    print("\nSelftest SNIS : 7/7 OK")


# ---------------------------------------------------------------------------
# 6) main : parse les args SNIS, substitue la rollout_func, délègue à tg.main()
# ---------------------------------------------------------------------------

def main() -> None:
    global SNIS_ALPHA, SNIS_REAL_ROLLOUTS, SNIS_KEEP_ORIGINALS, SNIS_SCORE_BATCH, \
        SNIS_MAX_CTX_TOKENS, SNIS_SEED

    parser = argparse.ArgumentParser(add_help=False)  # --help passe à tg.main()
    parser.add_argument("--snis-alpha", type=float, default=1.0)
    parser.add_argument("--snis-real-rollouts", type=int, default=0)
    parser.add_argument("--snis-keep-originals", type=int, default=0)
    parser.add_argument("--snis-score-batch-size", type=int, default=4)
    parser.add_argument("--snis-max-ctx-tokens", type=int, default=15000)
    parser.add_argument("--snis-seed", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    snis_args, rest = parser.parse_known_args()

    if snis_args.selftest:
        run_selftest()
        return

    SNIS_ALPHA = snis_args.snis_alpha
    SNIS_REAL_ROLLOUTS = snis_args.snis_real_rollouts
    SNIS_KEEP_ORIGINALS = snis_args.snis_keep_originals
    SNIS_SCORE_BATCH = snis_args.snis_score_batch_size
    SNIS_MAX_CTX_TOKENS = snis_args.snis_max_ctx_tokens
    SNIS_SEED = snis_args.snis_seed
    print(f"[snis] α={SNIS_ALPHA} real_rollouts_G={SNIS_REAL_ROLLOUTS or 'M (=num-generations)'} "
          f"keep_originals={SNIS_KEEP_ORIGINALS} "
          f"score_batch={SNIS_SCORE_BATCH} max_ctx={SNIS_MAX_CTX_TOKENS} seed={SNIS_SEED}", flush=True)

    # Substitution : tg.main() référence textcraft_rollout_func par son global de
    # module au moment de construire le GRPOTrainer -> on le remplace AVANT l'appel.
    # Tout le reste (args CLI, vLLM in-process, callbacks, éval, saves) est hérité
    # de train_grpo.py à l'identique — garantit la comparabilité des runs.
    tg.textcraft_rollout_func = textcraft_snis_rollout_func
    sys.argv = [sys.argv[0]] + rest
    tg.main()


if __name__ == "__main__":
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
