"""Boucle d'environnement TextCraft unique + contrat rollout_func de TRL.

C'est LE seul endroit du projet où l'on joue des épisodes d'entraînement :
`collect_episodes` génère (vLLM in-process ou HF) → env.step → observe → repeat,
et retourne une structure PAR TOUR (`Episode` / `Turn`). Les deux consommateurs :

  - GRPO pur   : `grpo_rollout_func` = collect_episodes + `episodes_to_trl_batch`
                 (mise à plat du flux entrelacé actions/observations/marqueurs) ;
  - SNIS       : `snis.make_rollout_func` recombine les tours entre épisodes
                 avant la même mise à plat (via `extend_stream`).

Conventions de masquage (contrat TRL "env_mask", cf. verl RolloutHandler,
external/AgentGym-RL/verl/workers/rollout/schemas.py, format_config["qwen"]) :
completion_ids contient le flux COMPLET entrelacé — actions, observations ET
marqueurs de chat template (<|im_start|>user, <|im_end|>, generation prompts) —
pour que le backward voie un contexte token-identique à la génération. env_mask :
  1 = token d'action (sortie modèle, gradient actif)
  0 = token d'observation / template (gradient masqué)
Sans les marqueurs interposés, les logprobs de chaque token d'action à partir du
tour 2 seraient calculées dans un contexte malformé.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agentenv.envs import TextCraftEnvClient

from src.train import vllm_engine
from src.train.data import ENV_SERVER_URL, ITEM_TAG_RE
from src.train.schedules import current_max_rounds, log_schedule_state


def count_actions(text: str) -> int:
    return len(re.findall(r"Action:\s*(.*?)(?=\n|$)", text, flags=re.DOTALL))


# ---------------------------------------------------------------------------
# Structure par tour
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
    reward: float             # max des rewards env observés (sparse 0/1 en pratique)


# ---------------------------------------------------------------------------
# LA boucle d'environnement (une seule implémentation)
# ---------------------------------------------------------------------------

def collect_episodes(prompts: list[list[dict[str, str]]], trainer: Any) -> list[Episode]:
    """Joue un épisode complet par prompt et retourne la structure par tour.

    Trois phases par round, sur les épisodes encore actifs :
      1. génération batchée (vLLM in-process si initialisé, sinon HF) ;
      2. env.step en parallèle (ThreadPool : IO-bound HTTP, wall ≈ max(latence)) ;
      3. mise à jour des états conversationnels + comptage invalid/reward.
    """
    tokenizer = trainer.processing_class
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    n = len(prompts)
    states = [list(p) for p in prompts]
    done = [False] * n
    episodes = [Episode(prompt_ids=[], turns=[], reward=0.0) for _ in range(n)]
    env_clients: list[TextCraftEnvClient] = []

    # Bootstrap : un client env par épisode, l'idx est encodé dans le marqueur
    # <ITEM_IDX:n> du dernier message user. Le marqueur (et lui seul) est remplacé
    # par l'observation initiale — le reste du message (ex. observation finale du
    # dernier exemple few-shot, exp19) est préservé.
    for i in range(n):
        content = str(states[i][-1]["content"])
        m = ITEM_TAG_RE.search(content)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}] last user message: {content!r}")
        env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(int(m.group(1)))
        states[i][-1]["content"] = content.replace(m.group(0), env.observe()).strip()
        env_clients.append(env)

    cap = current_max_rounds(trainer)
    log_schedule_state(trainer, cap)

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

            if vllm_engine.LLM_ENGINE is not None:
                # vLLM in-process : tous les épisodes actifs en un appel (prefix caching).
                round_completion_ids, round_logprobs = vllm_engine.generate_round(
                    tokenizer, states, active, max_tokens=trainer.args.max_completion_length
                )
            else:
                # HF _generate_single_turn : ré-encode le contexte complet à chaque round.
                round_completion_ids, round_logprobs = trainer._generate_single_turn(
                    round_prompt_ids, images=None, multimodal_fields={}
                )

            # Phase 1 — décodage des complétions (CPU, séquentiel).
            actions: dict[int, str] = {}
            for j, i in enumerate(active):
                ids = list(round_completion_ids[j])
                lps = list(round_logprobs[j][: len(ids)]) if round_logprobs is not None else [0.0] * len(ids)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                states[i].append({"role": "assistant", "content": text})
                # Message assistant COMPLET : le client env extrait l'action lui-même et
                # renvoie une obs d'erreur sur les réponses multi-actions (comportement
                # papier — verl envoie le contenu décodé brut, cf. vllm_rollout.py agent_step).
                actions[i] = text
                if not ids or ids[-1] != tokenizer.eos_token_id:
                    # Génération tronquée par max_tokens : on ferme quand même le tour,
                    # comme le suffixe add_assistant_message de verl (loss_mask=1 sur <|im_end|>).
                    ids = ids + im_end_ids
                    lps = lps + [0.0] * len(im_end_ids)
                episodes[i].turns.append(Turn(ids=ids, logprobs=lps, obs_ids=[],
                                              action_count=count_actions(text), invalid=False))

            # Phase 2 — appels HTTP parallèles au serveur TextCraft (IO-bound).
            futures = [(executor.submit(env_clients[i].step, actions[i]), i) for i in active]
            step_results = {i: fut.result() for fut, i in futures}

            # Phase 3 — mise à jour des états avec les réponses env (CPU, séquentiel).
            for i in active:
                out = step_results[i]
                obs = out.state
                states[i].append({"role": "user", "content": obs})
                episodes[i].reward = max(episodes[i].reward, float(out.reward))
                low = obs.lower()
                turn = episodes[i].turns[-1]
                turn.invalid = ("could not" in low or "error:" in low or "wrong item format" in low)
                # Observation AVEC ses marqueurs de template : le backward voit le flux
                # token-identique à la génération suivante (user_prefix/suffix de verl).
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
# Mise à plat tours → flux TRL (partagée avec snis._append_turn)
# ---------------------------------------------------------------------------

def extend_stream(stream_ids: list[int], stream_lps: list[float], stream_mask: list[int],
                  turn: Turn, k: int, lps: list[float],
                  assistant_prefix_ids: list[int]) -> None:
    """Ajoute un tour (tokens assistant + bloc obs) à un flux TRL, avec les
    conventions de masque du module : generation prompt et obs masqués
    (env_mask=0), tokens d'action actifs (env_mask=1). k est 1-based : à partir
    du tour 2, les tokens du generation prompt (\\n<|im_start|>assistant\\n) —
    présents dans le contexte de génération mais absents de completion_ids —
    sont réinjectés, masqués. (Le generation prompt du round 0 termine déjà
    prompt_ids.)"""
    if k > 1:
        stream_ids.extend(assistant_prefix_ids)
        stream_lps.extend([0.0] * len(assistant_prefix_ids))
        stream_mask.extend([0] * len(assistant_prefix_ids))
    stream_ids.extend(turn.ids)
    stream_lps.extend(lps[: len(turn.ids)] + [0.0] * max(0, len(turn.ids) - len(lps)))
    stream_mask.extend([1] * len(turn.ids))
    stream_ids.extend(turn.obs_ids)
    stream_lps.extend([0.0] * len(turn.obs_ids))
    stream_mask.extend([0] * len(turn.obs_ids))


def episodes_to_trl_batch(episodes: list[Episode], tokenizer: Any) -> dict[str, Any]:
    """Aplati les épisodes au contrat rollout_func de TRL (+ stats pour le reward)."""
    assistant_prefix_ids = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)

    prompt_ids_out, completion_ids_out, logprobs_out, env_mask_out = [], [], [], []
    rewards_out, invalid_out, mean_actions_out, n_turns_out = [], [], [], []
    for ep in episodes:
        ids: list[int] = []
        lps: list[float] = []
        mask: list[int] = []
        for k, turn in enumerate(ep.turns, start=1):
            extend_stream(ids, lps, mask, turn, k, turn.logprobs, assistant_prefix_ids)
        # Garantir des tableaux non vides (contrat TRL).
        if not ids:
            ids, lps, mask = [tokenizer.eos_token_id], [0.0], [1]
        prompt_ids_out.append(ep.prompt_ids or [tokenizer.eos_token_id])
        completion_ids_out.append(ids)
        logprobs_out.append(lps)
        env_mask_out.append(mask)
        rewards_out.append(ep.reward)
        invalid_out.append(sum(int(t.invalid) for t in ep.turns))
        # Moyenne par épisode du nb d'actions par message (1.0 = idéal) ; neutre (1.0)
        # si 0 tour actif (extrêmement rare).
        mean_actions_out.append(
            sum(t.action_count for t in ep.turns) / len(ep.turns) if ep.turns else 1.0
        )
        n_turns_out.append(len(ep.turns))

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        # env_mask : 1=token d'action (gradient actif), 0=obs/template (masqué).
        "env_mask": env_mask_out,
        # Champs supplémentaires transmis à la reward function via reward_kwargs.
        "episode_reward": rewards_out,
        "invalid_steps": invalid_out,
        "mean_n_actions_per_turn": mean_actions_out,
        "_n_turns": n_turns_out,  # consommé par le print de grpo_rollout_func uniquement
    }


def grpo_rollout_func(prompts: list[list[dict[str, str]]], trainer: Any) -> dict[str, Any]:
    """rollout_func GRPO pur : collecte interactive + mise à plat TRL."""
    episodes = collect_episodes(prompts, trainer)
    batch = episodes_to_trl_batch(episodes, trainer.processing_class)
    n_turns = batch.pop("_n_turns")
    print(
        f"[rollout] n={len(episodes)} mean_n_actions_per_turn="
        f"{[round(x, 2) for x in batch['mean_n_actions_per_turn']]} "
        f"n_active_turns={n_turns} "
        f"episode_reward={batch['episode_reward']} invalid_steps={batch['invalid_steps']}",
        flush=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def textcraft_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    mean_n_actions_per_turn: list[float],
    **kwargs: Any,
) -> list[float]:
    """Sparse outcome reward only (matches AgentGym-RL TextCraft training recipe).

    Removes the v2/v3 shaping that empirically degraded Pass@1 (18 -> 14 -> 8 over
    baseline -> v2 -> v3). AgentGym-RL uses only the env 0/1 scalar at episode end,
    KL handled in the loss via use_kl_loss. See worker investigation 2026-05-12.

    invalid_steps and mean_n_actions_per_turn are kept in the signature only so
    rollout_func's return dict matches; they are NOT used to reshape the reward.
    """
    rewards = [float(r) for r in episode_reward]
    debug = " | ".join(
        f"r={r:+.3f} mean_n={mn:.2f} bad={bs}"
        for r, mn, bs in zip(rewards, mean_n_actions_per_turn, invalid_steps)
    )
    print(f"[reward] sparse outcome | {debug}", flush=True)
    return rewards
