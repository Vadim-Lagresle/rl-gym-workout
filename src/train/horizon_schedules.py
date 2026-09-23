"""Interaction budgets that grow during training: the horizon and budget curricula.

In plain words: at the start of training the agent may be allowed only a few turns
per episode (10), or only a few tokens per turn (256); these caps are raised at
chosen epochs. This file parses the schedules given on the command line and answers
"what is the cap right now?" for the rollout loop (rollout.py). The Horizon
curriculum of the report (10/20/30 turns at epochs 0/15/30) and the Budget
curriculum (256/512/1024 tokens) are both defined here.

Notes (FR) : le schedule est écrit en CLI comme des paires "<cap>:<seuil>" ; seuil =
global step (--max-rounds-schedule) ou epoch (--max-rounds-schedule-epochs,
--max-completion-schedule-epochs). train_grpo.main() assigne les listes parsées aux
variables de module ci-dessous ; rollout.py les lit à chaque tour.
"""

from __future__ import annotations

from typing import Any

# Paper appendix B.3 : 20 tours pour le GRPO pur TextCraft (le 30 de
# textcraft_train.sh est le stage FINAL de ScalingInter, pas le run GRPO ;
# Figure 7 : un budget d'interaction trop grand déstabilise l'entraînement).
MAX_SIM_ROUNDS = 20  # cap training — l'éval utilise 30 (periodic_eval.EVAL_MAX_ROUNDS)

# Fixés par train_grpo.main() depuis la CLI (assignation d'attribut de module).
MAX_ROUNDS_SCHEDULE: list[tuple[int, int]] | None = None
MAX_ROUNDS_SCHEDULE_EPOCHS: list[tuple[float, int]] | None = None
# Curriculum du budget de sortie PAR TOUR (exp35, ajout 2026-08-26) : même format
# et même mécanique que ScalingInter, mais pilote max_tokens de la génération
# (rollout.py) au lieu du nombre de tours. Epoch-based uniquement.
MAX_COMPLETION_SCHEDULE_EPOCHS: list[tuple[float, int]] | None = None
_LAST_COMPLETION_CAP: int | None = None  # trace [completion-schedule] au changement


def _parse_rounds_schedule(spec: str, thr_cast):
    """Parse '6:0,11:2,...' into [(threshold, max_rounds), ...] sorted by threshold.

    The CLI order is "<max_rounds>:<threshold>" (max_rounds FIRST, threshold
    SECOND). thr_cast is `int` for step thresholds, `float` for epoch thresholds."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for p in parts:
        rounds_str, thr_str = p.split(":")
        out.append((thr_cast(thr_str), int(rounds_str)))
    out.sort(key=lambda x: x[0])
    return out


def parse_max_rounds_schedule(spec: str) -> list[tuple[int, int]]:
    """Step-based ScalingInter schedule (thresholds = global steps)."""
    return _parse_rounds_schedule(spec, int)


def parse_max_rounds_schedule_epochs(spec: str) -> list[tuple[float, int]]:
    """Epoch-based ScalingInter schedule (thresholds = fractional epochs)."""
    return _parse_rounds_schedule(spec, float)


def current_max_rounds(trainer: Any) -> int:
    """Resolve the active max_rounds cap from the active ScalingInter schedule.

    Epoch-based schedule takes priority over step-based; if neither is set,
    fall back to the fixed MAX_SIM_ROUNDS cap."""
    state = getattr(trainer, "state", None)
    if MAX_ROUNDS_SCHEDULE_EPOCHS is not None:
        epoch = float(getattr(state, "epoch", 0.0) or 0.0) if state is not None else 0.0
        cap = MAX_SIM_ROUNDS
        for thr_epoch, thr_rounds in MAX_ROUNDS_SCHEDULE_EPOCHS:
            if epoch >= thr_epoch:
                cap = thr_rounds
        return cap
    if MAX_ROUNDS_SCHEDULE is not None:
        step = int(getattr(state, "global_step", 0) or 0) if state is not None else 0
        cap = MAX_SIM_ROUNDS
        for thr_step, thr_rounds in MAX_ROUNDS_SCHEDULE:
            if step >= thr_step:
                cap = thr_rounds
        return cap
    return MAX_SIM_ROUNDS


def log_schedule_state(trainer: Any, cap: int) -> None:
    """Trace [scaling-inter] au début de chaque rollout quand un schedule est actif."""
    if MAX_ROUNDS_SCHEDULE_EPOCHS is not None:
        state = getattr(trainer, "state", None)
        ep = float(getattr(state, "epoch", 0.0) or 0.0)
        step = int(getattr(state, "global_step", 0) or 0)
        print(f"[scaling-inter] step={step} epoch={ep:.2f} max_rounds={cap}", flush=True)
    elif MAX_ROUNDS_SCHEDULE is not None:
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        print(f"[scaling-inter] step={step} max_rounds={cap}", flush=True)


def parse_max_completion_schedule_epochs(spec: str) -> list[tuple[float, int]]:
    """Schedule epoch-based du budget de sortie par tour, format '<max_tokens>:<epoch>'
    (même convention valeur-d'abord que ScalingInter), e.g. '256:0,512:15,1024:35'."""
    return _parse_rounds_schedule(spec, float)


def current_max_completion(trainer: Any, default: int) -> int:
    """Budget max_tokens actif pour la génération d'UN tour (consommé par
    rollout.py à chaque round). Sans schedule : `default` (= la valeur CLI
    --max-completion-length, comportement historique inchangé). Avec schedule :
    palier de l'epoch courante ; trace [completion-schedule] à chaque changement."""
    global _LAST_COMPLETION_CAP
    if MAX_COMPLETION_SCHEDULE_EPOCHS is None:
        return default
    state = getattr(trainer, "state", None)
    epoch = float(getattr(state, "epoch", 0.0) or 0.0) if state is not None else 0.0
    cap = default
    for thr_epoch, thr_tokens in MAX_COMPLETION_SCHEDULE_EPOCHS:
        if epoch >= thr_epoch:
            cap = thr_tokens
    if cap != _LAST_COMPLETION_CAP:
        step = int(getattr(state, "global_step", 0) or 0) if state is not None else 0
        print(f"[completion-schedule] step={step} epoch={epoch:.2f} max_tokens={cap}",
              flush=True)
        _LAST_COMPLETION_CAP = cap
    return cap
