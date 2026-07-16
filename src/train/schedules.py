"""Curriculum ScalingInter : budget d'interaction (max_rounds) progressif.

Deux variantes, écrites en CLI comme des paires "<max_rounds>:<threshold>" :
  - pilotée par STEP   (--max-rounds-schedule)        : threshold = global step
  - pilotée par EPOCH  (--max-rounds-schedule-epochs) : threshold = epoch fractionnaire
En interne : liste de (threshold, max_rounds) triée par threshold. None = cap
fixe MAX_SIM_ROUNDS. Une seule variante active à la fois (main() valide).
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
