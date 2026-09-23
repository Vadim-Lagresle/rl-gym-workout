"""Which training tasks are drawn, and how often: the depth curriculum and depth rebalancing.

In plain words: by default TRL draws the training tasks uniformly. This file
replaces that draw by a weighted one, where the weights can change during training.
Three uses: the Depth curriculum (only recipes up to a given depth, raised on a
calendar or when the training reward reaches a threshold), and the fixed
rebalancing of experiment F (each depth gets a quarter of the draws, or a mass
proportional to the square root of its size). install_weighted_sampler() plugs the
chosen weights into a GRPOTrainer; the rest of the training loop is unchanged.

Notes (FR) : WeightedRepeatSampler reproduit la structure du RepeatSampler de TRL
(chunks répétés G fois) ; les poids sont relus à chaque chunk, donc un changement de
palier prend effet au chunk suivant. Issu du port MAGELLAN (archivé dans archive/train/).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import numpy as np

from src.train.data import REPO_ROOT, TRAIN_PATH, depth_file_for


class WeightedRepeatSampler:
    """Structure IDENTIQUE au RepeatSampler de TRL (trl/trainer/utils.py:697) —
    chunks de `batch_size` indices uniques, chacun répété mini_repeat_count fois,
    le bloc répété repeat_count fois — mais chaque chunk est TIRÉ ∝ prob_fn() au
    moment où il est produit (itération paresseuse : les poids mis à jour en cours
    d'epoch prennent effet au chunk suivant, ≡ leur sample() par épisode).
    Tirage avec remise, comme leur échantillonnage i.i.d. par épisode."""

    def __init__(self, data_source: Any, mini_repeat_count: int, batch_size: int,
                 repeat_count: int, prob_fn: Callable[[], np.ndarray],
                 seed: int | None = None):
        self.n = len(data_source)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.prob_fn = prob_fn
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n // self.batch_size):
            p = np.asarray(self.prob_fn(), dtype=np.float64)
            p = p / p.sum()
            chunk = self.rng.choice(self.n, size=self.batch_size, replace=True, p=p)
            for _ in range(self.repeat_count):
                for idx in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield int(idx)

    def __len__(self) -> int:
        # même formule que RepeatSampler (chunks entiers uniquement)
        return (self.n // self.batch_size) * self.batch_size \
            * self.mini_repeat_count * self.repeat_count


def depth_balance_probs(depths: list[int], mode: str) -> np.ndarray:
    """Probabilités fixes par item pour un rééquilibrage par profondeur (exp48).
    uniform : chaque profondeur reçoit la même masse, répartie uniformément sur ses items ;
    sqrt    : masse d'une profondeur ∝ √n_d (compromis entre l'uniforme par item et par profondeur)."""
    d = np.asarray(depths)
    levels = sorted(set(d.tolist()))
    n = {k: int((d == k).sum()) for k in levels}
    mass = {k: (1.0 if mode == "uniform" else np.sqrt(n[k])) for k in levels}
    tot = sum(mass.values())
    p = np.array([mass[k] / tot / n[k] for k in d.tolist()], dtype=np.float64)
    return p / p.sum()


class DepthScheduleProvider:
    """Curriculum depth par paliers (exp33) : prob_fn = uniforme sur les items de
    depth <= palier(epoch). Spec '1:0,2:6,3:20,4:45' = depth max 1 dès l'epoch 0,
    2 dès l'epoch 6, etc. Lit l'epoch en direct sur trainer.state (le sampler
    paresseux prend le palier en compte au chunk suivant)."""

    def __init__(self, spec: str, depths: list[int]):
        self.stages = sorted(
            (float(e), int(d)) for d, e in
            (tok.split(":") for tok in spec.split(","))
        )
        if self.stages[0][0] != 0.0:
            raise ValueError(f"--depth-schedule-epochs doit commencer à l'epoch 0 : {spec!r}")
        self.depths = np.asarray(depths)
        self.trainer_ref: Any = None
        self._last_stage = -1

    def current_max_depth(self) -> int:
        epoch = 0.0
        if self.trainer_ref is not None and getattr(self.trainer_ref, "state", None) \
                and self.trainer_ref.state.epoch is not None:
            epoch = float(self.trainer_ref.state.epoch)
        d = self.stages[0][1]
        for e, dd in self.stages:
            if epoch >= e:
                d = dd
        return d

    def probabilities(self) -> np.ndarray:
        d = self.current_max_depth()
        if d != self._last_stage:
            n_ok = int(np.sum(self.depths <= d))
            print(f"[depth-schedule] palier depth<={d} : {n_ok}/{len(self.depths)} items",
                  flush=True)
            self._last_stage = d
        mask = (self.depths <= d).astype(np.float64)
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask / mask.sum()


class DepthAutoScheduleProvider:
    """Curriculum depth AUTO-DÉCLENCHÉ (exp33.1, règle Vadim 28/08) : le palier
    avance quand la MOYENNE du reward train sur la dernière epoch complète
    atteint `threshold` (défaut 0.8), ou à défaut après `max_stage_epochs` au
    palier courant (défaut 10). probabilities() = uniforme sur les items de
    depth <= palier (masque identique à DepthScheduleProvider). Le reward et
    l'epoch arrivent via on_log (duck-typing TrainerCallback, même pattern que
    MagellanCallback) ; la fenêtre est vidée à chaque passage de palier, donc
    le check suivant attend une epoch complète de mesures fraîches."""

    def __init__(self, depths: list[int], steps_per_epoch: int,
                 threshold: float = 0.8, max_stage_epochs: float = 10.0,
                 start_depth: int = 1):
        self.depths = np.asarray(depths)
        real = self.depths[self.depths < 99]
        self.max_depth = int(real.max()) if len(real) else start_depth
        self.stage = int(start_depth)
        self.threshold = float(threshold)
        self.max_stage_epochs = float(max_stage_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.window: deque = deque(maxlen=self.steps_per_epoch)
        self.stage_start_epoch = 0.0
        self._announced = -1

    def on_log(self, targs: Any, state: Any, control: Any,
               logs: dict | None = None, **kwargs: Any) -> None:
        if not logs or "reward" not in logs:
            return
        epoch = float(logs.get("epoch",
                               getattr(state, "epoch", 0.0) or 0.0) or 0.0)
        self.window.append(float(logs["reward"]))
        if self.stage >= self.max_depth:
            return
        full = len(self.window) == self.steps_per_epoch
        mean_r = sum(self.window) / len(self.window) if self.window else 0.0
        hit = full and mean_r >= self.threshold
        timeout = (epoch - self.stage_start_epoch) >= self.max_stage_epochs
        if hit or timeout:
            reason = (f"reward moyen 1 epoch {mean_r:.3f} >= {self.threshold}"
                      if hit else f"cap {self.max_stage_epochs:g} epochs atteint")
            self.stage += 1
            self.stage_start_epoch = epoch
            self.window.clear()
            n_ok = int(np.sum(self.depths <= self.stage))
            print(f"[depth-auto] >>> PALIER depth<={self.stage} "
                  f"(epoch {epoch:.2f}, {reason}) : {n_ok}/{len(self.depths)} items",
                  flush=True)

    def probabilities(self) -> np.ndarray:
        d = self.stage
        if d != self._announced:
            n_ok = int(np.sum(self.depths <= d))
            print(f"[depth-auto] palier depth<={d} : {n_ok}/{len(self.depths)} items",
                  flush=True)
            self._announced = d
        mask = (self.depths <= d).astype(np.float64)
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask / mask.sum()

    # HF appelle toutes les méthodes on_* : no-op par défaut
    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def install_weighted_sampler(trainer: Any, args: argparse.Namespace, rows: list[dict]) -> None:
    """Replaces the trainer's uniform task sampler when a depth option is set; no-op otherwise."""
    modes = [bool(args.depth_schedule_epochs), bool(args.depth_schedule_auto), args.depth_balance != "none"]
    if not any(modes):
        return
    if sum(modes) > 1:
        raise SystemExit("--depth-schedule-epochs, --depth-schedule-auto et --depth-balance "
                         "sont mutuellement exclusifs.")
    if args.max_depth or args.depth_exact:
        raise SystemExit("Le curriculum par échantillonnage travaille sur le dataset "
                         "ENTIER : incompatible avec --max-depth/--depth-exact.")
    if getattr(trainer.args, "dataloader_num_workers", 0):
        raise SystemExit("Le sampler pondéré lit les poids en direct : "
                         "dataloader_num_workers doit rester 0.")

    train_path = Path(args.train_file) if args.train_file else TRAIN_PATH
    if not train_path.is_absolute():
        train_path = REPO_ROOT / train_path
    with depth_file_for(train_path).open() as f:
        depth_map = json.load(f)
    depths = [int(depth_map.get(r["item_id"], 99)) for r in rows]

    if args.depth_balance != "none":
        probs_fixed = depth_balance_probs(depths, args.depth_balance)
        prob_fn: Callable[[], np.ndarray] = lambda: probs_fixed  # noqa: E731
        d = np.asarray(depths)
        print(f"[depth-balance] échantillonnage '{args.depth_balance}' : masse par profondeur "
              + ", ".join(f"d{k}={probs_fixed[d == k].sum():.3f} ({int((d == k).sum())} items)"
                          for k in sorted(set(depths))), flush=True)
    elif args.depth_schedule_auto:
        # exp33.1 — palier déclenché au reward train (fenêtre = 1 epoch de steps).
        steps_per_epoch = max(1, len(rows) * args.num_generations // args.gradient_accumulation_steps)
        provider = DepthAutoScheduleProvider(
            depths, steps_per_epoch=steps_per_epoch, threshold=args.depth_auto_threshold,
            max_stage_epochs=args.depth_auto_max_epochs, start_depth=args.depth_auto_start_stage)
        trainer.add_callback(provider)   # reçoit reward/epoch via on_log
        prob_fn = provider.probabilities
        print(f"[depth-auto] curriculum depth déclenché au succès : seuil "
              f"{args.depth_auto_threshold} sur {steps_per_epoch} steps (1 epoch), "
              f"cap {args.depth_auto_max_epochs:g} epochs/palier"
              + (f", REPRISE au palier {args.depth_auto_start_stage}"
                 if args.depth_auto_start_stage > 1 else ""), flush=True)
    else:
        provider = DepthScheduleProvider(args.depth_schedule_epochs, depths)
        provider.trainer_ref = trainer
        prob_fn = provider.probabilities
        print(f"[depth-schedule] curriculum depth par paliers : "
              f"{args.depth_schedule_epochs} (échantillonnage pondéré, dataset entier)", flush=True)

    def _weighted_sampler(self, dataset=None):
        ds = dataset if dataset is not None else self.train_dataset
        return WeightedRepeatSampler(
            data_source=ds, mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            prob_fn=prob_fn, seed=self.args.seed)

    trainer._get_train_sampler = MethodType(_weighted_sampler, trainer)
