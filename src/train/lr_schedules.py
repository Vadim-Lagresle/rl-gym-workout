"""Learning-rate and KL-coefficient schedules, and saving/restoring the best model.

In plain words: these callbacks lower the learning rate (and the KL coefficient
beta by the same factor) during a run, on a fixed calendar, and can restart each
stage from the best model of the previous one. They were the first attempt at
stabilising LoRA training (runs exp20-exp23); the final recipe no longer uses them
because the moving KL anchor (kl_anchor.py) made them unnecessary, but they are
kept so those runs stay reproducible. The adaptive variant, driven by the training
reward, lives in lr_adaptive.py.

Contents: apply_lr_beta (change LR and beta of a live trainer), save_best_adapter /
restore_best_adapter (atomic save of the adapter + optimizer), StagedLrBetaCallback
(divide LR and beta every N epochs), StagedBestRestoreCallback (long stages, each
restarting from the best of the previous stage).
"""

from __future__ import annotations

import statistics  # noqa: F401  (utilisé par les callbacks importés d'ici)
from typing import Any

from transformers import TrainerCallback


def apply_lr_beta(trainer: Any, lr: float, beta: float, optimizer: Any = None) -> None:
    """Applique un nouveau couple (LR, beta) à un GRPOTrainer en cours de run.

    Mécanique (vérifiée sur TRL 1.9.0 / transformers 5.14) :
      - LR : le run utilise lr_scheduler_type="constant" (LambdaLR, lambda≡1) →
        le LR effectif est base_lrs[i] × 1. On modifie donc `base_lrs` du scheduler
        (déballé de l'éventuel wrapper accelerate) ET les param_groups de
        l'optimizer, pour que le changement survive au scheduler.step() suivant.
      - beta : GRPOTrainer relit `self.beta` à CHAQUE calcul de loss (la capture
        à l'init ne concerne que le chemin Liger, inutilisé ici) → assigner
        `trainer.beta` suffit.

    Source unique partagée par StagedLrBetaCallback (paliers calendaires, exp20)
    et RewardAdaptiveLrCallback (paliers pilotés par le reward train, exp22)."""
    sched = getattr(trainer, "lr_scheduler", None)
    inner = getattr(sched, "scheduler", sched)  # déballe AcceleratedScheduler
    if inner is not None and hasattr(inner, "base_lrs"):
        inner.base_lrs = [lr] * len(inner.base_lrs)
    optimizer = optimizer or getattr(trainer, "optimizer", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            group["lr"] = lr
    trainer.beta = beta


def save_best_adapter(trainer: Any, best_dir: str, info: str,
                      save_optimizer: bool = False, tag: str = "best") -> bool:
    """Sauve l'adapter LoRA courant (+ optimizer si demandé) sur disque persistant.

    Swap atomique (.tmp puis os.replace) : en cas de coupure infra pendant
    l'écriture, l'ancien best reste intact. Le fichier .besttrain_info rend la
    reprise traçable. Erreur non fatale : le save est de la consolidation, pas
    du training. Source unique partagée par RewardAdaptiveLrCallback (exp22.1)
    et StagedBestRestoreCallback (exp22.5)."""
    import os
    import shutil
    try:
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        tmp = best_dir + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        unwrapped.save_pretrained(tmp)  # PEFT → adapter seul
        if save_optimizer:
            import torch
            optim = getattr(trainer, "optimizer", None)
            inner = getattr(optim, "optimizer", optim)  # déballe AcceleratedOptimizer
            if inner is not None:
                torch.save(inner.state_dict(), os.path.join(tmp, "optimizer.pt"))
        with open(os.path.join(tmp, ".besttrain_info"), "w") as f:
            f.write(info + "\n")
        shutil.rmtree(best_dir, ignore_errors=True)
        os.replace(tmp, best_dir)
        print(f"[{tag}] best train sauvé → {best_dir} ({info})", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] AVERTISSEMENT : save du best train raté ({e})", flush=True)
        return False


def restore_best_adapter(trainer: Any, best_dir: str,
                         restore_optimizer: bool = False, tag: str = "best") -> bool:
    """Recharge les poids de l'adapter « best » (+ moments Adam si demandé).

    Même mécanisme que la continuation d'adapter validée le 2026-07-21
    (set_peft_model_state_dict → injecte les poids SANS recréer d'adapter ni
    bouger l'ancre KL). Si restore_optimizer et optimizer.pt présent : les
    moments Adam DU MOMENT DU BEST sont rechargés (cohérents avec les poids) ;
    sinon les moments sont remis à zéro (l'optimizer courant garde l'élan de
    la trajectoire divergente). NB : load_state_dict recharge aussi
    param_groups (LR du best) → l'appelant doit repasser derrière avec
    apply_lr_beta. TRL resynchronise vLLM au step suivant (colocate)."""
    import os
    try:
        path = os.path.join(best_dir, "adapter_model.safetensors")
        if not os.path.exists(path):
            print(f"[{tag}] AVERTISSEMENT : pas de best à restaurer ({path} absent) "
                  f"— on continue avec les poids courants", flush=True)
            return False
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        set_peft_model_state_dict(unwrapped, load_file(path))
        optim = getattr(trainer, "optimizer", None)
        inner = getattr(optim, "optimizer", optim)  # déballe AcceleratedOptimizer
        opt_path = os.path.join(best_dir, "optimizer.pt")
        if inner is not None and restore_optimizer and os.path.exists(opt_path):
            import torch
            inner.load_state_dict(torch.load(opt_path, map_location="cpu",
                                             weights_only=True))
            opt_msg = "moments Adam restaurés (état du best)"
        elif inner is not None and hasattr(inner, "state"):
            inner.state.clear()
            opt_msg = "moments Adam remis à zéro"
        else:
            opt_msg = "optimizer introuvable (non touché)"
        info = ""
        info_path = os.path.join(best_dir, ".besttrain_info")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = " (" + f.read().strip() + ")"
        print(f"[{tag}] poids RESTAURÉS depuis le best train{info}, {opt_msg}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] AVERTISSEMENT : restore du best train raté ({e}) "
              f"— on continue avec les poids courants", flush=True)
        return False


class StagedLrBetaCallback(TrainerCallback):
    """LR (et beta KL) divisés par `factor` toutes les `every_epochs` epochs.

    Application du couple (LR, beta) : voir apply_lr_beta. `scale_beta=False`
    garde beta constant (ablation).

    Le palier est recalculé depuis `state.epoch` (stateless) → correct après un
    --resume-from-checkpoint. Nécessite `trainer_ref` assigné après construction
    (même pattern que TestEvalCallback)."""

    def __init__(self, base_lr: float, base_beta: float,
                 every_epochs: float = 3.0, factor: float = 3.0,
                 scale_beta: bool = True):
        self.base_lr = base_lr
        self.base_beta = base_beta
        self.every_epochs = every_epochs
        self.factor = factor
        self.scale_beta = scale_beta
        self.trainer_ref: Any = None
        self._stage = -1  # force l'application au tout premier step (et au resume)

    def on_step_begin(self, args, state, control, **kwargs):
        stage = int(float(state.epoch or 0.0) // self.every_epochs)
        if stage == self._stage:
            return
        self._stage = stage
        lr = self.base_lr / (self.factor ** stage)
        beta = (self.base_beta / (self.factor ** stage)) if self.scale_beta else self.base_beta
        apply_lr_beta(self.trainer_ref, lr, beta, optimizer=kwargs.get("optimizer"))
        print(f"[lr-beta-stage] step={state.global_step} epoch={float(state.epoch or 0.0):.2f} "
              f"palier={stage} lr={lr:.4e} beta={beta:.4e}", flush=True)


class StagedBestRestoreCallback(TrainerCallback):
    """Paliers calendaires LONGS + restore du best du palier à chaque coupe (exp22.5).

    Constat des lignées précédentes :
      - adaptatif (exp22/22.1/22.3) : les coupes arrivent APRÈS le décrochage
        (il faut `patience` checks de stagnation pour déclencher) ;
      - calendaire court (exp20 ÷3/3ep, exp22.4 ÷2/4ep) : coupe pendant la
        montée, mais consolide l'état COURANT — qui peut être une dérive.

    Synthèse (idée Vadim, 2026-07-31) :
      - chaque palier de LR dure `every_epochs` (10) epochs — assez long pour
        de vraies mesures stables ;
      - pendant le palier, on suit la moyenne roulante du reward train
        (fenêtre `window_epochs`, check toutes les `check_every_epochs`) ;
        à chaque nouveau best DU PALIER : save adapter + optimizer dans
        `stage_dir` (swap atomique) ; s'il bat aussi le best GLOBAL du run :
        promotion (copie) dans `global_dir` ;
      - à la frontière de palier : RESTORE du best du palier écoulé (poids +
        moments Adam du moment du best), PUIS LR/beta ÷ `factor` — le nouveau
        palier repart du meilleur état vu, pas de là où la trajectoire a fini.

    Le numéro de palier est recalculé depuis `state.epoch` (stateless, correct
    après resume) ; le suivi du best vit en mémoire mais les dirs sur disque
    survivent aux coupures. Nécessite `trainer_ref` assigné après construction."""

    def __init__(self, base_lr: float, base_beta: float,
                 every_epochs: float = 10.0, factor: float = 2.0,
                 window_epochs: float = 1.0, check_every_epochs: float = 0.5,
                 scale_beta: bool = True, stage_dir: str | None = None,
                 global_dir: str | None = None, save_optimizer: bool = True):
        self.base_lr = base_lr
        self.base_beta = base_beta
        self.every_epochs = every_epochs
        self.factor = factor
        self.window_epochs = window_epochs
        self.check_every_epochs = check_every_epochs
        self.scale_beta = scale_beta
        self.stage_dir = stage_dir      # best du palier courant (écrasé à chaque palier)
        self.global_dir = global_dir    # best global du run (promotion depuis stage_dir)
        self.save_optimizer = save_optimizer
        self.trainer_ref: Any = None
        self._stage = -1  # force l'application du LR au tout premier step (et au resume)
        self._lr = base_lr
        self._beta = base_beta
        self._history: list[tuple[float, float]] = []  # (epoch, reward mean du step)
        self._stage_best: float | None = None
        self._global_best: float | None = None
        self._next_check = max(window_epochs, check_every_epochs)
        self._in_log = False  # anti-réentrance (trainer.log() re-déclenche on_log)

    # -- transition de palier (calendaire, comme StagedLrBetaCallback) --------

    def on_step_begin(self, args, state, control, **kwargs):
        epoch = float(state.epoch or 0.0)
        stage = int(epoch // self.every_epochs)
        if stage == self._stage:
            return
        first = self._stage < 0
        self._stage = stage
        self._lr = self.base_lr / (self.factor ** stage)
        self._beta = (self.base_beta / (self.factor ** stage)) if self.scale_beta \
            else self.base_beta
        restored = False
        if not first and self.stage_dir is not None:
            # Restore AVANT apply_lr_beta : l'optimizer rechargé contient les LR
            # du moment du best dans param_groups, il faut repasser derrière.
            restored = self._restore_stage_best()
        apply_lr_beta(self.trainer_ref, self._lr, self._beta,
                      optimizer=kwargs.get("optimizer"))
        # Nouveau palier : best du palier remis à zéro, cooldown d'une fenêtre
        # complète (la moyenne roulante ne mélange pas les régimes avant/après).
        self._history = []
        self._stage_best = None
        self._next_check = epoch + self.window_epochs
        print(f"[lr-stage-restore] step={state.global_step} epoch={epoch:.2f} "
              f"palier={stage} lr={self._lr:.4e} beta={self._beta:.4e}"
              + (" — repart du best du palier précédent" if restored else ""),
              flush=True)

    # -- suivi du reward train et save des bests (comme RewardAdaptiveLr) -----

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._in_log or not logs or "reward" not in logs:
            return
        epoch = float(logs.get("epoch") or state.epoch or 0.0)
        self._history.append((epoch, float(logs["reward"])))
        if epoch < self._next_check:
            return
        self._next_check += self.check_every_epochs
        window = [r for e, r in self._history if e > epoch - self.window_epochs]
        if not window:
            return
        mean = sum(window) / len(window)
        promoted = False
        if self._stage_best is None or mean > self._stage_best:
            self._stage_best = mean
            info = (f"step={state.global_step} epoch={epoch:.2f} rolling_mean={mean:.4f} "
                    f"lr={self._lr:.4e} beta={self._beta:.4e} palier={self._stage}")
            if self.stage_dir is not None:
                self._save_stage_best(info)
            if self._global_best is None or mean > self._global_best:
                self._global_best = mean
                promoted = True
                if self.global_dir is not None and self.stage_dir is not None:
                    self._promote_global(info)
        print(f"[lr-stage-restore] step={state.global_step} epoch={epoch:.2f} "
              f"rolling_mean={mean:.4f} (n={len(window)}) "
              f"stage_best={self._stage_best:.4f} global_best={self._global_best:.4f} "
              f"palier={self._stage} lr={self._lr:.4e}"
              + (" >>> BEST GLOBAL" if promoted else ""), flush=True)
        if self.trainer_ref is not None:
            self._in_log = True
            try:
                self.trainer_ref.log({
                    "stagebest/rolling_reward_mean": mean,
                    "stagebest/stage_best": self._stage_best,
                    "stagebest/global_best": self._global_best,
                    "stagebest/stage": float(self._stage),
                    "stagebest/lr": self._lr,
                })
            finally:
                self._in_log = False

    # -- hooks disque (méthodes d'instance pour mockabilité du selftest) ------

    def _save_stage_best(self, info: str) -> None:
        save_best_adapter(self.trainer_ref, self.stage_dir, info,
                          save_optimizer=self.save_optimizer, tag="lr-stage-restore")

    def _restore_stage_best(self) -> bool:
        return restore_best_adapter(self.trainer_ref, self.stage_dir,
                                    restore_optimizer=self.save_optimizer,
                                    tag="lr-stage-restore")

    def _promote_global(self, info: str) -> None:
        """Copie le best du palier vers le best global (swap atomique)."""
        import os
        import shutil
        try:
            tmp = self.global_dir + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(self.stage_dir, tmp)
            shutil.rmtree(self.global_dir, ignore_errors=True)
            os.replace(tmp, self.global_dir)
            print(f"[lr-stage-restore] BEST GLOBAL promu → {self.global_dir} ({info})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[lr-stage-restore] AVERTISSEMENT : promotion du best global ratée "
                  f"({e})", flush=True)


# ---------------------------------------------------------------------------
# Selftest à sec (CPU) : python -m src.train.schedules
# ---------------------------------------------------------------------------
