"""Schedules d'entraînement : ScalingInter (max_rounds) et paliers LR/beta.

1) Curriculum ScalingInter : budget d'interaction (max_rounds) progressif.
   Deux variantes, écrites en CLI comme des paires "<max_rounds>:<threshold>" :
     - pilotée par STEP   (--max-rounds-schedule)        : threshold = global step
     - pilotée par EPOCH  (--max-rounds-schedule-epochs) : threshold = epoch fractionnaire
   En interne : liste de (threshold, max_rounds) triée par threshold. None = cap
   fixe MAX_SIM_ROUNDS. Une seule variante active à la fois (main() valide).

2) StagedLrBetaCallback (ajout 2026-07-23, exp20) : LR et beta KL divisés par un
   facteur toutes les N epochs — automatise en un seul run l'échelle de LR que la
   lignée exp10 faisait à la main par warm-starts successifs (5e-6 → 3.33e-6 → …),
   et protège du collapse KL d'exp19.2 (survenu à l'epoch ~7.5 ENCORE à 5e-6).

3) RewardAdaptiveLrCallback (ajout 2026-07-27, exp22) : mêmes divisions LR/beta,
   mais pilotées par le REWARD TRAIN et non par le calendrier — constat exp20 :
   les paliers calendaires stabilisent mais coupent des dynamiques d'apprentissage
   en cours (cf. docs/hebdo/31juillet/session_2026-07-27_paliers_lr_et_reward_train.md).

4) StagedBestRestoreCallback (ajout 2026-07-31, exp22.5) : synthèse des deux —
   paliers CALENDAIRES longs (≥10 epochs, vraies mesures stables par palier)
   MAIS chaque frontière de palier repart du BEST du palier écoulé (poids +
   moments Adam), pas de l'état courant. Constat exp22.3/22.4 : l'adaptatif
   coupe trop tard (après le décrochage), le calendaire court coupe la montée
   et consolide l'état COURANT qui peut être une dérive post-pic.
"""

from __future__ import annotations

import statistics
from typing import Any

from transformers import TrainerCallback

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


class RewardAdaptiveLrCallback(TrainerCallback):
    """LR (et beta KL) divisés par `factor` quand le reward TRAIN stagne/décroît.

    Motivation (exp20, 2026-07-27) : les paliers calendaires (÷3 / 3 epochs)
    stabilisent et améliorent le pic test, mais coupent des dynamiques
    d'apprentissage en cours — ici la décroissance est pilotée par le signal
    d'apprentissage lui-même.

    Mécanique :
      - `on_log` collecte le reward train moyen de chaque step (clé 'reward' du
        dict loggé par GRPOTrainer, présente à chaque step avec logging_steps=1 ;
        les logs eval/ et nos propres logs adaptive/ n'ont pas cette clé → ignorés).
      - Toutes les `check_every_epochs` (défaut 0.5 = demi-epoch), on calcule la
        MOYENNE ROULANTE du reward sur la fenêtre `window_epochs` (défaut 1 epoch
        ≈ 47 steps × 64 traj ≈ 3000 épisodes → bruit du tirage d'items lissé,
        std de la moyenne ≈ 0.009). Premier check quand une fenêtre complète
        existe (epoch >= window_epochs).
      - Si la moyenne est SOUS le best historique moins `eps` (tolérance bruit),
        le compteur de checks « mauvais » s'incrémente ; sinon il retombe à 0
        (et le best est mis à jour si la moyenne le dépasse).
      - Après `patience` checks mauvais CONSÉCUTIFS (défaut 3 = 1.5 epoch), LR
        (et beta sauf scale_beta=False) sont divisés par `factor`. Plancher
        `min_lr`. Après la coupe (sémantique exp22.1) : historique VIDÉ, best
        remis à None, et prochain check seulement quand une fenêtre COMPLÈTE de
        rewards post-coupe existe — cooldown propre, la nouvelle référence ne
        mélange pas les régimes avant/après coupe.

    Save/restore du BEST TRAIN (exp22.1, si `best_dir` fourni — LoRA uniquement) :
      - à chaque nouveau best de moyenne roulante, l'adapter est sauvé sur
        disque persistant dans `best_dir` (swap atomique .tmp → dir, l'ancien
        best est remplacé ; + fichier .besttrain_info pour reprise après
        coupure infra) ;
      - à chaque coupe de LR, les poids de ce best sont RECHARGÉS dans le
        modèle (set_peft_model_state_dict) et les moments Adam sont remis à
        zéro : le LR réduit consolide le meilleur état connu au lieu de polir
        la dérive post-pic (leçon exp22 : 4 coupes tardives ont figé un état
        dégradé, test 49 → 24). TRL resynchronise les poids vers vLLM au step
        suivant (sync colocate par step). Best 100 % TRAIN : aucun lien avec
        le best test de TestEvalCallback.

    Durcissements exp22.2 (2026-07-29 — autopsie exp22.1 : 7 coupes pilotées par
    le bruit à cadence quasi minimale, LR au plancher dès l'epoch 28) :
      - `ref_median_k` > 0 : la détection de stagnation compare à la MÉDIANE des
        K derniers checks du palier au lieu du max historique (le max de tirages
        bruités est biaisé de +1-2σ → les fenêtres typiques paraissaient
        « mauvaises » par simple retour à la moyenne). Le max reste la cible du
        save/restore.
      - `min_stage_epochs` : durée minimale d'un palier de LR — une coupe
        déclenchée avant est retenue (n_bad conservé) et exécutée au premier
        check suivant si la stagnation persiste → chaque palier fournit de
        vraies mesures stables.
      - `save_optimizer` : l'état Adam (fp32) est sauvé avec le best et restauré
        à la coupe (au lieu du reset à zéro) — poids et moments cohérents.
      - `stop_at_floor` : une coupe qui passerait sous min_lr ARRÊTE le run
        (leçon exp22.1 : 18 epochs brûlées à LR ≈ 0).

    Télémétrie : lignes [reward-adaptive-lr] dans le log à chaque check +
    métriques adaptive/ dans wandb (rolling_mean, best, n_bad, lr, cuts).

    Limite connue : la fenêtre et la référence best vivent en mémoire → après
    un --resume-from-checkpoint elles repartent de zéro (le LR est relu depuis
    l'optimizer restauré ; l'adapter best_dir du run interrompu reste sur
    disque comme point de départ). Nécessite `trainer_ref` assigné après
    construction."""

    def __init__(self, base_lr: float, base_beta: float, factor: float = 3.0,
                 window_epochs: float = 1.0, check_every_epochs: float = 0.5,
                 patience: int = 3, eps: float = 0.005, min_lr: float = 1e-9,
                 scale_beta: bool = True, best_dir: str | None = None,
                 min_stage_epochs: float = 0.0, ref_median_k: int = 0,
                 save_optimizer: bool = False, stop_at_floor: bool = False):
        self.factor = factor
        self.window_epochs = window_epochs
        self.check_every_epochs = check_every_epochs
        self.patience = patience
        self.eps = eps
        self.min_lr = min_lr
        self.scale_beta = scale_beta
        self.best_dir = best_dir  # None = pas de save/restore (sémantique exp22)
        # Durcissements exp22.2 (défauts = sémantique exp22.1) :
        self.min_stage_epochs = min_stage_epochs  # durée minimale d'un palier de LR
        self.ref_median_k = ref_median_k          # K>0 : référence = médiane des K derniers checks
        self.save_optimizer = save_optimizer      # sauve/restaure aussi les moments Adam du best
        self.stop_at_floor = stop_at_floor        # stoppe le run au plancher min_lr
        self.trainer_ref: Any = None
        self._lr = base_lr
        self._beta = base_beta
        self._history: list[tuple[float, float]] = []  # (epoch, reward_mean du step)
        self._best: float | None = None
        self._n_bad = 0
        self._n_cuts = 0
        self._check_means: list[float] = []  # moyennes des checks du palier courant (réf médiane)
        self._stage_start_epoch = 0.0
        # Premier check quand une fenêtre complète existe, puis toutes les check_every.
        self._next_check = max(window_epochs, check_every_epochs)
        self._in_check = False  # anti-réentrance (trainer.log() re-déclenche on_log)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._in_check or not logs or "reward" not in logs:
            return
        epoch = float(logs.get("epoch") or state.epoch or 0.0)
        self._history.append((epoch, float(logs["reward"])))
        if epoch < self._next_check:
            return
        self._next_check += self.check_every_epochs
        self._check(state, control, epoch)

    def _reset_stage(self, epoch: float) -> None:
        """Nouveau palier : cooldown propre — historique et référence vidés, prochain
        check après une fenêtre COMPLÈTE de rewards du nouveau régime."""
        self._n_bad = 0
        self._best = None
        self._history = []
        self._check_means = []
        self._next_check = epoch + self.window_epochs
        self._stage_start_epoch = epoch

    def _check(self, state, control, epoch: float) -> None:
        window = [r for e, r in self._history if e > epoch - self.window_epochs]
        if not window:
            return
        mean = sum(window) / len(window)

        # 1) Suivi du best (max du palier) — pilote le SAVE de l'adapter uniquement.
        if self._best is None or mean > self._best:
            self._best = mean
            if self.best_dir is not None:
                self._save_best(state, epoch, mean)

        # 2) Détection de stagnation. Référence : max historique du palier (exp22.1)
        #    ou médiane des K derniers checks (exp22.2) — le max est biaisé de +1-2σ
        #    (c'est un max de tirages bruités), la médiane estime le niveau TYPIQUE
        #    du palier, donc eps y mesure une vraie régression et non le simple
        #    retour à la moyenne après une fenêtre chanceuse.
        if self.ref_median_k > 0:
            prev = self._check_means[-self.ref_median_k:]
            ref = statistics.median(prev) if prev else None
        else:
            ref = self._best
        if ref is not None and mean < ref - self.eps:
            self._n_bad += 1
        else:
            self._n_bad = 0  # au niveau de la référence : dynamique non cassée
        self._check_means.append(mean)

        # 3) Coupe (avec durée minimale de palier et plancher).
        cut = held = stop = False
        if self._n_bad >= self.patience:
            if epoch - self._stage_start_epoch < self.min_stage_epochs:
                # Palier trop jeune : coupe RETENUE. n_bad n'est pas remis à zéro —
                # si la stagnation persiste au premier check après min_stage_epochs,
                # la coupe part ; si le reward remonte d'ici là, n_bad retombe seul.
                held = True
            else:
                new_lr = self._lr / self.factor
                if new_lr < self.min_lr and self.stop_at_floor:
                    stop = True
                    if control is not None:
                        control.should_training_stop = True
                elif new_lr < self.min_lr:
                    self._reset_stage(epoch)  # plancher : nouvelle référence, pas de coupe
                else:
                    # Restore AVANT apply_lr_beta : le state_dict de l'optimizer
                    # rechargé contient les LR du moment du best dans param_groups,
                    # apply_lr_beta doit repasser derrière avec le LR réduit.
                    self._n_cuts += 1
                    cut = True
                    if self.best_dir is not None:
                        self._restore_best()
                    self._lr = new_lr
                    if self.scale_beta:
                        self._beta = self._beta / self.factor
                    apply_lr_beta(self.trainer_ref, self._lr, self._beta)
                    self._reset_stage(epoch)

        best_disp = mean if self._best is None else self._best  # best=None juste après coupe
        suffix = ""
        if cut:
            suffix = f" >>> COUPE #{self._n_cuts} (LR ÷{self.factor:g})"
        elif held:
            suffix = (f" (coupe retenue : palier {epoch - self._stage_start_epoch:.1f} ep "
                      f"< min {self.min_stage_epochs:g} ep)")
        elif stop:
            suffix = f" >>> PLANCHER {self.min_lr:.1e} atteint : ARRÊT du run"
        print(f"[reward-adaptive-lr] step={state.global_step} epoch={epoch:.2f} "
              f"rolling_mean={mean:.4f} (n={len(window)}) best={best_disp:.4f} "
              f"bad={self._n_bad}/{self.patience} lr={self._lr:.4e} beta={self._beta:.4e}"
              + suffix, flush=True)
        if self.trainer_ref is not None:
            self._in_check = True
            try:
                self.trainer_ref.log({
                    "adaptive/rolling_reward_mean": mean,
                    "adaptive/best": best_disp,
                    "adaptive/n_bad": float(self._n_bad),
                    "adaptive/lr": self._lr,
                    "adaptive/n_cuts": float(self._n_cuts),
                })
            finally:
                self._in_check = False

    def _save_best(self, state, epoch: float, mean: float) -> None:
        """Sauve l'adapter courant comme « best train » (délègue à save_best_adapter)."""
        info = (f"step={state.global_step} epoch={epoch:.2f} rolling_mean={mean:.4f} "
                f"lr={self._lr:.4e} beta={self._beta:.4e} n_cuts={self._n_cuts}")
        save_best_adapter(self.trainer_ref, self.best_dir, info,
                          save_optimizer=self.save_optimizer, tag="reward-adaptive-lr")

    def _restore_best(self) -> bool:
        """Recharge le best train dans le modèle (délègue à restore_best_adapter)."""
        return restore_best_adapter(self.trainer_ref, self.best_dir,
                                    restore_optimizer=self.save_optimizer,
                                    tag="reward-adaptive-lr")


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

class MovingAnchorCallback(TrainerCallback):
    """Ancre KL MOBILE pour LoRA (exp25) : « merge-and-restart » périodique.

    Contexte : avec PEFT, TRL n'a pas de modèle de référence séparé — l'ancre KL
    est « le modèle avec adapter désactivé », c.-à-d. les poids de BASE (Qwen nu).
    Sur un run long, le rappel beta·KL vers la base devient de plus en plus
    contraignant à mesure que la politique s'améliore (plafond ~27-29 de l'arm
    β0.01 d'exp24). TRL propose sync_ref_model (TR-DPO) mais lève explicitement
    NotImplementedError avec PEFT — d'où cette implémentation.

    À chaque frontière de N epochs (détectée à on_step_end) :
      1. snapshot de l'adapter courant dans <anchors_dir>/cycle<k> (~qq Mo, home) :
         le modèle reste reconstructible à tout moment comme
         base ⊕ merge(cycle1) ⊕ … ⊕ merge(cycle_k) ⊕ adapter_courant
         (script : src/utils/merge_anchor_chain.py) ;
      2. FUSION de l'adapter dans les poids de base (merge_adapter), puis OUBLI
         du flag `merged_adapters` : le delta appartient désormais à la base et
         ne sera JAMAIS défusionné (un unmerge ultérieur soustrairait les
         nouvelles matrices A/B, pas celles qui ont été fusionnées) ;
      3. réinitialisation de l'adapter EN PLACE (reset_lora_parameters :
         A=kaiming, B=0). La POLITIQUE est fonctionnellement inchangée (delta ≡ 0)
         mais l'ancre (adapter désactivé) devient la politique de ce point ;
      4. purge des moments Adam des paramètres LoRA (mêmes tenseurs donc
         l'optimizer reste valide, mais les moments décrivent l'ancien paysage).

    vLLM colocate : rien à faire — la politique étant identique, les poids déjà
    chargés restent corrects, et la sync TRL suivante (merge du nouvel adapter ≈ 0
    sur la base modifiée) reproduit les mêmes tenseurs.

    Effet de bord assumé (blog Thinking Machines) : B=0 relance le « warmup
    implicite » du LR effectif à chaque cycle — le run alterne donc des phases
    lentes post-ré-ancrage et des phases pleines, au bénéfice d'une KL qui
    mesure toujours la distance au DERNIER point de confiance et non à Qwen nu.
    """

    def __init__(self, every_epochs: float, anchors_dir: str,
                 initial_cycle: int = 0) -> None:
        self.every = every_epochs
        self.anchors_dir = anchors_dir
        # initial_cycle > 0 : reprise (--resume-from-checkpoint) d'un run qui a déjà
        # ré-ancré `initial_cycle` fois — sans quoi la frontière (cycle+1)·every serait
        # déjà dépassée à la reprise et le callback enchaînerait des ré-ancrages
        # parasites dès le 1er step (purge immédiate des moments Adam repris).
        self.cycle = initial_cycle
        self.trainer_ref: Any = None

    def on_step_end(self, targs: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.epoch is None or state.epoch + 1e-9 < (self.cycle + 1) * self.every:
            return
        import json
        import os
        import time

        import torch
        from peft.tuners.lora.layer import LoraLayer

        t0 = time.time()
        self.cycle += 1
        model = self.trainer_ref.accelerator.unwrap_model(self.trainer_ref.model)

        # 1. Snapshot de l'adapter sortant (maillon k de la chaîne).
        cycle_dir = os.path.join(self.anchors_dir, f"cycle{self.cycle}")
        model.save_pretrained(cycle_dir)

        with torch.no_grad():
            # 2. Fusion dans la base + oubli du merge (délibérément jamais défusionné).
            model.merge_adapter()
            n_layers = 0
            for module in model.modules():
                if isinstance(module, LoraLayer) and module.merged:
                    # 3. Adapter neuf en place (mêmes objets Parameter → l'optimizer
                    #    et ses param_groups restent valides sans reconstruction).
                    #    On ne réinitialise QUE les adapters effectivement fusionnés
                    #    (= actifs, la politique) : un modèle multi-adapters (exp34
                    #    MAGELLAN : sr_adapters/delayed_adapters sur le même modèle)
                    #    garderait sinon un estimateur écrasé à chaque ré-ancrage.
                    merged_names = list(module.merged_adapters)
                    module.merged_adapters.clear()
                    for name in merged_names:
                        module.reset_lora_parameters(name, True)
                    n_layers += 1

        # 4. Moments Adam des params LoRA : purge (torch les recrée à zéro au
        #    prochain step, lazy). L'optimizer peut être wrappé (accelerate).
        opt = self.trainer_ref.optimizer
        while hasattr(opt, "optimizer"):
            opt = opt.optimizer
        # Restreint à l'adapter POLITIQUE (.default.) : les adapters SR de MAGELLAN
        # (exp34) ont leur propre optimizer, jamais touché ici.
        lora_params = [p for n, p in model.named_parameters()
                       if "lora_" in n and ".default." in n]
        n_purged = sum(1 for p in lora_params if opt.state.pop(p, None) is not None)

        with open(os.path.join(self.anchors_dir, "chain.jsonl"), "a") as f:
            f.write(json.dumps({"cycle": self.cycle, "step": state.global_step,
                                "epoch": round(float(state.epoch), 3)}) + "\n")
        print(f"[moving-anchor] >>> RÉ-ANCRAGE #{self.cycle} @ step {state.global_step} "
              f"(epoch {state.epoch:.2f}) : adapter sauvé ({cycle_dir}), fusionné dans "
              f"la base ({n_layers} couches), adapter réinitialisé (A=kaiming, B=0), "
              f"{n_purged} états Adam purgés — l'ancre KL est désormais la politique de "
              f"ce point ({time.time() - t0:.1f}s)", flush=True)
        if self.trainer_ref is not None and hasattr(self.trainer_ref, "log"):
            self.trainer_ref.log({"anchor/cycle": float(self.cycle),
                                  "anchor/step": float(state.global_step)})


if __name__ == "__main__":
    from types import SimpleNamespace

    class _FakeOptim:
        def __init__(self, lr): self.param_groups = [{"lr": lr}]

    class _FakeSched:
        def __init__(self, lr): self.base_lrs = [lr]

    logged: list[dict] = []
    trainer = SimpleNamespace(lr_scheduler=_FakeSched(5e-6), optimizer=_FakeOptim(5e-6),
                              beta=0.01, log=lambda d: logged.append(d))
    cb = RewardAdaptiveLrCallback(base_lr=5e-6, base_beta=0.01, factor=3.0,
                                  window_epochs=1.0, check_every_epochs=0.5,
                                  patience=3, eps=0.005)
    cb.trainer_ref = trainer

    steps_per_epoch = 10
    # Phase 1 : reward qui monte (epochs 0→3) — aucun palier attendu.
    # Phase 2 : reward qui redescend (epochs 3→6) — 1re coupe après patience
    # checks sous le best, puis best resetté (cooldown naturel).
    def reward_at(epoch: float) -> float:
        return 0.2 + 0.1 * min(epoch, 3.0) - 0.08 * max(0.0, epoch - 3.0)

    state = SimpleNamespace(global_step=0, epoch=0.0)
    for i in range(6 * steps_per_epoch):
        epoch = (i + 1) / steps_per_epoch
        state.global_step, state.epoch = i + 1, epoch
        cb.on_log(None, state, None, logs={"reward": reward_at(epoch), "epoch": epoch})

    assert cb._n_cuts >= 1, "au moins une coupe attendue sur la phase décroissante"
    assert abs(trainer.optimizer.param_groups[0]["lr"] - 5e-6 / (3 ** cb._n_cuts)) < 1e-15
    assert abs(trainer.beta - 0.01 / (3 ** cb._n_cuts)) < 1e-12
    assert trainer.lr_scheduler.base_lrs[0] == trainer.optimizer.param_groups[0]["lr"]
    # Aucune coupe pendant la phase montante : la 1re coupe survient après epoch 3.
    assert all(d["adaptive/n_cuts"] == 0.0 for d in logged if d.get("adaptive/lr") == 5e-6)
    print(f"[schedules] selftest RewardAdaptiveLrCallback OK — {cb._n_cuts} coupe(s), "
          f"lr final {trainer.optimizer.param_groups[0]['lr']:.3e}, beta {trainer.beta:.3e}")

    # — Test 2 : save/restore du best train (exp22.1), hooks disque mockés —
    calls: list[tuple] = []
    cb2 = RewardAdaptiveLrCallback(base_lr=5e-6, base_beta=0.01, factor=3.0,
                                   window_epochs=1.0, check_every_epochs=0.5,
                                   patience=3, eps=0.005, best_dir="/tmp/fake_besttrain")
    cb2.trainer_ref = SimpleNamespace(lr_scheduler=_FakeSched(5e-6), optimizer=_FakeOptim(5e-6),
                                      beta=0.01, log=lambda d: None)
    cb2._save_best = lambda state, epoch, mean: calls.append(("save", state.global_step))
    cb2._restore_best = lambda: (calls.append(("restore",)), True)[1]
    state = SimpleNamespace(global_step=0, epoch=0.0)
    for i in range(6 * steps_per_epoch):
        epoch = (i + 1) / steps_per_epoch
        state.global_step, state.epoch = i + 1, epoch
        cb2.on_log(None, state, None, logs={"reward": reward_at(epoch), "epoch": epoch})
    saves = [c for c in calls if c[0] == "save"]
    restores = [c for c in calls if c[0] == "restore"]
    assert cb2._n_cuts >= 1 and len(restores) == cb2._n_cuts, "un restore par coupe attendu"
    assert len(saves) >= 2, "saves attendus : bests de la montée + re-best post-coupe"
    # Cooldown : le save qui suit le restore arrive une fenêtre COMPLÈTE après la coupe
    # (best resauvé sous le nouveau régime LR), donc APRÈS le restore dans la séquence.
    assert calls.index(restores[0]) < calls.index(saves[-1])
    print(f"[schedules] selftest save/restore OK — {len(saves)} save(s), "
          f"{len(restores)} restore(s), lr final "
          f"{cb2.trainer_ref.optimizer.param_groups[0]['lr']:.3e}")

    # — Test 3 : durcissements exp22.2 (référence médiane + palier min + stop plancher) —
    # 3a. Bruit pur autour d'un niveau constant : la référence MÉDIANE ne doit
    #     déclencher AUCUNE coupe (là où le max exp22.1 coupait sur le bruit).
    import random
    rng = random.Random(42)
    cb3 = RewardAdaptiveLrCallback(base_lr=5e-6, base_beta=0.01, factor=3.0,
                                   window_epochs=1.0, check_every_epochs=0.5,
                                   patience=3, eps=0.02, ref_median_k=5,
                                   min_stage_epochs=3.0)
    cb3.trainer_ref = SimpleNamespace(lr_scheduler=_FakeSched(5e-6), optimizer=_FakeOptim(5e-6),
                                      beta=0.01, log=lambda d: None)
    state = SimpleNamespace(global_step=0, epoch=0.0)
    for i in range(40 * steps_per_epoch):  # 40 epochs de plateau bruité (σ≈0.04/step ⇒ σ_fenêtre≈0.012)
        epoch = (i + 1) / steps_per_epoch
        state.global_step, state.epoch = i + 1, epoch
        cb3.on_log(None, state, None, logs={"reward": 0.45 + rng.gauss(0, 0.04), "epoch": epoch})
    assert cb3._n_cuts == 0, f"réf médiane + eps 2σ : 0 coupe attendue sur bruit pur, obtenu {cb3._n_cuts}"

    # 3b. Vraie régression rapide (dès l'epoch 2) : la coupe est RETENUE par
    #     min_stage_epochs=3 puis exécutée dès que le palier a 3 epochs.
    cb4 = RewardAdaptiveLrCallback(base_lr=5e-6, base_beta=0.01, factor=3.0,
                                   window_epochs=1.0, check_every_epochs=0.5,
                                   patience=2, eps=0.01, ref_median_k=5,
                                   min_stage_epochs=3.0, min_lr=1e-6, stop_at_floor=True)
    cb4.trainer_ref = SimpleNamespace(lr_scheduler=_FakeSched(5e-6), optimizer=_FakeOptim(5e-6),
                                      beta=0.01, log=lambda d: None)
    control = SimpleNamespace(should_training_stop=False)
    cut_epochs: list[float] = []
    _orig_reset = cb4._reset_stage
    def _spy_reset(epoch):
        cut_epochs.append(epoch)
        _orig_reset(epoch)
    cb4._reset_stage = _spy_reset
    state = SimpleNamespace(global_step=0, epoch=0.0)
    for i in range(12 * steps_per_epoch):  # reward décroît linéairement dès l'epoch 1
        epoch = (i + 1) / steps_per_epoch
        state.global_step, state.epoch = i + 1, epoch
        cb4.on_log(None, state, control, logs={"reward": max(0.0, 0.5 - 0.06 * epoch), "epoch": epoch})
        if control.should_training_stop:
            break
    assert cb4._n_cuts >= 1, "une vraie régression doit finir par couper"
    assert cut_epochs[0] >= 3.0, f"la 1re coupe doit attendre min_stage_epochs=3, obtenue à {cut_epochs[0]}"
    # 5e-6 → 1.67e-6 (>1e-6 OK) → prochaine coupe 5.6e-7 < 1e-6 → arrêt demandé.
    assert cb4._n_cuts == 1 and control.should_training_stop, \
        f"stop au plancher attendu après 1 coupe (n_cuts={cb4._n_cuts}, stop={control.should_training_stop})"
    print(f"[schedules] selftest exp22.2 OK — médiane: 0 coupe sur bruit ; "
          f"régression: coupe retenue jusqu'à l'epoch {cut_epochs[0]:.1f}, "
          f"puis arrêt au plancher (n_cuts={cb4._n_cuts})")

    # — Test 4 : StagedBestRestoreCallback (exp22.5) — paliers 2 ep (10 en réel),
    #   montée puis dérive dans le palier 0, hooks disque mockés.
    events: list[tuple] = []
    cb5 = StagedBestRestoreCallback(base_lr=1.25e-6, base_beta=0.01,
                                    every_epochs=2.0, factor=2.0,
                                    window_epochs=0.5, check_every_epochs=0.25,
                                    stage_dir="/tmp/fake_stagebest",
                                    global_dir="/tmp/fake_besttrain")
    cb5.trainer_ref = SimpleNamespace(lr_scheduler=_FakeSched(1.25e-6),
                                      optimizer=_FakeOptim(1.25e-6),
                                      beta=0.01, log=lambda d: None)
    cb5._save_stage_best = lambda info: events.append(("save",))
    cb5._restore_stage_best = lambda: (events.append(("restore",)), True)[1]
    cb5._promote_global = lambda info: events.append(("promote",))

    def reward5(epoch: float) -> float:
        # Palier 0 : monte jusqu'à 0.5 (epoch 1.2) puis DÉRIVE vers 0.3 ;
        # palier 1 (post-restore) : remonte doucement au-dessus de 0.5.
        if epoch <= 1.2:
            return 0.2 + 0.25 * epoch
        if epoch <= 2.0:
            return 0.5 - 0.25 * (epoch - 1.2)
        return 0.3 + 0.12 * (epoch - 2.0)

    state = SimpleNamespace(global_step=0, epoch=0.0)
    for i in range(4 * steps_per_epoch - 1):  # epoch max 3.9 : une seule frontière (2.0)
        epoch = (i + 1) / steps_per_epoch
        state.global_step, state.epoch = i + 1, epoch
        cb5.on_step_begin(None, state, None)
        cb5.on_log(None, state, None, logs={"reward": reward5(epoch), "epoch": epoch})
    restores = [e for e in events if e[0] == "restore"]
    saves = [e for e in events if e[0] == "save"]
    promotes = [e for e in events if e[0] == "promote"]
    assert len(restores) == 1, f"1 restore attendu (frontière epoch 2), obtenu {len(restores)}"
    assert cb5._stage == 1 and abs(cb5._lr - 1.25e-6 / 2) < 1e-18 and abs(cb5._beta - 0.005) < 1e-12
    assert cb5.trainer_ref.optimizer.param_groups[0]["lr"] == cb5._lr, "apply_lr_beta doit repasser après restore"
    assert len(saves) >= 3, "saves attendus : bests de la montée palier 0 + re-best palier 1"
    # La dérive de fin de palier 0 (0.5 → 0.3) ne doit PAS écraser le stage best.
    assert len(promotes) < len(saves) or cb5._global_best >= 0.45, "le best global suit les vrais pics"
    # Le palier 1 finit au-dessus du best global du palier 0 → au moins 2 promotions.
    assert len(promotes) >= 2, f"promotion du best global attendue au palier 1, obtenu {len(promotes)}"
    print(f"[schedules] selftest StagedBestRestore OK — {len(saves)} save(s), "
          f"1 restore à la frontière, {len(promotes)} promotion(s) globale(s), "
          f"lr final {cb5._lr:.3e}, beta {cb5._beta:.3e}")

    # ---- Selftest 6 : MovingAnchorCallback (exp25) — invariance de la politique,
    # déplacement de l'ancre, purge Adam, snapshot de la chaîne. CPU, ~2 s.
    import os
    import tempfile

    import torch
    from peft import LoraConfig as _LoraConfig, get_peft_model
    from torch import nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(16, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    torch.manual_seed(0)
    pm = get_peft_model(_Tiny(), _LoraConfig(r=4, lora_alpha=32, target_modules=["proj"]))
    with torch.no_grad():  # simule un adapter APPRIS (B ≠ 0, sinon delta ≡ 0)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.1)
    opt6 = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    x6 = torch.randn(4, 16)
    pm(x6).sum().backward()
    opt6.step()  # crée des moments Adam
    policy_before = pm(x6).detach().clone()
    with pm.disable_adapter():
        anchor_before = pm(x6).detach().clone()
    assert not torch.allclose(policy_before, anchor_before), "l'adapter simulé doit avoir un effet"
    assert len(opt6.state) > 0, "moments Adam attendus avant ré-ancrage"

    tmp6 = tempfile.mkdtemp()
    cb6 = MovingAnchorCallback(every_epochs=2.0, anchors_dir=tmp6)
    cb6.trainer_ref = SimpleNamespace(model=pm,
                                      accelerator=SimpleNamespace(unwrap_model=lambda m: m),
                                      optimizer=opt6, log=lambda d: None)
    cb6.on_step_end(None, SimpleNamespace(epoch=1.96, global_step=92), None)
    assert cb6.cycle == 0, "pas de ré-ancrage avant la frontière"
    cb6.on_step_end(None, SimpleNamespace(epoch=2.02, global_step=95), None)
    assert cb6.cycle == 1, "ré-ancrage attendu à la frontière"
    assert torch.allclose(policy_before, pm(x6).detach(), atol=1e-5), \
        "la POLITIQUE ne doit pas changer au ré-ancrage (merge + adapter nul)"
    with pm.disable_adapter():
        anchor_after = pm(x6).detach()
    assert torch.allclose(anchor_after, policy_before, atol=1e-5), \
        "l'ANCRE doit devenir la politique du point de ré-ancrage"
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            assert float(p.abs().max()) == 0.0, "B doit être réinitialisé à zéro"
    assert len(opt6.state) == 0, "moments Adam des params LoRA purgés"
    assert os.path.exists(os.path.join(tmp6, "cycle1", "adapter_model.safetensors"))
    assert os.path.exists(os.path.join(tmp6, "chain.jsonl"))
    cb6.on_step_end(None, SimpleNamespace(epoch=2.5, global_step=110), None)
    assert cb6.cycle == 1, "pas de re-déclenchement avant la frontière suivante"

    # Reprise (--resume-from-checkpoint) : initial_cycle=3 → aucun ré-ancrage parasite
    # aux epochs 15.x (frontière suivante = 16), là où cycle=0 en aurait enchaîné 3.
    cb7 = MovingAnchorCallback(every_epochs=4.0, anchors_dir=tempfile.mkdtemp(),
                               initial_cycle=3)
    cb7.on_step_end(None, SimpleNamespace(epoch=15.02, global_step=691), None)
    cb7.on_step_end(None, SimpleNamespace(epoch=15.98, global_step=735), None)
    assert cb7.cycle == 3, "reprise : pas de ré-ancrage avant l'epoch 16"
    print("[schedules] selftest MovingAnchor OK — politique invariante, ancre déplacée, "
          "B=0, moments purgés, chaîne écrite, reprise sans ré-ancrage parasite")

    # — Test : schedule du budget de sortie par tour (exp35) —
    # (le bloc __main__ est le module lui-même : on assigne la globale en direct)
    assert parse_max_completion_schedule_epochs("256:0,512:15,1024:35") == \
        [(0.0, 256), (15.0, 512), (35.0, 1024)]
    t8 = SimpleNamespace(state=SimpleNamespace(epoch=0.0, global_step=0))
    MAX_COMPLETION_SCHEDULE_EPOCHS = None
    assert current_max_completion(t8, default=512) == 512  # sans schedule : la CLI
    MAX_COMPLETION_SCHEDULE_EPOCHS = [(0.0, 256), (15.0, 512), (35.0, 1024)]
    assert current_max_completion(t8, default=1024) == 256
    t8.state.epoch = 14.99
    assert current_max_completion(t8, default=1024) == 256
    t8.state.epoch = 15.0
    assert current_max_completion(t8, default=1024) == 512
    t8.state.epoch = 60.0
    assert current_max_completion(t8, default=1024) == 1024
    t8.state = None  # avant le 1er step (trainer.state absent) : palier epoch 0
    assert current_max_completion(t8, default=1024) == 256
    MAX_COMPLETION_SCHEDULE_EPOCHS = None
    print("[schedules] selftest completion-schedule OK — paliers epoch, fallback CLI, "
          "état absent = palier 0")
