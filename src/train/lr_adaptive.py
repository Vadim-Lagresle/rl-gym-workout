"""Reward-adaptive learning-rate cuts (exp22 lineage).

In plain words: instead of lowering the learning rate on a calendar, this callback
watches the rolling mean of the training reward and cuts LR and beta when the
reward stops improving, then restores the best weights seen so far. It was a step
of the stabilisation study; the final recipe does not use it (see lr_schedules.py
for the context and the shared helpers).
"""

from __future__ import annotations

import statistics
from typing import Any

from transformers import TrainerCallback

from src.train.lr_schedules import apply_lr_beta, restore_best_adapter, save_best_adapter


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
