"""Port fidèle de MAGELLAN (Gaven et al. 2025) pour notre stack TRL/TextCraft — exp34.

Source : external/MAGELLAN (github.com/flowersteam/MAGELLAN), analysé dans
docs/MAGELLAN_ANALYSE.md. Le port suit leur code fonction par fonction ; chaque
classe/méthode cite le fichier:lignes d'origine. Défauts d'hyperparamètres = leur
configs/little_zoo/local_gpu_config_magellan.yaml (magellan_args).

Ce qui est transposé À L'IDENTIQUE :
  - les 4 goal samplers (goal_sampler.py) : Random / Online / EK-Online / MAGELLAN,
    y compris l'ε-greedy recuit exponentiel, le seuil lp<0.01→0, la formule du
    demi-buffer pour les samplers online ;
  - la tête SR (models.py SRHeadModuleFn) : MLP hidden→128, Tanh, →1, entraînée
    par BCE-with-logits, branchée sur le hidden state (dernier layer) du dernier
    token du contexte du but, en float32 ;
  - les adapters LoRA SÉPARÉS de la politique (sr_adapters, entraînés avec la tête
    car train_llm=true chez eux) + les adapters retardés gelés (delayed_adapters) ;
  - la compétence retardée par snapshots de poids (updater.py update_buffer /
    set_weights) : deque de maxlen N/recompute_freq+1, delayed ← plus vieux ;
  - la boucle d'update (main.py:311-371) : extension des buffers (goal, succès) →
    sr_update sur un échantillon pondéré par récence (p ∝ arange) → sampler.update
    (recompute de l'ALP tous les recompute_freq updates).

Adaptations IMPOSÉES par le changement de stack (documentées, pas des choix) :
  - lamorel (serving distribué + BaseModuleFunction) → forwards directs sur le
    modèle PEFT du GRPOTrainer, adapters commutés par set_adapter (toujours
    restaurés sur 'default' en finally, équivalent de leur updater.py:358) ;
  - « but » = item du train set TextCraft ; son contexte sémantique = l'observation
    initiale de l'env (commandes de craft + goal), cachée une fois pour toutes ;
  - leur sample() par épisode → un vecteur de probabilités pour le sampler TRL
    (WeightedRepeatSampler). Distribution IDENTIQUE : leur ε-greedy par tirage
    (proba ε uniforme, sinon ∝ lp) est exactement le mélange
    p = ε/n + (1-ε)·lp/Σlp (uniforme si Σlp=0) appliqué i.i.d. ;
  - « 1 update » chez eux = 1 update SAC ; chez nous = 1 optimizer step GRPO
    (même cadence : 1 collecte d'épisodes → 1 update policy → 1 sr_update).

Selftest CPU : python src/train/magellan.py (obligatoire avant tout run GPU).
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Défauts = magellan_args de configs/little_zoo/local_gpu_config_magellan.yaml
# ---------------------------------------------------------------------------
MAGELLAN_DEFAULTS = {
    "N": 100,                # horizon du retard de compétence, en updates
    "epsilon_start": 1.0,
    "epsilon_end": 0.2,
    "epsilon_decay": 320,    # updates (décroissance exponentielle)
    "buffer_size": 5000,     # buffer (but, succès) pour l'entraînement de la tête
    "batch_size": 256,       # buts par sr_update
    "recompute_freq": 32,    # updates entre deux recomputes de l'ALP
    "sr_lora_r": 16,         # rl_script_args.lora_r (leurs adapters, tous)
    "sr_lora_alpha": 32,     # rl_script_args.lora_alpha
    "sr_lr": 1e-4,           # rl_script_args.lr — passé tel quel à sr_update (updater.py:93)
    "gradient_batch_size": 32,  # micro-batch des forwards SR (eux : 128 sur Flan-T5 248M)
}


# ---------------------------------------------------------------------------
# goal_sampler.py:11-23 — interface
# ---------------------------------------------------------------------------
class GoalSampler:
    """Port de goal_sampler.py:11. goals = liste d'indices de lignes du dataset."""

    def __init__(self, goals: list[int]):
        self.goals = goals

    def sample(self) -> int:
        raise NotImplementedError

    def update(self, **kwargs: Any) -> dict | None:
        raise NotImplementedError

    def sampling_probs(self) -> np.ndarray:
        """Vecteur p sur les buts, équivalent en distribution à sample() (cf. docstring
        module). Consommé par WeightedRepeatSampler."""
        raise NotImplementedError


class RandomGoalSampler(GoalSampler):
    """Port de goal_sampler.py:26-38 — uniforme (≡ notre GRPO de base)."""

    def sample(self) -> int:
        return self.goals[np.random.randint(0, len(self.goals))]

    def update(self, **kwargs: Any) -> None:
        return None

    def sampling_probs(self) -> np.ndarray:
        return np.full(len(self.goals), 1.0 / len(self.goals))


def _annealed_epsilon(eps_start: float, eps_end: float, eps_decay: float, step: int) -> float:
    """goal_sampler.py:70 — ε recuit exponentiel."""
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * step / eps_decay)


def _expit(x: np.ndarray) -> np.ndarray:
    """Sigmoïde stable (≡ scipy.special.expit / leur F.sigmoid) sans dépendre de
    scipy, absent de l'env v2 : expit(x) = ½(1+tanh(x/2)), tanh borné = pas d'overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _mixture_probs(lp: np.ndarray, epsilon: float, n: int) -> np.ndarray:
    """Mélange ε-greedy en forme vectorielle (≡ goal_sampler.py:58-66 en distribution)."""
    uniform = np.full(n, 1.0 / n)
    s = float(np.sum(lp))
    if s == 0:
        return uniform
    return epsilon * uniform + (1.0 - epsilon) * (lp / s)


class OnlineGoalSampler(GoalSampler):
    """Port de goal_sampler.py:41-108 — ALP par BUT, deque par but, sans réseau."""

    def __init__(self, goals: list[int], epsilon_start: float, epsilon_end: float,
                 epsilon_decay: float, buffer_size: int):
        super().__init__(goals)
        self.idx_of = {g: i for i, g in enumerate(goals)}
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step = 0
        n = len(goals)
        self.lp, self.sr, self.sr_delayed = np.zeros(n), np.zeros(n), np.zeros(n)
        self.goals_success = [deque(maxlen=buffer_size) for _ in range(n)]

    def sample(self) -> int:
        return self.goals[np.random.choice(len(self.goals), p=self.sampling_probs())]

    def update(self, **kwargs: Any) -> dict:
        # goal_sampler.py:68-79
        self.epsilon = _annealed_epsilon(self.epsilon_start, self.epsilon_end,
                                         self.epsilon_decay, self.step)
        self.step += 1
        for g, r in zip(kwargs["goals"], kwargs["returns"]):
            self.goals_success[self.idx_of[g]].append(r)
        self.compute_lp()
        return {"sr": self.sr, "sr_delayed": self.sr_delayed, "lp": self.lp}

    def compute_lp(self) -> None:
        # goal_sampler.py:97-108 — formule du demi-buffer
        for i, buffer in enumerate(self.goals_success):
            if len(buffer) < 2:
                self.sr[i] = 0
                self.sr_delayed[i] = 0
            else:
                buffer_array = np.array(buffer)
                midpoint = len(buffer_array) // 2
                self.sr[i] = np.mean(buffer_array[midpoint:])
                self.sr_delayed[i] = np.mean(buffer_array[:midpoint])
        self.lp = np.abs(self.sr - self.sr_delayed)

    def sampling_probs(self) -> np.ndarray:
        return _mixture_probs(self.lp, self.epsilon, len(self.goals))


class EKOnlineGoalSampler(GoalSampler):
    """Port de goal_sampler.py:110-232 — ALP par GROUPE expert.

    Leurs groupes (grasp/grow_plants/…/impossibles) → nos depths TextCraft.
    depth_of : index de ligne dataset → depth (1..4)."""

    def __init__(self, goals: list[int], depth_of: dict[int, int],
                 epsilon_start: float, epsilon_end: float, epsilon_decay: float,
                 buffer_size: int):
        super().__init__(goals)
        self.depth_of = depth_of
        self.buckets = sorted(set(depth_of.values()))
        self.goals_by_bucket = {d: [g for g in goals if depth_of[g] == d] for d in self.buckets}
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step = 0
        self.lp_bucket = np.zeros(len(self.buckets))
        self.sr_bucket = np.zeros(len(self.buckets))
        self.sr_delayed_bucket = np.zeros(len(self.buckets))
        # goal_sampler.py:132 — une deque de succès PAR groupe
        self.goals_success = [deque(maxlen=buffer_size) for _ in self.buckets]

    def sample(self) -> int:
        # goal_sampler.py:134-155 — groupe ∝ LP puis but uniforme dans le groupe
        sum_lp = np.sum(self.lp_bucket)
        if np.random.rand() < self.epsilon or sum_lp == 0:
            bucket = self.buckets[np.random.randint(0, len(self.buckets))]
        else:
            p = self.lp_bucket / sum_lp
            bucket = self.buckets[np.random.choice(len(self.buckets), p=p)]
        cands = self.goals_by_bucket[bucket]
        return cands[np.random.randint(0, len(cands))]

    def update(self, **kwargs: Any) -> dict:
        # goal_sampler.py:157-174
        self.epsilon = _annealed_epsilon(self.epsilon_start, self.epsilon_end,
                                         self.epsilon_decay, self.step)
        self.step += 1
        for g, r in zip(kwargs["goals"], kwargs["returns"]):
            self.goals_success[self.buckets.index(self.depth_of[g])].append(r)
        self.compute_lp()
        return {"sr": self.sr_bucket, "sr_delayed": self.sr_delayed_bucket,
                "lp": self.lp_bucket}

    def compute_lp(self) -> None:
        # goal_sampler.py:193-205
        for i, buffer in enumerate(self.goals_success):
            if len(buffer) >= 2:
                buffer_array = np.array(buffer)
                midpoint = len(buffer_array) // 2
                self.sr_bucket[i] = np.mean(buffer_array[midpoint:])
                self.sr_delayed_bucket[i] = np.mean(buffer_array[:midpoint])
        self.lp_bucket = np.abs(self.sr_bucket - self.sr_delayed_bucket)

    def sampling_probs(self) -> np.ndarray:
        # groupe ∝ LP (mélange ε), puis uniforme dans le groupe
        p_bucket = _mixture_probs(self.lp_bucket, self.epsilon, len(self.buckets))
        p = np.zeros(len(self.goals))
        pos = {g: i for i, g in enumerate(self.goals)}
        for b, d in enumerate(self.buckets):
            cands = self.goals_by_bucket[d]
            for g in cands:
                p[pos[g]] = p_bucket[b] / len(cands)
        return p / p.sum()


# ---------------------------------------------------------------------------
# models.py:114-170 — SRHeadModuleFn → tête SR
# ---------------------------------------------------------------------------
def make_sr_head(hidden_size: int, device: Any, trainable: bool) -> Any:
    """models.py:137-143 — MLP hidden→128, Tanh, →1 ; delayed = gelée."""
    import torch
    head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 1),
    ).to(device)
    head.requires_grad_(trainable)
    return head


class MagellanEstimator:
    """Remplace la plomberie lamorel (agent.custom_module_fns / SACUpdater côté SR).

    Porte : SRHeadModuleFn.forward (models.py:145-163), sr_update
    (updater.py:88-139), update_buffer (updater.py:361-371), set_weights
    (updater.py:373-375). Les adapters sr/delayed vivent sur LE MÊME modèle que la
    politique (leur design), commutés par set_adapter et TOUJOURS restaurés sur
    'default' (updater.py:358) — la sync vLLM de TRL ne voit donc jamais autre
    chose que l'adapter politique actif."""

    SR_ADAPTER = "sr_adapters"          # magellan_args.sr_adapters
    DELAYED_ADAPTER = "delayed_adapters"

    def __init__(self, sr_lr: float, batch_size: int, gradient_batch_size: int,
                 sr_lora_r: int, sr_lora_alpha: int):
        self.sr_lr = sr_lr
        self.batch_size = batch_size
        self.gradient_batch_size = gradient_batch_size
        self.sr_lora_r = sr_lora_r
        self.sr_lora_alpha = sr_lora_alpha
        self.model: Any = None
        self.sr_head: Any = None
        self.delayed_head: Any = None
        self.optimizer_sr: Any = None    # créé paresseusement (updater.py:92-93)
        self.weights_buffer: deque | None = None  # updater.py:361-371
        self.goal_ids: list[list[int]] = []       # token ids du contexte de chaque but
        self.pad_token_id: int = 0
        self.device: Any = None

    # -- attache au modèle de la politique ---------------------------------
    def attach(self, model: Any, goal_token_ids: list[list[int]], pad_token_id: int) -> None:
        """Ajoute les adapters sr/delayed (LoRA r16/α32, leurs valeurs) au modèle PEFT
        de la politique et construit les deux têtes. Les paramètres des deux adapters
        sont mis hors-gradient (requires_grad=False) pour que l'optimizer de la
        POLITIQUE ne les voie jamais ; sr_update les réactive temporairement."""
        from peft import LoraConfig

        self.model = model
        self.goal_ids = goal_token_ids
        self.pad_token_id = pad_token_id
        self.device = next(model.parameters()).device

        base_cfg = model.peft_config["default"]
        for name in (self.SR_ADAPTER, self.DELAYED_ADAPTER):
            cfg = LoraConfig(
                r=self.sr_lora_r, lora_alpha=self.sr_lora_alpha,
                target_modules=list(base_cfg.target_modules),
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            model.add_adapter(name, cfg)
        model.set_adapter("default")  # l'adapter actif reste celui de la politique
        self._set_adapter_requires_grad(self.SR_ADAPTER, False)
        self._set_adapter_requires_grad(self.DELAYED_ADAPTER, False)

        hidden = model.config.hidden_size
        self.sr_head = make_sr_head(hidden, self.device, trainable=True)       # models.py:143
        self.delayed_head = make_sr_head(hidden, self.device, trainable=False)
        n_sr = sum(p.numel() for _, p in self._named_params(self.SR_ADAPTER, self.sr_head))
        print(f"[magellan] adapters {self.SR_ADAPTER}/{self.DELAYED_ADAPTER} ajoutés "
              f"(r={self.sr_lora_r}/α={self.sr_lora_alpha}) + têtes SR — "
              f"{n_sr / 1e6:.1f}M params SR, {len(self.goal_ids)} buts", flush=True)

    def _set_adapter_requires_grad(self, adapter: str, flag: bool) -> None:
        for n, p in self.model.named_parameters():
            if f".{adapter}." in n:
                p.requires_grad_(flag)

    def _named_params(self, adapter: str, head: Any) -> list[tuple[str, Any]]:
        """Ordre déterministe adapter puis tête — l'appariement sr↔delayed en dépend
        (équivalent des name filters de models.py:165-170)."""
        out = [(n, p) for n, p in self.model.named_parameters() if f".{adapter}." in n]
        out += [(f"head.{n}", p) for n, p in head.named_parameters()]
        return out

    # -- forward (models.py:145-163) ----------------------------------------
    def _forward_goals(self, goal_rows: list[int], adapter: str, head: Any,
                       require_grad: bool, restore_adapter: bool = True) -> Any:
        """Logits SR des buts goal_rows : hidden state (dernier layer) du DERNIER
        token du contexte, en float32, passé dans la tête (models.py:147-163 ;
        chez eux causal = position de fin de contexte).

        restore_adapter=False (sr_update) : l'adapter reste actif au retour —
        NÉCESSAIRE sous gradient checkpointing, car le backward RECALCULE le
        forward avec l'adapter actif à cet instant ; restaurer 'default' avant
        le backward rejouerait le forward avec le mauvais adapter (rangs
        différents → CheckpointError ; rangs égaux → gradients FAUX silencieux).
        Bug attrapé par le smoke GPU du job 45 (31/08)."""
        import torch

        ids = [self.goal_ids[g] for g in goal_rows]
        max_len = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(ids), max_len), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            attn[i, : len(x)] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)

        self.model.set_adapter(adapter)
        try:
            ctx = torch.enable_grad() if require_grad else torch.no_grad()
            with ctx:
                out = self.model(input_ids=input_ids, attention_mask=attn,
                                 output_hidden_states=True)
                last = out.hidden_states[-1]
                pos = attn.sum(dim=1) - 1
                emb = last[torch.arange(len(ids), device=self.device), pos]
                emb = emb.to(torch.float32)                 # models.py:157
                return head(emb).squeeze(-1)                # models.py:162
        finally:
            if restore_adapter:
                self.model.set_adapter("default")           # updater.py:358

    def scores(self, goal_rows: list[int], delayed: bool) -> np.ndarray:
        """Logits (sans grad, micro-batchés) — côté inférence de compute_lp."""
        import torch
        adapter = self.DELAYED_ADAPTER if delayed else self.SR_ADAPTER
        head = self.delayed_head if delayed else self.sr_head
        outs = []
        for s in range(0, len(goal_rows), self.gradient_batch_size):
            chunk = goal_rows[s: s + self.gradient_batch_size]
            outs.append(self._forward_goals(chunk, adapter, head, require_grad=False)
                        .detach().float().cpu().reshape(-1))
        return torch.cat(outs).numpy()

    # -- sr_update (updater.py:88-139) ---------------------------------------
    def sr_update(self, goal_rows: list[int], success: list[float]) -> float:
        import torch
        import torch.nn.functional as F

        self._set_adapter_requires_grad(self.SR_ADAPTER, True)
        try:
            if self.optimizer_sr is None:  # updater.py:92-93 — Adam, lr passé en kwargs
                self.optimizer_sr = torch.optim.Adam(
                    [p for _, p in self._named_params(self.SR_ADAPTER, self.sr_head)],
                    lr=self.sr_lr)
            success_t = torch.tensor(success, dtype=torch.float32, device=self.device)
            self.optimizer_sr.zero_grad()
            gas = math.ceil(len(goal_rows) / self.gradient_batch_size)  # updater.py:107
            total = 0.0
            for b in range(gas):
                sl = slice(b * self.gradient_batch_size, (b + 1) * self.gradient_batch_size)
                _rows, _succ = goal_rows[sl], success_t[sl]
                if len(_rows) <= 1:                          # updater.py:118
                    continue
                # restore_adapter=False : l'adapter SR reste actif pendant le
                # backward (recalcul du forward sous gradient checkpointing).
                sr = self._forward_goals(_rows, self.SR_ADAPTER, self.sr_head,
                                         require_grad=True,
                                         restore_adapter=False).reshape(-1)
                sr_loss = F.binary_cross_entropy_with_logits(sr, _succ)  # updater.py:128
                loss = sr_loss / gas                                     # updater.py:131
                loss.backward()
                total += float(loss.detach())
            self.optimizer_sr.step()                                     # updater.py:136
            return total
        finally:
            self.model.set_adapter("default")                # updater.py:358
            self._set_adapter_requires_grad(self.SR_ADAPTER, False)

    # -- update_buffer / set_weights (updater.py:361-375) --------------------
    def update_buffer(self, buff_size: int) -> None:
        if self.weights_buffer is None:                      # updater.py:362-367
            self.weights_buffer = deque(maxlen=buff_size)
        weights = [p.data.detach().clone()                   # updater.py:369
                   for _, p in self._named_params(self.SR_ADAPTER, self.sr_head)]
        self.weights_buffer.append(weights)

    def set_weights(self, idx: int) -> None:
        delayed = self._named_params(self.DELAYED_ADAPTER, self.delayed_head)
        assert len(delayed) == len(self.weights_buffer[idx])
        for (_, p), w in zip(delayed, self.weights_buffer[idx]):  # updater.py:373-375
            p.data.copy_(w)


class MAGELLANGoalSampler(GoalSampler):
    """Port de goal_sampler.py:235-303."""

    def __init__(self, goals: list[int], estimator: MagellanEstimator,
                 N: int, epsilon_start: float, epsilon_end: float,
                 epsilon_decay: float, recompute_freq: int):
        super().__init__(goals)
        self.estimator = estimator
        self.N = N
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step = 0
        self.recompute_freq = recompute_freq
        # goal_sampler.py:254-255 — 1er snapshot + ALP initiale
        self.estimator.update_buffer(buff_size=int(self.N / self.recompute_freq + 1))
        self.sr, self.sr_delayed, self.lp = self.compute_lp()

    def sample(self) -> int:
        # goal_sampler.py:257-265
        return self.goals[np.random.choice(len(self.goals), p=self.sampling_probs())]

    def update(self, **kwargs: Any) -> dict:
        # goal_sampler.py:267-276
        self.epsilon = _annealed_epsilon(self.epsilon_start, self.epsilon_end,
                                         self.epsilon_decay, self.step)
        self.step += 1
        if self.step % self.recompute_freq == 0:
            self.estimator.update_buffer(buff_size=int(self.N / self.recompute_freq + 1))
            self.sr, self.sr_delayed, self.lp = self.compute_lp()
        return {"sr": self.sr, "sr_delayed": self.sr_delayed, "lp": self.lp}

    def compute_lp(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # goal_sampler.py:286-303
        rows = list(range(len(self.goals)))
        self.estimator.set_weights(0)  # delayed ← plus vieux snapshot (idx=0)
        sr_delayed = _expit(self.estimator.scores(rows, delayed=True))
        sr = _expit(self.estimator.scores(rows, delayed=False))
        lp = np.abs(sr - sr_delayed)
        lp[lp < 0.01] = 0.0            # goal_sampler.py:301 — stabilité numérique
        return sr, sr_delayed, lp

    def sampling_probs(self) -> np.ndarray:
        return _mixture_probs(self.lp, self.epsilon, len(self.goals))


# ---------------------------------------------------------------------------
# main.py:245-371 — la boucle d'update, portée en callback (duck-typing HF)
# ---------------------------------------------------------------------------
class MagellanCallback:
    """Boucle d'update de main.py transposée : à CHAQUE optimizer step GRPO
    (≡ leur update SAC), dans le même ordre qu'eux :
      1. extension des buffers (goal, succès) avec les épisodes du step (main.py:311-313)
      2. sr_update sur un échantillon pondéré par récence p ∝ arange (main.py:350-367)
      3. goal_sampler.update (main.py:370) — recompute ALP tous les recompute_freq

    Les épisodes arrivent via record() (appelé par grpo_rollout_func). Duck-typing
    des TrainerCallback HF : __getattr__ renvoie un no-op pour tout on_*."""

    def __init__(self, sampler: GoalSampler, estimator: MagellanEstimator | None,
                 buffer_size: int, batch_size: int, depth_of: dict[int, int],
                 log_path: str = ""):
        self.sampler = sampler
        self.estimator = estimator
        self.goal_buffer: deque = deque(maxlen=buffer_size)     # main.py:252
        self.success_buffer: deque = deque(maxlen=buffer_size)  # main.py:253
        self.batch_size = batch_size
        self.depth_of = depth_of
        self.log_path = log_path
        self.pending: list[tuple[int, float]] = []
        self.trainer_ref: Any = None

    # rollout → (index de ligne dataset, retour 0/1)
    def record(self, pairs: list[tuple[int, float]]) -> None:
        self.pending.extend(pairs)

    def on_step_end(self, targs: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self.pending:
            return
        goals = [g for g, _ in self.pending]
        rets = [r for _, r in self.pending]
        self.pending = []

        self.goal_buffer.extend(goals)       # main.py:312
        self.success_buffer.extend(rets)     # main.py:313

        sr_loss = None
        if self.estimator is not None and len(self.goal_buffer) > 0:   # main.py:350
            p = np.arange(1, len(self.goal_buffer) + 1)                # main.py:352
            p = p / p.sum()                                            # main.py:353
            idx = np.random.choice(len(self.goal_buffer), size=self.batch_size, p=p)
            g = [self.goal_buffer[i] for i in idx]                     # main.py:355
            s = [self.success_buffer[i] for i in idx]                  # main.py:356
            sr_loss = self.estimator.sr_update(g, s)                   # main.py:357-367

        res = self.sampler.update(goals=goals, returns=rets)           # main.py:370
        self._log(state, res, sr_loss)

    # HF appelle toutes les méthodes on_* : no-op par défaut
    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)

    def _log(self, state: Any, res: dict, sr_loss: float | None) -> None:
        lp, sr = np.asarray(res["lp"]), np.asarray(res["sr"])
        logs: dict[str, float] = {"magellan/epsilon": float(self.sampler.epsilon),
                                  "magellan/buffer": float(len(self.goal_buffer))}
        if sr_loss is not None:
            logs["magellan/sr_loss"] = float(sr_loss)
        # agrégats par depth (l'histoire complète va dans le jsonl)
        if lp.shape and len(lp) == len(self.sampler.goals):
            probs = self.sampler.sampling_probs()
            for d in sorted(set(self.depth_of.values())):
                rows = [i for i, g in enumerate(self.sampler.goals)
                        if self.depth_of[g] == d]
                logs[f"magellan/lp_d{d}"] = float(np.mean(lp[rows]))
                logs[f"magellan/sr_d{d}"] = float(np.mean(sr[rows]))
                logs[f"magellan/p_d{d}"] = float(np.sum(probs[rows]))
        if self.trainer_ref is not None and hasattr(self.trainer_ref, "log"):
            try:
                self.trainer_ref.log(logs)
            except Exception:
                pass
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"step": int(state.global_step),
                                    **{k: round(v, 5) for k, v in logs.items()}}) + "\n")


# ---------------------------------------------------------------------------
# Intégration TRL : sampler pondéré (remplace RepeatSampler) + schedule depth
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Contexte des buts : l'observation initiale de chaque item (cachée sur disque)
# ---------------------------------------------------------------------------
def load_goal_texts(rows: list[dict], cache_path: str) -> list[str]:
    """Contexte sémantique de chaque but = observation initiale TextCraft
    (commandes de craft + goal) — l'équivalent de leur description textuelle de
    but. Sondée une fois par item (ThreadPool) puis cachée en JSON."""
    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    missing = [r for r in rows if r["item_id"] not in cache]
    if missing:
        from concurrent.futures import ThreadPoolExecutor

        from agentenv.envs import TextCraftEnvClient

        from src.train.data import ENV_SERVER_URL

        def probe(r: dict) -> tuple[str, str]:
            env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000,
                                     timeout=60)
            env.reset(r["item_idx"])
            obs = env.observe()
            try:
                env.close()
            except Exception:
                pass
            return r["item_id"], obs

        print(f"[magellan] sonde des contextes de {len(missing)} buts...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            for item_id, obs in ex.map(probe, missing):
                cache[item_id] = obs
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return [cache[r["item_id"]] for r in rows]


# ---------------------------------------------------------------------------
# Selftest CPU (obligatoire avant tout run GPU)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    np.random.seed(0)

    # Modèle causal minuscule + adapter politique 'default' (comme le trainer)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=128)
    base = LlamaForCausalLM(cfg)
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

    n_goals = 20
    goal_ids = [[(7 * i + j) % 120 + 1 for j in range(6 + i % 3)] for i in range(n_goals)]
    probe_ids = torch.tensor([goal_ids[0] + [1] * 2])
    with torch.no_grad():
        logits_before = model(input_ids=probe_ids).logits.clone()

    est = MagellanEstimator(sr_lr=1e-2, batch_size=16, gradient_batch_size=8,
                            sr_lora_r=4, sr_lora_alpha=8)
    est.attach(model, goal_ids, pad_token_id=0)

    # 1. La politique est INTACTE : adapters sr inactifs, forward identique.
    with torch.no_grad():
        logits_after = model(input_ids=probe_ids).logits
    assert torch.allclose(logits_before, logits_after), "politique modifiée par attach()"

    # 2. Hygiène des gradients : aucun param sr/delayed vu par l'optimizer politique.
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if ".sr_adapters." in n or ".delayed_adapters." in n)

    # 3. Phase A — la tête apprend d'abord "tout échoue" (SR ≈ 0 partout), PUIS on
    #    crée le sampler : son 1er snapshot capture cet état (sinon le snapshot
    #    serait l'init aléatoire SR≈0.5 et l'ALP absolue |ΔSR| verrait de la
    #    "calibration" sur TOUS les buts — dans le vrai run c'est le rôle du ε=1
    #    recuit sur epsilon_decay updates d'absorber cette phase).
    for _ in range(60):
        rows = list(np.random.choice(n_goals, size=16))
        est.sr_update(rows, [0.0] * len(rows))
    sampler = MAGELLANGoalSampler(list(range(n_goals)), est, N=8, epsilon_start=1.0,
                                  epsilon_end=0.2, epsilon_decay=10, recompute_freq=2)
    assert np.allclose(sampler.sampling_probs(), 1.0 / n_goals)  # ε=1 → uniforme

    # 4. Phase B — les buts 0-9 se mettent à réussir : la tête doit les séparer.
    #    Seuils recalibrés le 31/08 : depuis le fix du backward sous adapter actif,
    #    les adapters SR s'entraînent VRAIMENT (avant, requires_grad retombait à
    #    False avant le backward et seule la tête apprenait — gradients adapters
    #    silencieusement jetés). La dynamique conjointe tête+adapters du jouet
    #    est plus lente/diffuse : mesuré 0.600 vs 0.223 (écart 0.377), seuils
    #    posés avec marge.
    for _ in range(80):
        rows = list(np.random.choice(n_goals, size=16))
        succ = [1.0 if r < 10 else 0.0 for r in rows]
        est.sr_update(rows, succ)
    probs = _expit(est.scores(list(range(n_goals)), delayed=False))
    assert probs[:10].mean() > 0.5 and probs[10:].mean() < 0.35 \
        and probs[:10].mean() - probs[10:].mean() > 0.25, \
        f"la tête SR n'a pas séparé les deux groupes : {probs.round(2)}"
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if ".sr_adapters." in n), "requires_grad non restauré après sr_update"

    # 5. Compétence retardée : delayed ← snapshot de phase A ("tout échoue") →
    #    l'ALP isole les buts en progrès (0-9) et reste ~nulle sur les stables (10-19).
    sampler.step = sampler.recompute_freq - 1
    res = sampler.update(goals=[0], returns=[1.0])   # déclenche le recompute
    # Seuils recalibrés 31/08 (adapters réellement entraînés) : les buts stables
    # dérivent légèrement (~0.22, couplage de représentations) — le critère est
    # la SÉPARATION, lp progrès > 2× lp stables (mesuré : ~0.60 vs ~0.22).
    assert res["lp"][:10].mean() > 0.3, f"lp devrait détecter le progrès : {res['lp'].round(2)}"
    assert res["lp"][:10].mean() > 2.0 * res["lp"][10:].mean(), \
        f"lp progrès devrait dominer lp stables : {res['lp'].round(2)}"

    # 6. Mélange ε-greedy : ε=0 → masse sur les buts à progrès ; ε=1 → uniforme.
    sampler.epsilon = 0.0
    p = sampler.sampling_probs()
    assert p[:10].sum() > 0.65, f"masse attendue sur les buts en progrès : {p.round(3)}"
    sampler.epsilon = 1.0
    assert np.allclose(sampler.sampling_probs(), 1.0 / n_goals)

    # 7. WeightedRepeatSampler : longueur = formule RepeatSampler TRL, tirages ∝ p.
    ws = WeightedRepeatSampler(list(range(7)), mini_repeat_count=2, batch_size=3,
                               repeat_count=4, prob_fn=lambda: np.ones(7) / 7, seed=0)
    assert len(list(ws)) == len(ws) == 48   # exemple de trl/trainer/utils.py:717-727
    conc = np.zeros(5)
    conc[2] = 1.0
    ws2 = WeightedRepeatSampler(list(range(5)), 1, 2, 1, prob_fn=lambda: conc, seed=0)
    assert all(i == 2 for i in ws2)

    # 8. DepthScheduleProvider : paliers.
    depths = [1, 1, 2, 3, 4]
    prov = DepthScheduleProvider("1:0,2:6,3:20,4:45", depths)
    prov.trainer_ref = SimpleNamespace(state=SimpleNamespace(epoch=0.5))
    assert prov.probabilities()[:2].sum() == 1.0
    prov.trainer_ref.state.epoch = 21.0
    p = prov.probabilities()
    assert p[3] > 0 and p[4] == 0.0
    prov.trainer_ref.state.epoch = 50.0
    assert prov.probabilities()[4] > 0

    # 9. EK-Online par depth : LP d'un groupe qui progresse > groupe stable.
    ek = EKOnlineGoalSampler(list(range(5)), dict(enumerate(depths)),
                             epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
                             buffer_size=8)
    for k in range(8):
        ek.update(goals=[0, 2], returns=[1.0 if k >= 4 else 0.0, 0.0])
    assert ek.lp_bucket[0] > 0.5 and ek.lp_bucket[1] == 0.0
    assert ek.sampling_probs()[:2].sum() > 0.9

    # 10. Callback bout-en-bout (ordre main.py:311-371) sur le sampler MAGELLAN.
    cb = MagellanCallback(sampler, est, buffer_size=100, batch_size=8,
                          depth_of={i: 1 + i // 10 for i in range(n_goals)})
    cb.record([(i % n_goals, 1.0 if i % n_goals < 10 else 0.0) for i in range(16)])
    cb.on_step_end(None, SimpleNamespace(global_step=1, epoch=0.1), None)
    assert len(cb.goal_buffer) == 16 and not cb.pending
    cb.on_save(None, None, None)  # __getattr__ : no-op HF

    # 11. DepthAutoScheduleProvider : passage au succès, cap d'epochs, masque.
    auto = DepthAutoScheduleProvider(depths=[1, 1, 2, 3, 4], steps_per_epoch=5,
                                     threshold=0.8, max_stage_epochs=10.0)
    st11 = SimpleNamespace(epoch=0.0, global_step=0)
    assert auto.probabilities()[:2].sum() == 1.0          # palier 1 : items d1 seuls
    for i in range(4):                                    # 4 logs à 0.9 : fenêtre incomplète
        auto.on_log(None, st11, None, logs={"reward": 0.9, "epoch": 0.1 * (i + 1)})
    assert auto.stage == 1, "pas de passage avant une epoch complète de mesures"
    auto.on_log(None, st11, None, logs={"reward": 0.9, "epoch": 0.5})
    assert auto.stage == 2, "passage attendu : moyenne 0.9 >= 0.8 sur fenêtre pleine"
    assert len(auto.window) == 0, "fenêtre vidée au passage (cooldown)"
    p11 = auto.probabilities()
    assert p11[3] == 0.0 and p11[:3].sum() == 1.0         # palier 2 : d1+d2
    for i in range(5):                                    # reward faible : pas de passage au succès
        auto.on_log(None, st11, None, logs={"reward": 0.2, "epoch": 1.0 + i})
    assert auto.stage == 2, "reward 0.2 < 0.8 : pas de passage anticipé"
    auto.on_log(None, st11, None, logs={"reward": 0.2, "epoch": 10.6})  # 10.6-0.5 >= 10
    assert auto.stage == 3, "cap 10 epochs : passage forcé"
    auto.on_log(None, st11, None, logs={"reward": 0.9, "epoch": 20.7})
    for _ in range(5):
        auto.on_log(None, st11, None, logs={"reward": 0.9, "epoch": 20.8})
    assert auto.stage == 4, "passage au succès vers le palier final"
    for _ in range(6):
        auto.on_log(None, st11, None, logs={"reward": 1.0, "epoch": 40.0})
    assert auto.stage == 4, "aucun passage au-delà de la profondeur max"

    # 12. sr_update SOUS GRADIENT CHECKPOINTING (régression du bug smoke 31/08) :
    #     modèle frais, adapter politique r=4, adapters SR r=6 (rangs DIFFÉRENTS
    #     pour que tout recalcul avec le mauvais adapter casse en CheckpointError
    #     au lieu de produire des gradients faux silencieux), checkpointing
    #     non-réentrant comme TRL. Avant le fix (backward après restauration de
    #     l'adapter 'default'), ce test levait CheckpointError.
    base12 = LlamaForCausalLM(cfg)
    model12 = get_peft_model(base12, LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model12.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    est12 = MagellanEstimator(sr_lr=1e-2, batch_size=8, gradient_batch_size=4,
                              sr_lora_r=6, sr_lora_alpha=12)
    est12.attach(model12, goal_ids, pad_token_id=0)
    loss12 = est12.sr_update(list(range(8)), [1.0, 0.0] * 4)
    assert loss12 == loss12 and loss12 > 0.0, "loss SR finie attendue sous checkpointing"
    # L'adapter actif est bien restauré et l'hygiène des gradients tient.
    assert model12.active_adapter == "default" or "default" in str(model12.active_adapter)
    assert not any(p.requires_grad for n, p in model12.named_parameters()
                   if ".sr_adapters." in n), "requires_grad non restauré (test 12)"

    print("[magellan] selftest OK — politique intacte, hygiène requires_grad, tête SR "
          "qui apprend (BCE), delayed par snapshot, lp>0 sur progrès, mélange ε, "
          "WeightedRepeatSampler conforme RepeatSampler, paliers depth, EK-depth, callback, "
          "depth-auto (succès + cap), sr_update sous gradient checkpointing")
