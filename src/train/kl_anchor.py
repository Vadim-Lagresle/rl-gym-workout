"""The moving KL reference ("moving anchor") for LoRA training.

In plain words: GRPO penalises the policy for moving away from a reference model
(the KL term). With a fixed reference (the base model), LoRA training drifts away
geometrically and collapses. These callbacks move the reference to the current
policy every N epochs, so the penalty only resists fast drift. Two mechanisms:

- MovingAnchorCallback ("merge" mode, all runs exp25-exp45): merge the adapter into
  the base weights, restart a fresh adapter and reset Adam. This is in fact the
  ReLoRA method (rank accumulates across cycles) — see the report, Section 4.2.4.
- MovingRefAdapterCallback ("ref" mode, exp46): keep one living adapter that is never
  reset, and a frozen copy named 'ref' that TRL uses as the KL reference; the copy is
  refreshed every N epochs. It isolates the effect of moving the reference.

Selected with --moving-anchor-every-epochs and --moving-anchor-mode in train_grpo.py.
"""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


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


class MovingRefAdapterCallback(TrainerCallback):
    """Ancre KL MOBILE SANS merge-and-restart (exp46, « vraie LoRA à référence mobile »).

    Objet : séparer les trois facteurs confondus dans MovingAnchorCallback (ReLoRA) —
    déplacement de la référence, accumulation de rang + reset du produit B·A, purge
    d'Adam. Ici UN SEUL adaptateur vivant (`default`, rang r pendant tout le run,
    jamais fusionné, jamais réinitialisé, Adam jamais purgé) et un second adaptateur
    FIGÉ nommé `ref`, de même config, qui sert de référence KL.

    Mécanique côté TRL (grpo_trainer, calcul des ref-logprobs avec PEFT) :
        with use_adapter(model, "ref" if "ref" in model.peft_config else None): ...
    → si un adaptateur `ref` existe, TRL l'active pour le passage avant de référence
    au lieu de désactiver l'adaptateur (= Qwen nu). Il ne manque que la recopie
    périodique : c'est ce callback.

    À chaque frontière de N epochs (on_step_end) :
      1. snapshot de l'adaptateur vivant dans <anchors_dir>/cycle<k> (traçabilité,
         et reprise : `initial_cycle` recharge cycle<k> dans `ref`) ;
      2. copie en place des matrices A et B de `default` vers `ref` → la référence
         devient la politique de ce point, la politique elle-même est inchangée.
    Rien d'autre ne bouge : ni la base, ni l'optimiseur.

    Invariants (vérifiés par src/train/selftest_moving_ref.py) : `default` reste
    l'adaptateur actif ; `ref` a requires_grad=False (hors optimiseur, hors
    gradient) ; juste après une copie, logits(ref) == logits(default) donc KL = 0 ;
    la sync vLLM de TRL fusionne l'adaptateur ACTIF seulement et ignore les clés
    `lora_` → `ref` n'atteint jamais le moteur de génération.
    """

    REF = "ref"

    def __init__(self, every_epochs: float, anchors_dir: str,
                 initial_cycle: int = 0, reset_adam: bool = False) -> None:
        self.every = every_epochs
        self.anchors_dir = anchors_dir
        self.cycle = initial_cycle
        # reset_adam (--moving-ref-reset-adam, test « Adam ou adaptateur ? ») : purge en plus
        # les moments Adam de l'adaptateur vivant à chaque ré-ancrage, comme l'étape 4 du mode
        # merge — l'adaptateur, lui, n'est toujours ni fusionné ni réinitialisé.
        self.reset_adam = reset_adam
        self.trainer_ref: Any = None

    # ---- appelé UNE fois après la création du GRPOTrainer (avant train()) ----
    def attach(self, trainer: Any) -> None:
        import copy
        import os

        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        self.trainer_ref = trainer
        model = trainer.accelerator.unwrap_model(trainer.model)
        if self.REF in model.peft_config:
            raise RuntimeError("un adaptateur 'ref' existe déjà sur le modèle")
        live = model.active_adapter
        ref_cfg = copy.deepcopy(model.peft_config[live])
        ref_cfg.inference_mode = True
        # add_adapter ne change pas l'adaptateur actif ; init PEFT : A=kaiming, B=0 →
        # `ref` ≡ base au départ, la KL commence à 0 comme avec l'ancre fixe.
        model.add_adapter(self.REF, ref_cfg)
        model.set_adapter(live)
        n_ref = self._freeze_ref(model)
        if self.cycle > 0:
            # Reprise : la référence est la politique du dernier ré-ancrage = snapshot cycle<k>.
            cyc = os.path.join(self.anchors_dir, f"cycle{self.cycle}", "adapter_model.safetensors")
            sd = load_file(cyc)
            res = set_peft_model_state_dict(model, sd, adapter_name=self.REF)
            print(f"[moving-ref] reprise : 'ref' rechargé depuis {cyc} "
                  f"(clés inattendues : {len(getattr(res, 'unexpected_keys', []))})", flush=True)
        print(f"[moving-ref] adaptateur de référence 'ref' ajouté ({n_ref} tenseurs, figés) ; "
              f"actif = '{model.active_adapter}' ; recopie default→ref toutes les "
              f"{self.every:g} epochs", flush=True)

    def _freeze_ref(self, model: Any) -> int:
        n = 0
        for name, p in model.named_parameters():
            if f".{self.REF}." in name:
                p.requires_grad_(False); n += 1
        return n

    @staticmethod
    def _lora_pairs(model: Any, src: str, dst: str):
        """Couples (tenseur source, tenseur destination) A et B de chaque couche LoRA."""
        from peft.tuners.lora.layer import LoraLayer
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for attr in ("lora_A", "lora_B"):
                d = getattr(module, attr)
                if src in d and dst in d:
                    yield d[src].weight, d[dst].weight

    def copy_live_to_ref(self, model: Any) -> int:
        import torch
        n = 0
        with torch.no_grad():
            for s, d in self._lora_pairs(model, model.active_adapter, self.REF):
                d.copy_(s); n += 1
        self._freeze_ref(model)
        return n

    def purge_adam(self, model: Any, live: str) -> int:
        """Même purge que l'étape 4 de MovingAnchorCallback, restreinte à l'adaptateur vivant."""
        opt = self.trainer_ref.optimizer
        while hasattr(opt, "optimizer"):
            opt = opt.optimizer
        lora_params = [p for n, p in model.named_parameters()
                       if "lora_" in n and f".{live}." in n]
        return sum(1 for p in lora_params if opt.state.pop(p, None) is not None)

    def on_step_end(self, targs: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.epoch is None or state.epoch + 1e-9 < (self.cycle + 1) * self.every:
            return
        import json
        import os
        import time

        t0 = time.time()
        self.cycle += 1
        model = self.trainer_ref.accelerator.unwrap_model(self.trainer_ref.model)
        live = model.active_adapter
        cycle_dir = os.path.join(self.anchors_dir, f"cycle{self.cycle}")
        model.save_pretrained(cycle_dir, selected_adapters=[live])
        n = self.copy_live_to_ref(model)
        n_purged = self.purge_adam(model, live) if self.reset_adam else 0
        os.makedirs(self.anchors_dir, exist_ok=True)
        with open(os.path.join(self.anchors_dir, "chain.jsonl"), "a") as f:
            f.write(json.dumps({"cycle": self.cycle, "step": state.global_step,
                                "epoch": round(float(state.epoch), 3), "mode": "ref",
                                "adam_purged": n_purged}) + "\n")
        opt_msg = (f"{n_purged} états Adam purgés (adaptateur et base inchangés)" if self.reset_adam
                   else "adaptateur, base et optimiseur inchangés")
        print(f"[moving-ref] >>> RÉ-ANCRAGE #{self.cycle} @ step {state.global_step} "
              f"(epoch {state.epoch:.2f}) : adaptateur vivant sauvé ({cycle_dir}), "
              f"{n} tenseurs A/B recopiés dans 'ref' — la référence KL est la politique de "
              f"ce point ; {opt_msg} ({time.time() - t0:.1f}s)",
              flush=True)
        if hasattr(self.trainer_ref, "log"):
            self.trainer_ref.log({"anchor/cycle": float(self.cycle),
                                  "anchor/step": float(state.global_step)})
