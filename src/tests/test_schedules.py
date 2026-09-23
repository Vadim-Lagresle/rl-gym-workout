"""CPU self-tests of the training schedules and of the moving KL anchor.

In plain words: fast checks, with fake trainers and a tiny model, that the LR/beta
callbacks cut when they should, that the save/restore helpers are called in the
right order, that the horizon/budget schedules return the right caps, and that the
merge-and-restart anchor leaves the policy unchanged while moving the reference.
Run it before any GPU job:  python -m src.tests.test_schedules
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train.horizon_schedules import *  # noqa: F401,F403,E402
from src.train.horizon_schedules import _LAST_COMPLETION_CAP  # noqa: F401,E402
from src.train.kl_anchor import *  # noqa: F401,F403,E402
from src.train.lr_adaptive import *  # noqa: F401,F403,E402
from src.train.lr_schedules import *  # noqa: F401,F403,E402
import src.train.horizon_schedules as schedules  # noqa: E402  (les tests écrivent les variables de module)


def main() -> None:
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
    schedules.MAX_COMPLETION_SCHEDULE_EPOCHS = None
    assert current_max_completion(t8, default=512) == 512  # sans schedule : la CLI
    schedules.MAX_COMPLETION_SCHEDULE_EPOCHS = [(0.0, 256), (15.0, 512), (35.0, 1024)]
    assert current_max_completion(t8, default=1024) == 256
    t8.state.epoch = 14.99
    assert current_max_completion(t8, default=1024) == 256
    t8.state.epoch = 15.0
    assert current_max_completion(t8, default=1024) == 512
    t8.state.epoch = 60.0
    assert current_max_completion(t8, default=1024) == 1024
    t8.state = None  # avant le 1er step (trainer.state absent) : palier epoch 0
    assert current_max_completion(t8, default=1024) == 256
    schedules.MAX_COMPLETION_SCHEDULE_EPOCHS = None
    print("[schedules] selftest completion-schedule OK — paliers epoch, fallback CLI, "
          "état absent = palier 0")


if __name__ == "__main__":
    main()
