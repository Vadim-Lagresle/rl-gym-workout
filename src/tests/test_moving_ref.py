"""CPU self-test of the frozen-copy KL reference (moving-anchor mode 'ref').

In plain words: checks on a tiny model that the reference adapter is added without
changing what is trained, equals the base model at the start, becomes an exact copy of
the policy at each re-anchoring (KL back to zero), is never sent to the generation
engine, and is reloaded correctly when a run resumes. Run: python -m
src.tests.test_moving_ref

Notes (FR) — Test à sec (CPU, modèle Qwen2 jouet) de MovingRefAdapterCallback (exp46, ancre mobile par
adaptateur `ref` figé). Vérifie les invariants dont dépend la sémantique KL de TRL :
  1. attach : 'ref' ajouté, 'default' reste actif, 'ref' hors gradient, mêmes paramètres entraînables ;
  2. au départ logits(ref) == logits(base) (B=0) ; use_adapter restaure actif + requires_grad ;
  3. après « entraînement » (bruit sur B) logits(default) != logits(ref) ;
  4. ré-ancrage : A/B recopiés, logits(ref) == logits(default) → KL = 0, politique inchangée,
     snapshot cycle<k> ne contient que 'default' ;
  5. merge_adapter/unmerge (sync vLLM de TRL) ne touchent que l'adaptateur actif ;
  6. reprise : initial_cycle=k recharge cycle<k> dans 'ref' ;
  7. save_pretrained complet (checkpoint) écrit 'ref' en sous-dossier.
Lancer : CUDA_VISIBLE_DEVICES= /tmp/envs/agentgym-rl-v2/bin/python python -m src.tests.test_moving_ref
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.train.kl_anchor import MovingRefAdapterCallback  # noqa: E402

torch.manual_seed(0)
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from trl.trainer.utils import use_adapter  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_model():
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    base = Qwen2ForCausalLM(cfg).eval()
    return get_peft_model(base, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, bias="none",
                                           task_type="CAUSAL_LM", target_modules=TARGETS))


def fake_trainer(model):
    return SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=lambda m: m),
                           log=lambda d: None)


def logits(model, x, adapter=None, disable=False):
    with torch.no_grad():
        if disable:
            with model.disable_adapter():
                return model(x).logits.clone()
        with use_adapter(model, adapter):
            return model(x).logits.clone()


def grad_flags(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="moving_ref_selftest_")
    model = tiny_model()
    x = torch.randint(0, 128, (2, 12))
    trainable_before = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    base_logits = logits(model, x, disable=True)

    cb = MovingRefAdapterCallback(every_epochs=4.0, anchors_dir=os.path.join(tmp, "anchors"))
    cb.attach(fake_trainer(model))

    # 1. structure
    assert "ref" in model.peft_config and model.active_adapter == "default"
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".ref." in n)
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == trainable_before
    n_ref = sum(1 for n, _ in model.named_parameters() if ".ref." in n)
    assert n_ref == 2 * 2 * len(TARGETS), n_ref  # A et B × 2 couches × 7 projections

    # 2. au départ ref ≡ base ; use_adapter restaure l'état
    flags = grad_flags(model)
    assert torch.allclose(logits(model, x, "ref"), base_logits, atol=1e-6)
    assert model.active_adapter == "default" and grad_flags(model) == flags

    # 3. « entraînement » : bruit sur B (et A) de default → la politique s'écarte de ref
    with torch.no_grad():
        for s, _ in MovingRefAdapterCallback._lora_pairs(model, "default", "ref"):
            s.add_(0.5 * torch.randn_like(s))
    pol = logits(model, x, "default")
    assert not torch.allclose(pol, logits(model, x, "ref"), atol=1e-4)
    kl_before = torch.nn.functional.kl_div(logits(model, x, "ref").log_softmax(-1), pol.log_softmax(-1),
                                           log_target=True, reduction="batchmean").item()
    assert kl_before > 1e-3, kl_before

    # 4. ré-ancrage à la frontière d'époque
    state = SimpleNamespace(global_step=100, epoch=3.99)
    cb.on_step_end(None, state, None)
    assert cb.cycle == 0, "pas de ré-ancrage avant la frontière"
    state.epoch = 4.0
    cb.on_step_end(None, state, None)
    assert cb.cycle == 1
    for s, d in MovingRefAdapterCallback._lora_pairs(model, "default", "ref"):
        assert torch.equal(s, d)
    assert torch.allclose(logits(model, x, "ref"), pol, atol=1e-6), "KL doit être nulle après copie"
    assert torch.allclose(logits(model, x, "default"), pol, atol=1e-6), "la politique ne doit pas bouger"
    assert grad_flags(model) == flags and model.active_adapter == "default"
    from safetensors.torch import load_file
    cyc = os.path.join(tmp, "anchors", "cycle1", "adapter_model.safetensors")
    keys = list(load_file(cyc).keys())
    assert keys and all(".ref." not in k for k in keys), "le snapshot ne doit contenir que l'adaptateur vivant"
    assert os.path.exists(os.path.join(tmp, "anchors", "chain.jsonl"))

    # 5. merge/unmerge (sync vLLM TRL) : actif seulement, politique restaurée
    with torch.no_grad():
        for s, _ in MovingRefAdapterCallback._lora_pairs(model, "default", "ref"):
            s.add_(0.3 * torch.randn_like(s))   # default ≠ ref à nouveau
    pol2 = logits(model, x, "default")
    model.merge_adapter()
    from peft.tuners.lora.layer import LoraLayer
    merged = {tuple(m.merged_adapters) for m in model.modules() if isinstance(m, LoraLayer)}
    assert merged == {("default",)}, merged
    with torch.no_grad():
        merged_logits = model.base_model.model(x).logits.clone()   # ce que verrait vLLM
    model.unmerge_adapter()
    assert torch.allclose(merged_logits, pol2, atol=1e-5)
    diff = (logits(model, x, "default") - pol2).abs().max().item()
    assert diff < 1e-4, f"merge/unmerge round trip : écart {diff:.2e} (bruit fp32 attendu ~1e-6)"
    assert grad_flags(model) == flags

    # 6. reprise : un modèle neuf + initial_cycle=1 recharge cycle1 dans ref
    model2 = tiny_model()
    cb2 = MovingRefAdapterCallback(every_epochs=4.0, anchors_dir=os.path.join(tmp, "anchors"), initial_cycle=1)
    cb2.attach(fake_trainer(model2))
    ref1 = dict(load_file(cyc))
    for n, p in model2.named_parameters():
        if ".ref." in n:
            k = n.replace(".ref.", ".")
            assert k in ref1 and torch.equal(p, ref1[k]), n
    assert model2.active_adapter == "default"

    # 7. checkpoint complet : 'ref' en sous-dossier, 'default' à la racine
    ck = os.path.join(tmp, "ckpt"); model.save_pretrained(ck)
    assert os.path.exists(os.path.join(ck, "adapter_model.safetensors"))
    assert os.path.exists(os.path.join(ck, "ref", "adapter_model.safetensors"))

    # 8. --moving-ref-reset-adam : purge Adam de 'default' seulement, adaptateur intact.
    #    (Les étapes 4-6 tournent sans optimizer sur le faux trainer : le mode par défaut n'y touche pas.)
    model3 = tiny_model()
    tr3 = fake_trainer(model3)
    tr3.optimizer = torch.optim.AdamW([p for p in model3.parameters() if p.requires_grad], lr=1e-2)
    cb3 = MovingRefAdapterCallback(every_epochs=4.0, anchors_dir=os.path.join(tmp, "anchors3"),
                                   reset_adam=True)
    cb3.attach(tr3)
    model3(x).logits.float().pow(2).mean().backward()
    tr3.optimizer.step()                                     # moments Adam non nuls, B ≠ 0
    live_params = [p for n, p in model3.named_parameters() if "lora_" in n and ".default." in n]
    assert all(p in tr3.optimizer.state for p in live_params)
    pol3 = logits(model3, x, "default")
    cb3.on_step_end(None, SimpleNamespace(global_step=10, epoch=4.0), None)
    assert all(p not in tr3.optimizer.state for p in live_params), "moments Adam non purgés"
    assert torch.allclose(logits(model3, x, "default"), pol3, atol=1e-6), "l'adaptateur ne doit pas bouger"
    assert torch.allclose(logits(model3, x, "ref"), pol3, atol=1e-6), "ref = politique après copie"
    import json
    last = json.loads(open(os.path.join(tmp, "anchors3", "chain.jsonl")).read().splitlines()[-1])
    assert last["adam_purged"] == len(live_params), last

    print(f"[selftest_moving_ref] OK — {n_ref} tenseurs ref, KL avant copie {kl_before:.4f}, "
          f"reset Adam : {last['adam_purged']} états purgés, politique inchangée ; "
          f"nulle après ; merge vLLM = default seul (écart aller-retour {diff:.1e}) ; reprise cycle1 OK ; ckpt avec sous-dossier ref "
          f"({tmp})")


if __name__ == "__main__":
    main()
