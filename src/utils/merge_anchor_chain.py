"""Reconstruit un modèle complet depuis une chaîne d'ancres mobiles (exp25).

Avec --moving-anchor-every-epochs (schedules.MovingAnchorCallback), le modèle
final n'est PAS « base + adapter » : à chaque ré-ancrage, l'adapter du cycle a
été fusionné dans la base puis réinitialisé. Le modèle à un instant t est donc :

    base ⊕ merge(cycle1) ⊕ merge(cycle2) ⊕ … ⊕ merge(cycle_k) ⊕ adapter_t

où les cycle<i> vivent dans saves/trl_grpo/<run>_anchors/ (avec chain.jsonl qui
donne le step de chaque ré-ancrage) et adapter_t est par exemple le best
(saves/trl_grpo/<run>_best, dont .best_info donne le step). Ce script applique
les cycles dont le step est ≤ celui de l'adapter final, puis l'adapter final.

Usage :
    python src/utils/merge_anchor_chain.py \
        --base /tmp/models/Qwen2.5-3B-Instruct \
        --anchors saves/trl_grpo/exp25_r8_anchor4ep_anchors \
        --adapter saves/trl_grpo/exp25_r8_anchor4ep_best \
        --out /tmp/models/exp25_best_merged

Nota : les checkpoints périodiques (/tmp) et le modèle final écrit par
train_grpo en fin de run sont déjà des modèles COMPLETS (save_model_for_vllm
fusionne à partir de la base modifiée en mémoire) — ce script ne sert que pour
les artefacts adapter-only du home (best, cycles) après une purge de /tmp.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="modèle de base (ex. /tmp/models/Qwen2.5-3B-Instruct)")
    ap.add_argument("--anchors", required=True, help="dossier <run>_anchors (cycles + chain.jsonl)")
    ap.add_argument("--adapter", default="", help="adapter final à appliquer après la chaîne "
                                                  "(ex. <run>_best) ; vide = chaîne seule")
    ap.add_argument("--adapter-step", type=int, default=-1,
                    help="step de l'adapter final (défaut : lu dans .best_info) — ne fusionner "
                         "que les cycles de step ≤ celui-ci")
    ap.add_argument("--out", required=True, help="dossier de sortie (modèle HF complet)")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    anchors = Path(args.anchors)
    chain_file = anchors / "chain.jsonl"
    cycles: list[dict] = []
    if chain_file.exists():
        with chain_file.open() as f:
            cycles = [json.loads(line) for line in f if line.strip()]
    cycles.sort(key=lambda c: c["cycle"])

    adapter_step = args.adapter_step
    if args.adapter and adapter_step < 0:
        info = Path(args.adapter) / ".best_info"
        if info.exists():
            m = re.search(r"step=(\d+)", info.read_text())
            if m:
                adapter_step = int(m.group(1))
    if adapter_step >= 0:
        cycles = [c for c in cycles if c["step"] <= adapter_step]

    print(f"[merge-chain] base={args.base}")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    for c in cycles:
        cdir = anchors / f"cycle{c['cycle']}"
        print(f"[merge-chain] ⊕ cycle{c['cycle']} (step {c['step']}, epoch {c['epoch']})")
        model = PeftModel.from_pretrained(model, str(cdir)).merge_and_unload()
    if args.adapter:
        print(f"[merge-chain] ⊕ adapter final {args.adapter} (step {adapter_step})")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(out))
    print(f"[merge-chain] modèle complet écrit : {out} "
          f"({len(cycles)} cycle(s) + {'1 adapter final' if args.adapter else 'chaîne seule'})")


if __name__ == "__main__":
    main()
