"""Merges a LoRA adapter into its base model to get a complete model.

In plain words: the merged model is a standard HuggingFace checkpoint that vLLM can
serve for evaluation, or that can start a new run with --model-path.

Notes (FR) — Fusionne un adapter LoRA dans son modèle de base → modèle HF complet.

Version paramétrée des scripts jetables merge_exp10p{5,7,8}.py (archivés dans
archive/scripts/merge_lora_originaux/) qui ont produit la lignée warm-start
exp10.x : 35% → 51% → 53% → 58%(pic)/54(re-éval).

Le modèle fusionné est un checkpoint HF standard, servable par vLLM
(start_vllm_server.sh) ou chargeable comme --model-path d'un run suivant.

Usage (env agentgym-rl) :
    python src/utils/merge_lora.py \
        --base models/qwen25_3b_exp10p7_step92_53pct \
        --adapter saves/trl_grpo/exp10.8_warmstart53_lr_div3_best \
        --out models/qwen25_3b_exp10p8_step368_58pct
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True,
                        help="Dossier du modèle de base (HF complet)")
    parser.add_argument("--adapter", required=True,
                        help="Dossier de l'adapter PEFT (adapter_model.safetensors)")
    parser.add_argument("--out", required=True,
                        help="Dossier de sortie du modèle fusionné")
    args = parser.parse_args()

    print(f"[merge] base    : {args.base}", flush=True)
    print(f"[merge] adapter : {args.adapter}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merge_and_unload...", flush=True)
    model = model.merge_and_unload()
    print(f"[merge] écriture → {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("[merge] DONE", flush=True)


if __name__ == "__main__":
    main()
