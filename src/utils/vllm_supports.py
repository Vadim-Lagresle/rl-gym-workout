"""Tells whether the installed vLLM can serve a given model.

In plain words: reads the model's architecture from its config and checks it against
vLLM's registry, so that the evaluation can fall back to HuggingFace generation when
vLLM cannot serve it.

Notes (FR) — Teste si la vLLM installée peut servir un modèle donné (sans télécharger les poids).

Lit les `architectures` du config.json du modèle et les compare au registre des
architectures supportées par vLLM. Sert au lanceur d'éval pour choisir entre la
stack vLLM (rapide, KV cache) et le fallback HuggingFace generate().

Usage:
    python src/utils/vllm_supports.py <model_path_or_hub_id>
Sortie:
    affiche "yes"/"no" ; code retour 0 si supporté, 1 sinon (et 2 si erreur).

Exemples:
    python src/utils/vllm_supports.py models/Qwen2.5-3B-Instruct   -> yes (Qwen2ForCausalLM)
    python src/utils/vllm_supports.py Qwen/Qwen3.5-4B              -> no  (Qwen3_5ForConditionalGeneration)
"""

from __future__ import annotations

import sys


def vllm_supports(model_id: str) -> bool:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])
    if not archs:
        return False
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    return any(a in supported for a in archs)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python src/utils/vllm_supports.py <model_path_or_hub_id>", file=sys.stderr)
        return 2
    try:
        ok = vllm_supports(sys.argv[1])
    except Exception as e:  # config introuvable, réseau, etc.
        print(f"no  (erreur: {e!r})", file=sys.stderr)
        return 2
    print("yes" if ok else "no")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
