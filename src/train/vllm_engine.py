"""Accès au moteur vLLM géré par TRL (mode colocate) + utilitaires de sauvegarde.

Historique (migration 2026-07-22) : jusqu'à TRL 1.4.0 / vLLM 0.9.1 (contrainte
glibc 2.28 de l'ancienne VM), ce module gérait SON PROPRE moteur vLLM in-process :
initialisation manuelle, et surtout synchronisation des poids par écriture du
modèle complet (~5,8 Go) sur /tmp puis destruction/recréation du moteur à chaque
step (~15-20 s, ≈20-25 % du temps de step mesuré sur exp19). TRL ≥ 1.9.0 gère
nativement ce cas : avec GRPOConfig(use_vllm=True, vllm_mode="colocate"), il
construit le moteur dans le process ET synchronise les poids EN MÉMOIRE
(merge_adapter → push des tenseurs dans vLLM → unmerge → reset du prefix cache),
automatiquement avant chaque appel à rollout_func (grpo_trainer.py:2151-2153).

Ce module ne fait donc plus que :
  - exposer le moteur TRL (get_engine) pour la boucle multi-tour et l'éval ;
  - générer un tour (generate_round) — même logique/logprobs qu'avant, le moteur
    TRL en dessous ;
  - forcer une sync avant l'éval périodique (sync_before_eval) : TRL synchronise
    paresseusement au DÉBUT du step suivant, or TestEvalCallback évalue à
    on_step_end — sans sync explicite, l'éval verrait les poids d'avant l'update ;
  - écrire un modèle HF complet servable (save_model_for_vllm), pour la
    sauvegarde finale et le best full-ft (indépendant du moteur).
"""

from __future__ import annotations

from typing import Any

import torch


def get_engine(trainer: Any) -> Any:
    """Le moteur vLLM brut (objet vllm.LLM) géré par TRL en colocate, ou None.

    None = pas de vLLM (use_vllm=False) → generation HF via
    trainer._generate_single_turn (archis non servables, smoke tests CPU-ish)."""
    gen = getattr(trainer, "vllm_generation", None)
    return getattr(gen, "llm", None) if gen is not None else None


def generate_round(trainer: Any, tokenizer: Any, states: list, active: list[int],
                   max_tokens: int) -> tuple[list[list[int]], list[list[float]]]:
    """Génère un tour pour tous les épisodes actifs (batch unique, prefix caching).

    Utilise le moteur TRL directement avec nos propres SamplingParams (logprobs=1
    pour récupérer la logprob du token choisi), plutôt que
    trainer.vllm_generation.generate() dont le format de logprobs (liste par token)
    diffère de notre contrat (un float par token)."""
    from vllm import SamplingParams
    llm = get_engine(trainer)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        logprobs=1,  # top-1 logprob = logprob du token choisi
    )
    prompts = [
        tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
        for i in active
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    ids_list, lps_list = [], []
    for out in outputs:
        comp = out.outputs[0]
        ids = list(comp.token_ids)
        lps = [list(lp.values())[0].logprob for lp in (comp.logprobs or [])]
        ids_list.append(ids)
        lps_list.append(lps)
    return ids_list, lps_list


def sync_before_eval(trainer: Any) -> None:
    """Pousse les poids du step courant dans le moteur vLLM avant une éval à on_step_end.

    TRL ne synchronise que paresseusement (au début du _generate suivant, quand
    global_step a changé). On force la sync ici, puis on marque le step comme déjà
    chargé pour éviter une re-sync inutile au rollout suivant."""
    trainer.vllm_generation.sync_weights()
    trainer._last_loaded_step = trainer.state.global_step


def save_model_for_vllm(trl_model: Any, out_dir: str) -> None:
    """Écrit un checkpoint HF *complet* (chargeable par vLLM) à partir du modèle TRL.

    - Full-ft : `save_pretrained` écrit déjà un modèle complet -> on l'utilise tel quel.
    - LoRA    : `save_pretrained` n'écrirait QUE l'adapter (vLLM ne sait pas le charger
      comme un modèle complet -> ValueError 'base_model'). On fusionne donc l'adapter dans
      les poids de base (merge_adapter), on extrait un state_dict aux clés Qwen2 propres
      (sans préfixe 'base_model.model.' ni matrices 'lora_'), on l'écrit en safetensors avec
      la config du modèle de base, puis on défusionne (unmerge_adapter) pour ne PAS altérer
      l'état d'entraînement."""
    is_peft = hasattr(trl_model, "merge_adapter") and hasattr(trl_model, "unmerge_adapter")
    if not is_peft:
        trl_model.save_pretrained(out_dir)
        return

    import glob
    import os
    from safetensors.torch import save_file

    trl_model.merge_adapter()
    try:
        clean: dict[str, torch.Tensor] = {}
        for k, v in trl_model.state_dict().items():
            if "lora_" in k:  # matrices A/B de l'adapter : déjà fusionnées dans base_layer
                continue
            nk = k.replace("base_model.model.", "").replace(".base_layer.", ".")
            clean[nk] = v.detach().to(torch.bfloat16).contiguous().cpu()
        os.makedirs(out_dir, exist_ok=True)
        # Nettoyer tout fichier de poids périmé : un adapter_model.safetensors résiduel ferait
        # charger à vLLM des clés 'base_model.' (il glob TOUS les *.safetensors du dossier) -> crash.
        for pat in ("adapter_config.json", "adapter_model.safetensors",
                    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"):
            for f in glob.glob(os.path.join(out_dir, pat)):
                os.remove(f)
        base = trl_model.get_base_model()
        base.config.save_pretrained(out_dir)
        if getattr(base, "generation_config", None) is not None:
            try:
                base.generation_config.save_pretrained(out_dir)
            except Exception:
                pass
        save_file(clean, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    finally:
        trl_model.unmerge_adapter()
