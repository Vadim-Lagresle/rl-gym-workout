"""Moteur vLLM in-process : génération des rollouts + synchronisation des poids.

TRL ne sait pas qu'on génère avec vLLM (use_vllm=False dans GRPOConfig) : on
maintient nous-mêmes un moteur vLLM dans le process et on y recharge les poids
TRL à CHAQUE step (comme le hybrid engine de verl) pour rester on-policy.
vLLM 0.9.1 V1 tourne dans un sous-process (SyncMPClient) : pas de copie mémoire
directe possible → save state_dict sur disque, destruction et recréation du moteur.
Overhead ~15 s/step, négligeable devant le rollout de 64+ épisodes.

Tout l'état module (moteur, dossier modèle d'origine, compteurs d'échec) vit ici
et nulle part ailleurs. `LLM_ENGINE is None` = génération HF pure (cas des archis
que vLLM 0.9.1 ne sert pas, ex. Qwen3.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

# Moteur vLLM in-process (None = generation HF via trainer._generate_single_turn).
LLM_ENGINE: Any = None
# Dossier du modèle d'ORIGINE, pour recopier les fichiers tokenizer à chaque sync.
# Ne PAS le lire depuis le moteur : après la 1re sync, le dossier du moteur EST
# /tmp/vllm_weight_sync, et copier les tokenizer sur eux-mêmes lève SameFileError
# — ce qui avortait silencieusement toutes les syncs suivantes (bug qui a figé
# vLLM sur les poids du step 1 pendant tout exp7.2/début 7.3).
BASE_MODEL_DIR: str = ""

_SYNC_STEP_COUNT = 0
_SYNC_CONSECUTIVE_FAILURES = 0   # re-raise après N échecs consécutifs
_SYNC_MAX_FAILURES = 3           # 1 transitoire ok ; 3 = problème structurel → crash propre
_WEIGHT_TMP = "/tmp/vllm_weight_sync"


def init_engine(model_path: str, gpu_memory_utilization: float = 0.17,
                max_model_len: int = 16384) -> None:
    """Charge le moteur vLLM in-process sur le modèle de départ.

    gpu_memory_utilization=0.17 : 0.20 laissait trop peu de marge pendant les
    saves de checkpoint (~8 GiB) sur B200."""
    global LLM_ENGINE, BASE_MODEL_DIR
    from vllm import LLM
    BASE_MODEL_DIR = model_path
    print(f"[vllm] Chargement vLLM in-process depuis {model_path} ...", flush=True)
    LLM_ENGINE = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    print("[vllm] vLLM prêt.", flush=True)


def generate_round(tokenizer: Any, states: list, active: list[int],
                   max_tokens: int) -> tuple[list[list[int]], list[list[float]]]:
    """Génère un tour pour tous les épisodes actifs (batch unique, prefix caching)."""
    from vllm import SamplingParams
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
    outputs = LLM_ENGINE.generate(prompts, sampling_params, use_tqdm=False)
    ids_list, lps_list = [], []
    for out in outputs:
        comp = out.outputs[0]
        ids = list(comp.token_ids)
        lps = [list(lp.values())[0].logprob for lp in (comp.logprobs or [])]
        ids_list.append(ids)
        lps_list.append(lps)
    return ids_list, lps_list


def save_model_for_vllm(trl_model: Any, out_dir: str) -> None:
    """Écrit un checkpoint HF *complet* (chargeable par vLLM) à partir du modèle TRL.

    - Full-ft : `save_pretrained` écrit déjà un modèle complet -> on l'utilise tel quel.
    - LoRA    : `save_pretrained` n'écrirait QUE l'adapter (vLLM ne sait pas le charger
      comme un modèle complet -> ValueError 'base_model'). On fusionne donc l'adapter dans
      les poids de base (merge_adapter), on extrait un state_dict aux clés Qwen2 propres
      (sans préfixe 'base_model.model.' ni matrices 'lora_'), on l'écrit en safetensors avec
      la config du modèle de base, puis on défusionne (unmerge_adapter) pour ne PAS altérer
      l'état d'entraînement. C'est la même logique que TRL `_move_model_to_vllm`."""
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


def sync_trl_to_vllm(trl_model: Any) -> None:
    """Recharge le moteur vLLM avec les poids TRL du step courant (à chaque step).

    Un échec de sync est TRÈS visible dans le log (leçon du SameFileError silencieux) ;
    au 3e échec consécutif on crashe proprement plutôt que de continuer un run
    off-policy sans le savoir."""
    global _SYNC_STEP_COUNT, _SYNC_CONSECUTIVE_FAILURES, LLM_ENGINE

    if LLM_ENGINE is None:  # génération HF pure (archi non servable par vLLM)
        return

    _SYNC_STEP_COUNT += 1
    try:
        import shutil
        from vllm import LLM
        print(f"[vllm_sync] step {_SYNC_STEP_COUNT} — sauvegarde poids TRL vers {_WEIGHT_TMP}", flush=True)
        save_model_for_vllm(trl_model, _WEIGHT_TMP)  # full-ft: save direct ; LoRA: merge->save->unmerge
        # Fichiers tokenizer recopiés depuis le dossier d'ORIGINE (stable entre syncs).
        src_dir = Path(BASE_MODEL_DIR or LLM_ENGINE.llm_engine.model_config.model)
        if src_dir.resolve() != Path(_WEIGHT_TMP).resolve():
            for f in src_dir.glob("tokenizer*"):
                shutil.copy2(f, _WEIGHT_TMP)
        print("[vllm_sync] Rechargement vLLM avec nouveaux poids...", flush=True)

        # Réutilise les paramètres du moteur courant pour la recréation.
        vllm_config = LLM_ENGINE.llm_engine.vllm_config
        gpu_util = vllm_config.cache_config.gpu_memory_utilization
        max_len = vllm_config.model_config.max_model_len
        del LLM_ENGINE
        LLM_ENGINE = None
        torch.cuda.empty_cache()
        LLM_ENGINE = LLM(
            model=_WEIGHT_TMP,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        print(f"[vllm_sync] vLLM rechargé avec les poids du step {_SYNC_STEP_COUNT}.", flush=True)
        _SYNC_CONSECUTIVE_FAILURES = 0  # succès → reset compteur
    except Exception as e:
        _SYNC_CONSECUTIVE_FAILURES += 1
        print(
            f"[vllm_sync] !!! SYNC ÉCHOUÉE (échec {_SYNC_CONSECUTIVE_FAILURES}/{_SYNC_MAX_FAILURES}) "
            f"step {_SYNC_STEP_COUNT}: {e!r} — vLLM génère avec des poids FIGÉS.",
            flush=True,
        )
        if _SYNC_CONSECUTIVE_FAILURES >= _SYNC_MAX_FAILURES:
            raise RuntimeError(
                f"[vllm_sync] {_SYNC_MAX_FAILURES} syncs consécutives échouées — "
                f"arrêt pour éviter un run off-policy silencieux."
            ) from e


class VllmSyncCallback(TrainerCallback):
    """Après chaque optimizer step, resynchronise les poids TRL → moteur vLLM."""

    def on_step_end(self, args: Any, state: Any, control: Any,
                    model: Any = None, **kwargs: Any) -> None:
        if model is not None:
            sync_trl_to_vllm(model)
