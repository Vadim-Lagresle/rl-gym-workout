"""Entraînement GRPO multi-tour sur TextCraft avec TRL.

Implémente une boucle RL interactive : à chaque step GRPO, le modèle joue N
épisodes complets (generate → env.step → observe → repeat) via textcraft_rollout_func,
reçoit une récompense sparse 0/1 de l'environnement, et met à jour ses poids par GRPO.

Pré-requis :
  - Serveur TextCraft lancé : conda activate agentenv-textcraft &&
    textcraft --host 127.0.0.1 --port 36005
  - Env conda agentgym-rl (B200, vLLM 0.9.1 fonctionnel — voir docs/hebdo/5juin/session_2026-06-01.md)

Usage (commande de référence, voir CLAUDE.md ; --help pour tous les arguments) :
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_grpo.py --full-ft --num-generations 8 \
        --max-completion-length 512 --max-items 0 --max-steps 200 \
        --use-vllm-inprocess --run-name <run_name>

Nota : --use-vllm-inprocess (moteur vLLM maison dans le process, sync de poids
manuelle à chaque step) est le chemin actif ; --use-vllm (intégration vLLM native
de TRL) n'est pas utilisé en pratique.
"""

#Dependecies :
from __future__ import annotations # for Python 3.10+ type hinting, ie more flexible 

import argparse # parsing des arguments de ligne de commande
from concurrent.futures import ThreadPoolExecutor # pour exécuter les appels d'environnement en parallèle (IO-bound)
import json # pour charger les données à partir de fichiers JSON
import os
import re
import time
from pathlib import Path # pour la manipulation de chemins de fichiers
from typing import Any

import torch

import requests # HTTP client for env interaction 
from datasets import Dataset # transforms jsons into datasets for trl
from transformers import AutoTokenizer # tokenizer of Qwen 2.5 3B
from peft import LoraConfig # Tester la différence entre full-ft et LoRA
from trl import GRPOConfig, GRPOTrainer # on utilise notre propre rollout func multi tour, pas celle du trainer

from agentenv.envs import TextCraftEnvClient



# Paths
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"
TRAIN_PATH = REPO_ROOT / "data" / "train" / "textcraft_train.json"
ENV_SERVER_URL = "http://127.0.0.1:36005"


# In-process vLLM engine (None = use HF _generate_single_turn).
# Set in main() when --use-vllm-inprocess is passed.
VLLM_LLM: Any = None
# Original model dir, used by _sync_trl_to_vllm to copy tokenizer files.
# Do NOT read it from the live engine: after the first sync the engine's model
# dir IS /tmp/vllm_weight_sync, and copying tokenizer files onto themselves
# raises SameFileError — silently aborting every subsequent sync (bug that
# froze vLLM on step-1 weights for the whole exp7.2/early-7.3 runs).
VLLM_BASE_MODEL_DIR: str = ""

# Constants and utils for the interactive rollout and reward shaping.
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_SIM_ROUNDS = 20  # pour le training seulement # paper appendix B.3: 20 turns for plain GRPO TextCraft (the 30 in
                     # textcraft_train.sh is the *final* ScalingInter stage, not the GRPO run;
                     # Figure 7 shows too-large interaction budgets destabilize training)
ITEM_TAG_RE = re.compile(r"^<ITEM_IDX:(\d+)>$")

# ScalingInter curriculum (interaction-budget schedule).
# Two flavours, both written in the CLI as "<max_rounds>:<threshold>" pairs:
#   - step-based  (--max-rounds-schedule)        : threshold = global step
#   - epoch-based (--max-rounds-schedule-epochs) : threshold = trainer epoch
# Internally we store sorted (threshold, max_rounds) tuples. None = fixed
# MAX_SIM_ROUNDS cap. Only ONE flavour is active at a time.
# Example (epochs): "6:0,11:2,16:4,21:6,25:8" -> 6 rounds from epoch 0,
#                   11 from epoch 2, 16 from epoch 4, 21 from epoch 6, 25 from epoch 8.
MAX_ROUNDS_SCHEDULE: list[tuple[int, int]] | None = None
MAX_ROUNDS_SCHEDULE_EPOCHS: list[tuple[float, int]] | None = None


def _parse_rounds_schedule(spec: str, thr_cast):
    """Parse '6:0,11:2,...' into [(threshold, max_rounds), ...] sorted by threshold.

    The CLI order is "<max_rounds>:<threshold>" (max_rounds FIRST, threshold
    SECOND). thr_cast is `int` for step thresholds, `float` for epoch thresholds."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for p in parts:
        rounds_str, thr_str = p.split(":")
        out.append((thr_cast(thr_str), int(rounds_str)))
    out.sort(key=lambda x: x[0])
    return out


def parse_max_rounds_schedule(spec: str) -> list[tuple[int, int]]:
    """Step-based ScalingInter schedule (thresholds = global steps)."""
    return _parse_rounds_schedule(spec, int)


def parse_max_rounds_schedule_epochs(spec: str) -> list[tuple[float, int]]:
    """Epoch-based ScalingInter schedule (thresholds = fractional epochs)."""
    return _parse_rounds_schedule(spec, float)


def current_max_rounds(trainer: Any) -> int:
    """Resolve the active max_rounds cap from the active ScalingInter schedule.

    Epoch-based schedule takes priority over step-based; if neither is set,
    fall back to the fixed MAX_SIM_ROUNDS cap."""
    state = getattr(trainer, "state", None)
    if MAX_ROUNDS_SCHEDULE_EPOCHS is not None:
        epoch = float(getattr(state, "epoch", 0.0) or 0.0) if state is not None else 0.0
        cap = MAX_SIM_ROUNDS
        for thr_epoch, thr_rounds in MAX_ROUNDS_SCHEDULE_EPOCHS:
            if epoch >= thr_epoch:
                cap = thr_rounds
        return cap
    if MAX_ROUNDS_SCHEDULE is not None:
        step = int(getattr(state, "global_step", 0) or 0) if state is not None else 0
        cap = MAX_SIM_ROUNDS
        for thr_step, thr_rounds in MAX_ROUNDS_SCHEDULE:
            if step >= thr_step:
                cap = thr_rounds
        return cap
    return MAX_SIM_ROUNDS


def item_id_to_idx(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def check_server() -> None:
    """Check that the TextCraft server is running and reachable before any model interaction."""
    r = requests.post(f"{ENV_SERVER_URL}/create", json={}, timeout=8)
    r.raise_for_status()


def build_prompt_rows(max_items: int, max_depth: int = 0, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                      depth_in: set[int] | None = None) -> list[dict[str, Any]]:

    """Build the list of prompt rows for rollout_func, optionally filtering by depth and max_items.

    Deux modes de filtrage par depth (exclusifs) :
      - max_depth > 0   : garde les items de depth <= max_depth (curriculum cumulatif).
      - depth_in donné  : garde uniquement les items dont depth ∈ depth_in (curriculum par
                          stage, ex. {1} ou {3, 4}). Utilisé par run_curriculum_staged.sh."""

    with TRAIN_PATH.open() as f:
        rows = json.load(f)

    if max_depth > 0 or depth_in:

        """Filtre par depth. Requiert le mapping pré-calculé
        data/train/textcraft_train_with_depth.json (src/utils/label_depths.py)."""

        depth_file = REPO_ROOT / "data" / "train" / "textcraft_train_with_depth.json"
        if not depth_file.exists():
            raise FileNotFoundError(
                f"--max-depth/--depth-exact requires {depth_file}. "
                "Generate it with: <agentenv-textcraft python> src/utils/label_depths.py"
            )
        with depth_file.open() as f:
            depth_map: dict[str, int] = json.load(f)
        if depth_in:
            rows = [r for r in rows if depth_map.get(r["item_id"], 99) in depth_in]
            print(f"[curriculum] depth_exact={sorted(depth_in)} → {len(rows)} items retained.", flush=True)
        else:
            rows = [r for r in rows if depth_map.get(r["item_id"], 99) <= max_depth]
            print(f"[curriculum] max_depth={max_depth} → {len(rows)} items retained.", flush=True)

    if max_items > 0:
        rows = rows[:max_items]

    """ Récupérer l'amorcre de conversation (human + ack) depuis le serveur TextCraft pour construire les prompts.
    La première réponse du llm est précodée dans conversation_start[1]"""

    probe = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=len(rows), timeout=60)
    manual_human = probe.conversation_start[0]["value"]
    manual_ack = probe.conversation_start[1]["value"]

    """ Assemblage des prompts """
    out: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        idx = item_id_to_idx(item_id)
        # Hidden marker only for rollout_func bookkeeping (removed before generation).
        # If system_prompt is empty (e.g. Gemma-3 which rejects system role), inject
        # it at the start of the first user message instead.
        if system_prompt:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": manual_human},
                {"role": "assistant", "content": manual_ack},
                {"role": "user", "content": f"<ITEM_IDX:{idx}>"}, # hidden marker for rollout_func
            ]
        else:
            prompt = [
                {"role": "user", "content": manual_human},
                {"role": "assistant", "content": manual_ack},
                {"role": "user", "content": f"<ITEM_IDX:{idx}>"}, # hidden marker for rollout_func
            ]
        out.append({"prompt": prompt, "item_id": item_id, "item_idx": idx})

    try:
        probe.close()
    except Exception:
        pass
    return out


def completion_to_text(completion: Any) -> str:

    """ Old, check if called somewhere, if not delete it.
    Extract the assistant's text from a vLLM completion object, which may be a string or a list of dicts with 'content' keys. """

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(completion)


def count_actions(text: str) -> int:
    return len(re.findall(r"Action:\s*(.*?)(?=\n|$)", text, flags=re.DOTALL))


# vLLM weight sync (TRL → vLLM) for in-process vLLM engine. trl ne sait pas qu'on utilise vllm, donc on doit faire un 
# sync manuel des poids à chaque step pour ne pas être off policy. 

_VLLM_SYNC_STEP_COUNT = 0
_VLLM_SYNC_CONSECUTIVE_FAILURES = 0   # re-raise after this many consecutive failures
_VLLM_SYNC_MAX_FAILURES = 3            # (1 transitoire ok ; 3 = problème structurel → crash propre)
_VLLM_SYNC_EVERY = 1    # reload vLLM weights EVERY step, like verl's hybrid engine.
                         # With =5, rollouts at steps 1-4 after a sync were generated by a policy
                         # stale by up to 4 updates — uncorrected off-policy (TRL's vLLM importance
                         # sampling correction is inactive since use_vllm=False in GRPOConfig).
                         # Overhead ~15s/reload, small vs the 64-episode rollout per step.
_VLLM_WEIGHT_TMP = "/tmp/vllm_weight_sync"


def _save_model_for_vllm(trl_model: Any, out_dir: str) -> None:
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


def _sync_trl_to_vllm(trl_model: Any) -> None:
    """Periodic weight sync: every N steps, save TRL weights and reload vLLM model.

    vLLM 0.9.1 V1 runs in a separate subprocess (SyncMPClient), so direct memory
    copy is not possible. Instead, we save TRL's state_dict every _VLLM_SYNC_EVERY
    steps, delete the vLLM engine, and recreate it with the new weights.
    Overhead: ~15s reload / (50 steps × 40s) = ~0.7%.
    """
    global _VLLM_SYNC_STEP_COUNT, _VLLM_SYNC_CONSECUTIVE_FAILURES, VLLM_LLM # globales parce que réassignées

    if VLLM_LLM is None: #cas hf pur sans vllm car incompatibilité (qwen 3.5 par exemple)
        return

    _VLLM_SYNC_STEP_COUNT += 1
    if _VLLM_SYNC_STEP_COUNT % _VLLM_SYNC_EVERY != 0: #Deprecated, on ne fait plus de sync tous les N steps, on le fait à chaque step pour être sûr d'être on policy. 
                                                      #Cette condition est donc toujours fausse, mais on la garde pour le cas où on voudrait revenir à un sync tous les N steps.
        return

    #Bien que la fix a permis de corriger le bug du samefile error, ce try except est une source d'erreur silencieuse 
    try:
        import shutil
        from vllm import LLM
        print(f"[vllm_sync] step {_VLLM_SYNC_STEP_COUNT} — sauvegarde poids TRL vers {_VLLM_WEIGHT_TMP}", flush=True)
        _save_model_for_vllm(trl_model, _VLLM_WEIGHT_TMP)  # full-ft: save direct ; LoRA: merge->save->unmerge
        # Reload tokenizer files from the ORIGINAL model dir (stable across syncs).
        src_dir = Path(VLLM_BASE_MODEL_DIR or VLLM_LLM.llm_engine.model_config.model) #Zone de bug du samefile error
        if src_dir.resolve() != Path(_VLLM_WEIGHT_TMP).resolve():
            for f in src_dir.glob("tokenizer*"):
                shutil.copy2(f, _VLLM_WEIGHT_TMP)
        print("[vllm_sync] Rechargement vLLM avec nouveaux poids...", flush=True)

        vllm_config = VLLM_LLM.llm_engine.vllm_config # récupère la config du moteur vLLM actuel pour réutiliser les mêmes paramètres lors de la recréation du moteur
        gpu_util = vllm_config.cache_config.gpu_memory_utilization
        max_len = vllm_config.model_config.max_model_len
        del VLLM_LLM
        torch.cuda.empty_cache()
        VLLM_LLM = LLM(
            model=_VLLM_WEIGHT_TMP,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        print(f"[vllm_sync] vLLM rechargé avec les poids du step {_VLLM_SYNC_STEP_COUNT}.", flush=True)
        _VLLM_SYNC_CONSECUTIVE_FAILURES = 0  # succès → reset compteur
    except Exception as e:
        _VLLM_SYNC_CONSECUTIVE_FAILURES += 1
        print(
            f"[vllm_sync] !!! SYNC ÉCHOUÉE (échec {_VLLM_SYNC_CONSECUTIVE_FAILURES}/{_VLLM_SYNC_MAX_FAILURES}) "
            f"step {_VLLM_SYNC_STEP_COUNT}: {e!r} — vLLM génère avec des poids FIGÉS.",
            flush=True,
        )
        if _VLLM_SYNC_CONSECUTIVE_FAILURES >= _VLLM_SYNC_MAX_FAILURES:
            raise RuntimeError(
                f"[vllm_sync] {_VLLM_SYNC_MAX_FAILURES} syncs consécutives échouées — "
                f"arrêt pour éviter un run off-policy silencieux."
            ) from e


def _vllm_generate_round(tokenizer: Any, states: list, active: list[int],
                          max_tokens: int) -> tuple[list[list[int]], list[list[float]]]:
    """Generate one round for all active episodes via in-process vLLM (batched)."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        logprobs=1,  # top-1 logprob = logprob of chosen token
    )
    prompts = [
        tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
        for i in active
    ]
    outputs = VLLM_LLM.generate(prompts, sampling_params, use_tqdm=False)
    ids_list, lps_list = [], []
    for out in outputs:
        comp = out.outputs[0]
        ids = list(comp.token_ids)
        lps = [list(lp.values())[0].logprob for lp in (comp.logprobs or [])]
        ids_list.append(ids)
        lps_list.append(lps)
    return ids_list, lps_list


class VLLMWeightSyncCallback: #deprecated
    """TRL TrainerCallback-compatible: syncs TRL model weights to vLLM after each step."""
    def on_step_end(self, args: Any, state: Any, control: Any,
                    model: Any = None, **kwargs: Any) -> None:
        if model is not None:
            _sync_trl_to_vllm(model)


EVAL_DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
EVAL_DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"
EVAL_MAX_ROUNDS = 30  # protocole d'eval (eval_vllm.py), distinct de MAX_SIM_ROUNDS=20 (training)


def run_test_eval(tokenizer: Any, max_items: int = 0,
                  temperature: float = 1.0, max_tokens: int = 512) -> dict[str, float]:
    """Pass@1 sur le test set via le moteur vLLM in-process (poids du step courant).

    Réplique le protocole d'eval_vllm.py — même dataset, même bootstrap de messages,
    cap 30 tours, temperature 1.0 — pour que les scores soient comparables aux evals
    offline. Pure inférence : ces épisodes ne passent jamais par rollout_func ni par
    la loss, le test set n'est donc jamais appris.
    """
    from vllm import SamplingParams
    from collections import Counter
    from statistics import pstdev

    # Source UNIQUE de la taxonomie d'erreurs : on importe classify_error d'analyze_eval
    # (src/analysis) plutôt que de dupliquer les patterns (éviter toute dérive).
    import sys
    _analysis_dir = str(REPO_ROOT / "src" / "analysis")
    if _analysis_dir not in sys.path:
        sys.path.insert(0, _analysis_dir)
    from analyze_eval import classify_error, ERROR_PATTERNS
    etypes = [e[0] for e in ERROR_PATTERNS]

    with EVAL_DATASET_PATH.open() as f:
        items = json.load(f)
    if max_items > 0:
        items = items[:max_items]
    depth_map: dict[str, int] = {}
    if EVAL_DEPTH_PATH.exists():
        with EVAL_DEPTH_PATH.open() as f:
            depth_map = json.load(f)

    n = len(items)
    states: list[list[dict[str, str]]] = []
    clients: list[TextCraftEnvClient] = []
    done = [False] * n
    rewards = [0.0] * n
    rounds_used = [0] * n
    err_counts: list[Counter] = [Counter() for _ in range(n)]  # erreurs par type, par épisode

    for it in items:
        client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        client.reset(item_id_to_idx(it["item_id"]))
        states.append([
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": client.conversation_start[0]["value"]},
            {"role": "assistant", "content": client.conversation_start[1]["value"]},
            {"role": "user", "content": client.observe()},
        ])
        clients.append(client)

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    with ThreadPoolExecutor(max_workers=min(n, 32)) as executor:
        for round_idx in range(EVAL_MAX_ROUNDS):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break
            prompts = [
                tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
                for i in active
            ]
            outputs = VLLM_LLM.generate(prompts, sampling_params, use_tqdm=False)
            texts = [out.outputs[0].text for out in outputs]

            def env_step(pair: tuple[int, int]) -> Any:
                j, i = pair
                return clients[i].step(texts[j])

            step_outs = list(executor.map(env_step, enumerate(active)))
            for (j, i), out in zip(enumerate(active), step_outs):
                states[i].append({"role": "assistant", "content": texts[j]})
                states[i].append({"role": "user", "content": out.state})
                rewards[i] = float(out.reward)
                rounds_used[i] = round_idx + 1
                lab = classify_error(out.state)  # classe l'observation de l'env
                if lab:
                    err_counts[i][lab] += 1
                if bool(out.done):
                    done[i] = True

    solved = sum(1 for r in rewards if r >= 1.0)
    n_err = [sum(err_counts[i].values()) for i in range(n)]
    depths = [depth_map.get(items[i]["item_id"]) for i in range(n)]

    metrics: dict[str, float] = {
        "pass_at_1": solved / n,
        "solved": float(solved),
        "n_items": float(n),
        "mean_rounds": sum(rounds_used) / n,
        "errors_per_ep": sum(n_err) / n,
    }
    # Nb moyen de chaque type d'erreur par épisode -> eval/err_<type>_per_ep
    for et in etypes:
        metrics[f"err_{et}_per_ep"] = sum(err_counts[i].get(et, 0) for i in range(n)) / n
    # Par depth : pass@1, tours moy/std, erreurs moy -> eval/<metric>_d<depth>
    for d in (1, 2, 3, 4):
        idx = [i for i in range(n) if depths[i] == d]
        if not idx:
            continue
        rr = [rounds_used[i] for i in idx]
        metrics[f"pass1_d{d}"] = sum(1 for i in idx if rewards[i] >= 1.0) / len(idx)
        metrics[f"rounds_mean_d{d}"] = sum(rr) / len(idx)
        metrics[f"rounds_std_d{d}"] = pstdev(rr) if len(rr) > 1 else 0.0
        metrics[f"errors_d{d}"] = sum(n_err[i] for i in idx) / len(idx)
    return metrics


def textcraft_rollout_func(prompts: list[list[dict[str, str]]], trainer: GRPOTrainer) -> dict[str, Any]:
    """True interactive rollout: generate -> env.step -> observation -> repeat."""
    tokenizer = trainer.processing_class

    # Chat-template marker tokens (Qwen format), mirroring verl's RolloutHandler
    # (external/AgentGym-RL/verl/workers/rollout/schemas.py, format_config["qwen"]).
    # They are interleaved into completion_ids so that the training-time context is
    # token-identical to the generation-time context. Without them, the logprobs of
    # every action token from round 2 onward are computed in a malformed context.
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assistant_prefix_ids = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)

    n = len(prompts)
    states = [list(p) for p in prompts]
    done = [False] * n
    final_rewards = [0.0] * n
    invalid_counts = [0] * n
    # Per-episode counters for syntax shaping at the message level (vs full
    # concatenated completion). See WORKLOG §1 "Notes méthodologiques (2026-05-11)".
    actions_per_turn_sum = [0] * n
    n_active_turns = [0] * n
    env_clients: list[TextCraftEnvClient] = []

    # Bootstrap one env per sample with the item idx encoded in the last user turn.
    for i in range(n):
        marker = str(states[i][-1]["content"]).strip()
        m = ITEM_TAG_RE.match(marker)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}] last user message: {marker!r}")
        idx = int(m.group(1))

        env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(idx)
        obs = env.observe()
        states[i][-1]["content"] = obs  # replace marker by initial observation
        env_clients.append(env)

    # Outputs expected by TRL rollout_func contract.
    prompt_ids_out: list[list[int]] = [[] for _ in range(n)]
    completion_ids_out: list[list[int]] = [[] for _ in range(n)]
    logprobs_out: list[list[float]] = [[] for _ in range(n)]
    # env_mask: 1 = action token (model output, gradient active)
    #           0 = observation/template token (gradient masked)
    # Passed to TRL as "env_mask"; TRL multiplies completion_mask by it before loss.
    # completion_ids contains the FULL interleaved stream — actions, observations AND
    # chat-template markers (<|im_start|>user, <|im_end|>, generation prompts) — so the
    # backward pass sees a context token-identical to generation (verl RolloutHandler).
    env_mask_out: list[list[int]] = [[] for _ in range(n)]

    cap = current_max_rounds(trainer)
    if MAX_ROUNDS_SCHEDULE_EPOCHS is not None:
        _st = getattr(trainer, "state", None)
        ep = float(getattr(_st, "epoch", 0.0) or 0.0)
        step = int(getattr(_st, "global_step", 0) or 0)
        print(f"[scaling-inter] step={step} epoch={ep:.2f} max_rounds={cap}", flush=True)
    elif MAX_ROUNDS_SCHEDULE is not None:
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        print(f"[scaling-inter] step={step} max_rounds={cap}", flush=True)

    with ThreadPoolExecutor(max_workers=n) as executor:
        for round_idx in range(cap):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            round_prompt_ids: list[list[int]] = []
            for i in active:
                rendered = tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                round_prompt_ids.append(prompt_ids)
                if round_idx == 0:
                    prompt_ids_out[i] = prompt_ids

            if VLLM_LLM is not None:
                # vLLM in-process: batch all active episodes in one call (prefix caching).
                round_completion_ids, round_logprobs = _vllm_generate_round(
                    tokenizer, states, active, max_tokens=trainer.args.max_completion_length
                )
            else:
                # HF _generate_single_turn: re-encodes full context each round.
                round_completion_ids, round_logprobs = trainer._generate_single_turn(
                    round_prompt_ids, images=None, multimodal_fields={}
                )

            # Phase 1 — decode completions (CPU, sequential).
            actions: dict[int, str] = {}
            for j, i in enumerate(active):
                ids = round_completion_ids[j]
                lps = round_logprobs[j] if round_logprobs is not None else [0.0] * len(ids)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                states[i].append({"role": "assistant", "content": text})
                actions_per_turn_sum[i] += count_actions(text)
                n_active_turns[i] += 1
                # Full assistant message: the env client extracts the action itself and
                # returns an error obs on multi-action replies (paper behavior — verl
                # sends the raw decoded content, see vllm_rollout.py agent_step).
                actions[i] = text
                if round_idx > 0:
                    # Generation prompt tokens (<|im_start|>assistant\n) were part of the
                    # generation context but absent from completion_ids → add them, masked.
                    # (Round 0's generation prompt already ends prompt_ids_out.)
                    completion_ids_out[i].extend(assistant_prefix_ids)
                    logprobs_out[i].extend([0.0] * len(assistant_prefix_ids))
                    env_mask_out[i].extend([0] * len(assistant_prefix_ids))
                completion_ids_out[i].extend(ids)
                logprobs_out[i].extend(lps[: len(ids)])
                env_mask_out[i].extend([1] * len(ids))  # action tokens → gradient active
                if not ids or ids[-1] != tokenizer.eos_token_id:
                    # Generation truncated by max_tokens: close the assistant turn anyway,
                    # like verl's add_assistant_message suffix (loss_mask=1 on <|im_end|>).
                    completion_ids_out[i].extend(im_end_ids)
                    logprobs_out[i].extend([0.0] * len(im_end_ids))
                    env_mask_out[i].extend([1] * len(im_end_ids))

            # Phase 2 — parallel HTTP calls to TextCraft server (IO-bound).
            # All N env.step() calls are submitted simultaneously; wall time ≈ max(latency)
            # instead of N × latency, giving ~N× speedup on the HTTP bottleneck.
            futures = [(executor.submit(env_clients[i].step, actions[i]), i)
                       for i in active]
            step_results = {i: fut.result() for fut, i in futures}

            # Phase 3 — update state with env responses (CPU, sequential).
            for i in active:
                step_out = step_results[i]
                obs = step_out.state
                rew = float(step_out.reward)
                is_done = bool(step_out.done)
                states[i].append({"role": "user", "content": obs})
                final_rewards[i] = max(final_rewards[i], rew)
                low = obs.lower()
                if "could not" in low or "error:" in low or "wrong item format" in low:
                    invalid_counts[i] += 1
                done[i] = is_done
                # Include the observation WITH its chat-template markers so the backward
                # pass sees the exact token stream the model was conditioned on at the
                # next generation ("\n<|im_start|>user\n{obs}<|im_end|>", verl's
                # user_prefix/suffix in schemas.py). env_mask=0 on all of it: no gradient.
                obs_block = f"\n<|im_start|>user\n{obs}<|im_end|>"
                obs_ids = tokenizer.encode(obs_block, add_special_tokens=False)
                completion_ids_out[i].extend(obs_ids)
                logprobs_out[i].extend([0.0] * len(obs_ids))
                env_mask_out[i].extend([0] * len(obs_ids))

    for env in env_clients:
        try:
            env.close()
        except Exception:
            pass

    # Guarantee non-empty completion/logprobs/env_mask arrays.
    for i in range(n):
        if not completion_ids_out[i]:
            completion_ids_out[i] = [tokenizer.eos_token_id]
            logprobs_out[i] = [0.0]
            env_mask_out[i] = [1]
        if not prompt_ids_out[i]:
            prompt_ids_out[i] = [tokenizer.eos_token_id]

    # Aggregate per-message action counts to a per-episode mean (1.0 = ideal).
    # Default to 1.0 when an episode produced 0 active turns (extremely rare; keeps shaping neutral).
    mean_actions_per_turn = [
        (actions_per_turn_sum[i] / n_active_turns[i]) if n_active_turns[i] > 0 else 1.0
        for i in range(n)
    ]

    print(
        f"[rollout] n={n} mean_n_actions_per_turn="
        f"{[round(x, 2) for x in mean_actions_per_turn]} "
        f"n_active_turns={n_active_turns} "
        f"episode_reward={final_rewards} invalid_steps={invalid_counts}",
        flush=True,
    )

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        # env_mask: 1=action token (gradient active), 0=obs/template token (masked).
        "env_mask": env_mask_out,
        # Extra fields forwarded to reward function through reward_kwargs
        "episode_reward": final_rewards,
        "invalid_steps": invalid_counts,
        "mean_n_actions_per_turn": mean_actions_per_turn,
    }


def textcraft_reward(
    *,
    prompts: list[Any],
    completions: list[Any],
    episode_reward: list[float],
    invalid_steps: list[int],
    mean_n_actions_per_turn: list[float],
    **kwargs: Any,
) -> list[float]:
    """Sparse outcome reward only (matches AgentGym-RL TextCraft training recipe).

    Removes the v2/v3 shaping that empirically degraded Pass@1 (18 -> 14 -> 8 over
    baseline -> v2 -> v3). AgentGym-RL uses only the env 0/1 scalar at episode end,
    KL handled in the loss via use_kl_loss. See worker investigation 2026-05-12.

    invalid_steps and mean_n_actions_per_turn are kept in the signature only so
    rollout_func's return dict matches; they are NOT used to reshape the reward.
    """
    rewards = [float(r) for r in episode_reward]
    debug = " | ".join(
        f"r={r:+.3f} mean_n={mn:.2f} bad={bs}"
        for r, mn, bs in zip(rewards, mean_n_actions_per_turn, invalid_steps)
    )
    print(f"[reward] sparse outcome | {debug}", flush=True)
    return rewards
















def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=float, default=0,
                        help="Nb d'epochs (passes complètes sur le dataset). Si > 0, "
                             "prioritaire sur --max-steps (qui passe alors à -1).")
    parser.add_argument("--num-generations", type=int, default=2,
                        help="Rollouts GRPO par prompt (N). Papier TextCraft: 8.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64,
                        help="Micro-steps avant 1 optimizer step. Trajectoires/step ≈ "
                             "grad_accum ; prompts/step = grad_accum / num_generations. "
                             "Papier TextCraft: 256 (= 32 prompts × N=8).")
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Max tokens per assistant turn (128 = smoke test, 512 = proper run).")
    parser.add_argument("--full-ft", action="store_true", default=False,
                        help="Full fine-tuning (no LoRA). Requires more VRAM — use on B200.")
    parser.add_argument("--beta", type=float, default=0.001,
                        help=(
                            "Coefficient de la penalite KL (ancrage a la reference). "
                            "Defaut 0.001 (aligne verl/papier, OK en full-ft LR=1e-6). "
                            "En LoRA + LR eleve, monter a ~0.01 pour eviter le runaway KL "
                            "qui a fait diverger exp10."
                        ))
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help=(
                            "LR optimizer. Défaut 1e-6 (aligné full-ft / papier). En LoRA, "
                            "le blog Thinking Machines 'LoRA Without Regret' recommande ~10x "
                            "le LR full-ft -> 1e-5 (voire 1.5e-5 pour les runs <100 steps)."
                        ))
    parser.add_argument("--optim", type=str, default="",
                        help=(
                            "Optimizer HF Trainer. Défaut intelligent : 'adamw_bnb_8bit' en "
                            "full-ft (économise ~18 Go sur les 3B params), 'adamw_torch' (fp32, "
                            "non quantisé) en LoRA — les params entraînables sont alors minuscules "
                            "donc l'optimizer fp32 tient sans souci. Override possible."
                        ))
    parser.add_argument("--max-depth", type=int, default=0,
                        help=(
                            "Keep only training items with depth <= max-depth. "
                            "0 = all items. Requires data/train/textcraft_train_with_depth.json "
                            "(generate with src/utils/label_depths.py)."
                        ))
    parser.add_argument("--depth-exact", type=str, default="",
                        help=(
                            "Garde uniquement les items dont depth ∈ liste fournie. "
                            "Ex: '1' (depth 1 seul) ou '3,4' (depth 3 et 4). "
                            "Exclusif avec --max-depth. Utilisé par le curriculum par stage."
                        ))
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to model dir (default: models/Qwen2.5-3B-Instruct).")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt. Pass '' for models without system role (e.g. Gemma-3).")
    parser.add_argument("--run-name", type=str, default="trl_grpo_textcraft_smoke")
    parser.add_argument(
        "--max-rounds-schedule",
        type=str,
        default="",
        help=(
            "ScalingInter curriculum piloté par STEP, format '<rounds>:<step>', "
            "e.g. '5:0,10:13,15:26,20:38' = max_rounds=5 dès step 0, =10 dès step 13, "
            "=15 dès step 26, =20 dès step 38. Empty = cap fixe MAX_SIM_ROUNDS."
        ),
    )
    parser.add_argument(
        "--max-rounds-schedule-epochs",
        type=str,
        default="",
        help=(
            "ScalingInter curriculum piloté par EPOCH (et non step), format "
            "'<rounds>:<epoch>', e.g. '6:0,11:2,16:4,21:6,25:8' = 6 tours dès "
            "l'epoch 0, 11 dès l'epoch 2, ... Exclusif avec --max-rounds-schedule."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Éval Pass@1 sur le test set tous les N steps, logguée dans wandb "
                             "sous eval/pass_at_1 (0 = off ; nécessite --use-vllm-inprocess)")
    parser.add_argument("--eval-items", type=int, default=0,
                        help="Nb d'items du test set pour l'éval périodique (0 = tous)")
    parser.add_argument("--best-init-score", type=float, default=-1.0,
                        help="Seuil initial du save-best (Pass@1, 0..1). Mettre au score du "
                             "checkpoint de reprise (ex. 0.32) pour ne jamais sauver pire que lui. "
                             "Défaut -1 = le 1er eval sauve toujours.")
    parser.add_argument("--save-steps", type=int, default=0,
                        help="Période de sauvegarde des checkpoints (0 = défaut : 50 en full-ft, 5 sinon)")
    parser.add_argument("--save-total-limit", type=int, default=0,
                        help="Nb de checkpoints conservés (0 = défaut : 1 en full-ft, 3 sinon). "
                             "Augmenter pour garder l'historique, ex. un par point d'eval "
                             "(attention au disque : ~5.8 Go par checkpoint)")
    parser.add_argument("--save-best-only", action="store_true", default=False,
                        help="N'écrit JAMAIS le dernier checkpoint : ni les checkpoints HF "
                             "périodiques/de fin (save_strategy='no', donc pas d'optimizer.pt), "
                             "ni le modèle final dans out_dir. Seul le callback d'éval sauve "
                             "<run>_best, et uniquement quand le Pass@1 test s'améliore. "
                             "Évite de saturer le disque et de perdre le best (cf. exp10.3).")
    parser.add_argument("--use-vllm", action="store_true", default=False)
    parser.add_argument("--use-vllm-inprocess", action="store_true", default=False,
                        help="Use in-process vLLM for generation (faster, prefix caching). "
                             "Requires vllm installed. Weights synced to vLLM after each step.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help=(
            "Path to a TRL checkpoint dir (e.g. saves/.../checkpoint-25) "
            "to resume training state from. Empty = fresh run."
        ),
    )
    args = parser.parse_args()

    depth_in: set[int] | None = None
    if args.depth_exact:
        if args.max_depth:
            raise SystemExit("--depth-exact et --max-depth sont exclusifs.")
        depth_in = {int(x) for x in args.depth_exact.split(",") if x.strip()}

    if args.max_rounds_schedule and args.max_rounds_schedule_epochs:
        raise SystemExit("--max-rounds-schedule et --max-rounds-schedule-epochs sont exclusifs.")
    if args.max_rounds_schedule:
        global MAX_ROUNDS_SCHEDULE
        MAX_ROUNDS_SCHEDULE = parse_max_rounds_schedule(args.max_rounds_schedule)
        print(f"[scaling-inter] schedule (step-based) = {MAX_ROUNDS_SCHEDULE}", flush=True)
    if args.max_rounds_schedule_epochs:
        global MAX_ROUNDS_SCHEDULE_EPOCHS
        MAX_ROUNDS_SCHEDULE_EPOCHS = parse_max_rounds_schedule_epochs(args.max_rounds_schedule_epochs)
        print(f"[scaling-inter] schedule (epoch-based) = {MAX_ROUNDS_SCHEDULE_EPOCHS}", flush=True)

    if args.use_vllm_inprocess:
        global VLLM_LLM, VLLM_BASE_MODEL_DIR
        from vllm import LLM
        model_path_for_vllm = str(Path(args.model_path) if args.model_path else DEFAULT_MODEL_PATH)
        if not Path(model_path_for_vllm).is_absolute():
            model_path_for_vllm = str(REPO_ROOT / model_path_for_vllm)
        VLLM_BASE_MODEL_DIR = model_path_for_vllm
        print(f"[vllm] Chargement vLLM in-process depuis {model_path_for_vllm} ...", flush=True)
        VLLM_LLM = LLM(
            model=model_path_for_vllm,
            dtype="bfloat16",
            gpu_memory_utilization=0.17,  # 0.20 laissait trop peu de marge pendant les saves checkpoint (~8 GiB)
            max_model_len=16384,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        print("[vllm] vLLM prêt.", flush=True)

    model_path = Path(args.model_path) if args.model_path else DEFAULT_MODEL_PATH
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    print(f"[train] Model: {model_path}", flush=True)

    check_server()
    prompts_per_step = args.gradient_accumulation_steps // args.num_generations
    if args.gradient_accumulation_steps % args.num_generations != 0:
        raise SystemExit(
            f"gradient_accumulation_steps ({args.gradient_accumulation_steps}) must be "
            f"divisible by num_generations ({args.num_generations})."
        )
    print(
        f"[train] batch GRPO: {args.gradient_accumulation_steps} traj/step, "
        f"{prompts_per_step} prompts/step, N={args.num_generations}",
        flush=True,
    )
    rows = build_prompt_rows(max_items=args.max_items, max_depth=args.max_depth,
                             system_prompt=args.system_prompt, depth_in=depth_in)
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = REPO_ROOT / "saves" / "trl_grpo" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", "rl-gym-workout")
    # Le compte wandb v-lagresle n'a pas d'entity par défaut (CommError sinon) ;
    # l'entity utilisable est l'équipe v-lagresle-criteo (cf. wandb.Api().viewer.teams).
    os.environ.setdefault("WANDB_ENTITY", "v-lagresle-criteo")

    # Smoke test = run minuscule (≤10 steps ET pas de mode epochs) : on coupe wandb
    # et la sauvegarde des checkpoints. Un run en --num-epochs n'est JAMAIS un smoke.
    is_smoke = (args.num_epochs <= 0 and args.max_steps <= 10)

    # Optimizer : défaut intelligent selon full-ft vs LoRA (override possible via --optim).
    optim_choice = args.optim or ("adamw_bnb_8bit" if args.full_ft else "adamw_torch")
    print(f"[train] Optimizer: {optim_choice} ({'LoRA' if not args.full_ft else 'full-ft'})", flush=True)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        # Skip wandb for smoke tests to avoid cluttering the project.
        report_to=[] if is_smoke else ["wandb"],
        # bs=1 (micro-batch) keeps the forward pass at 1×seq×vocab to avoid OOM on B200.
        # 1 optimizer step = grad_accum micro-steps × 1 prompt × num_generations rollouts.
        # → trajectoires/step ≈ grad_accum ; prompts/step = grad_accum / num_generations.
        # Papier AgentGym-RL TextCraft: train_batch_size=32, n=8 → grad_accum=256, N=8.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Optimizer : voir optim_choice calculé plus haut.
        # - full-ft  -> adamw_bnb_8bit (8-bit) : optimizer states en int8 au lieu de fp32,
        #   ~18 Go libérés sur les 3B params.
        # - LoRA     -> adamw_torch (fp32) : seuls les adapters (~quelques M params) sont
        #   entraînés, donc l'optimizer non quantisé reste négligeable en VRAM.
        optim=optim_choice,
        learning_rate=args.learning_rate,
        # Explicit clip (default is also 1.0, but make intent clear after the
        # grad_norm=1765 spike at step 35 of v2 — see WORKLOG step50 anomaly).
        max_grad_norm=1.0,
        # KL regularization: paper uses kl_loss_coef=0.001 (low_var_kl type).
        # TRL beta is equivalent; TRL already uses the same k3 estimator as verl low_var_kl.
        # Configurable: en LoRA + LR eleve, beta=0.001 est trop faible -> runaway KL
        # (cf. effondrement exp10). Remonter beta resserre l'ancrage a la reference.
        beta=args.beta,
        # The paper's script sets ppo_inner_epochs=2 BUT the AgentGym-RL verl fork ignores
        # it: update_policy() does a single pass over the batch (ppo_epochs only appears in
        # the MFU metric, agent_fsdp_workers.py:413). The actual paper run is 1 epoch.
        # With num_iterations=1 and steps_per_generation == grad_accum, TRL skips the
        # old_logprobs forward and trains fully on-policy (ratio ≡ 1).
        num_iterations=1,
        # verl uses a constant LR schedule (warmup_style: constant, ppo_trainer.yaml).
        # HF Trainer default is "linear" decay to 0, which silently halves the average LR.
        lr_scheduler_type="constant",
        # Explicit: "dapo" is the TRL 1.4 default. Its token-level aggregation
        # (sum / total_tokens) matches verl's masked_mean closer than loss_type="grpo"
        # (per-sequence mean), pinned here for reproducibility across TRL versions.
        loss_type="dapo",
        use_vllm=args.use_vllm,
        # --num-epochs > 0 => piloter par epochs (max_steps=-1), sinon par max_steps.
        max_steps=(-1 if args.num_epochs > 0 else args.max_steps),
        num_train_epochs=(args.num_epochs if args.num_epochs > 0 else 3),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        logging_steps=1,
        # Disable saving for smoke tests to avoid wasting 6 GB per run.
        # For real runs: save every ~epoch, keep 1 checkpoint (peak disk = 12 GB during write).
        save_strategy="no" if (is_smoke or args.save_best_only) else "steps",
        save_steps=args.save_steps or (50 if args.full_ft else 5),
        save_total_limit=args.save_total_limit or (1 if args.full_ft else 3),
        save_only_model=args.full_ft,
        eval_strategy="no",
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True,
                           "attn_implementation": "sdpa"},   # flash_attn bloqué (glibc 2.28 < 2.32); sdpa = PyTorch built-in efficient attention
    )

    peft_config = None if args.full_ft else LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.full_ft:
        print("[train] Full fine-tuning (no LoRA).", flush=True)

    callbacks = []

    from transformers import TrainerCallback
    class _MemDiagCB(TrainerCallback):
        """Loggue un breakdown mémoire GPU détaillé toutes les 50 steps.

        Distingue :
          - PyTorch alloué/réservé  → torch.cuda.memory_stats()
          - États bitsandbytes      → itère optimizer.state (stockés hors pool PyTorch)
          - nvidia-smi (process)    → mémoire totale vue par le driver CUDA
        """
        def on_step_end(self, targs, state, control, model=None, optimizer=None, **kwargs):
            if state.global_step % 50 != 0:
                return
            import subprocess, torch
            torch_alloc  = torch.cuda.memory_allocated() / 1024**3
            torch_reserv = torch.cuda.memory_reserved()  / 1024**3

            # États bitsandbytes (int8) — hors pool PyTorch
            bnb_bytes = 0
            if optimizer is not None:
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        s = optimizer.state.get(p, {})
                        for v in s.values():
                            if hasattr(v, "nbytes"):
                                bnb_bytes += v.nbytes
                            elif hasattr(v, "element_size") and hasattr(v, "numel"):
                                bnb_bytes += v.element_size() * v.numel()
            bnb_gib = bnb_bytes / 1024**3

            # Mémoire totale du process vue par le driver (nvidia-smi)
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], text=True)
                import os; pid = os.getpid()
                driver_gib = next(
                    (int(row.split(",")[1]) / 1024
                     for row in out.strip().splitlines()
                     if row.split(",")[0].strip() == str(pid)),
                    None)
            except Exception:
                driver_gib = None

            phantom = (driver_gib - torch_alloc - bnb_gib) if driver_gib else None
            print(
                f"[mem step={state.global_step}] "
                f"torch_alloc={torch_alloc:.1f} GiB  "
                f"torch_reserved={torch_reserv:.1f} GiB  "
                f"bnb_states={bnb_gib:.1f} GiB  "
                f"driver_total={driver_gib:.1f} GiB  "
                f"phantom(driver-torch-bnb)={phantom:.1f} GiB"
                if phantom is not None else
                f"[mem step={state.global_step}] "
                f"torch_alloc={torch_alloc:.1f} GiB  "
                f"bnb_states={bnb_gib:.1f} GiB",
                flush=True,
            )
    callbacks.append(_MemDiagCB())

    if args.use_vllm_inprocess:
        class _VLLMSyncCB(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if model is not None:
                    _sync_trl_to_vllm(model)
        callbacks.append(_VLLMSyncCB())
        print("[vllm] VLLMWeightSyncCallback enregistré.", flush=True)

    test_eval_cb = None
    if args.eval_every > 0:
        if not args.use_vllm_inprocess:
            raise SystemExit("--eval-every nécessite --use-vllm-inprocess (le moteur vLLM sert aussi à l'éval)")

        class _TestEvalCB(TrainerCallback):
            """Éval test set périodique + sauvegarde du MEILLEUR checkpoint.

            Enregistré APRÈS _VLLMSyncCB : à on_step_end, le moteur vLLM contient
            déjà les poids du step courant.

            Quand le Pass@1 test s'améliore, le checkpoint est écrit dans
            saves/trl_grpo/<run>_best — sur le disque home PERSISTANT, pas /tmp
            volatil. En cas de coupure infra (3 en 3 jours, 13-15/06), on reprend
            de ce best au lieu de repartir de ckpt400. Écriture via .tmp + swap
            atomique : l'ancien best est préservé tant que le nouveau n'est pas
            complet (un échec disque ne détruit pas le best existant)."""
            trainer_ref: Any = None  # injecté après la création du GRPOTrainer
            best_score: float = args.best_init_score  # seuil initial (-1 => 1er eval sauve toujours)
            best_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{args.run_name}_best")

            def on_step_end(self, targs, state, control, **kwargs):
                if state.global_step == 0 or state.global_step % args.eval_every != 0:
                    return
                t0 = time.time()
                print(f"[test_eval] step {state.global_step} — éval test set en cours...", flush=True)
                try:
                    metrics = run_test_eval(self.trainer_ref.processing_class, max_items=args.eval_items)
                except Exception as e:
                    # Télémétrie pure : un échec d'éval ne doit pas tuer le run — mais il
                    # doit être impossible à rater dans le log (leçon du sync silencieux).
                    print(f"[test_eval] !!! ÉVAL ÉCHOUÉE au step {state.global_step}: {e!r} — "
                          f"le training continue, la métrique eval/ manquera ce point.", flush=True)
                    return
                metrics["duration_s"] = round(time.time() - t0, 1)
                score = metrics["pass_at_1"]
                print(f"[test_eval] step {state.global_step} — "
                      f"Pass@1 = {int(metrics['solved'])}/{int(metrics['n_items'])} "
                      f"({100 * score:.0f}%) en {metrics['duration_s']}s", flush=True)

                # Sauvegarde du best sur disque persistant si score >= record (>= pour
                # garder le checkpoint le PLUS RÉCENT à score égal -> reprise moins coûteuse).
                if score >= self.best_score:
                    import shutil
                    tmp = self.best_dir + ".tmp"
                    try:
                        shutil.rmtree(tmp, ignore_errors=True)
                        unwrapped = self.trainer_ref.accelerator.unwrap_model(self.trainer_ref.model)
                        # LoRA : on ne sauve QUE l'adapter PEFT (adapter_model.safetensors +
                        # adapter_config.json, ~qq dizaines de Mo) au lieu du modèle fusionné
                        # complet (~5.8 Go) qui saturait le disque (cf. exp10.3). Le best =
                        # modèle de base 35% + cet adapter, à fusionner plus tard (merge_and_unload)
                        # pour l'éval offline. NB : l'éval en cours de run tourne sur le moteur
                        # vLLM in-process (poids déjà synchronisés), PAS sur ce dossier -> sauver
                        # l'adapter seul n'impacte pas run_test_eval.
                        # Full-ft : on garde le modèle complet (_save_model_for_vllm).
                        is_peft = (hasattr(unwrapped, "merge_adapter")
                                   and hasattr(unwrapped, "unmerge_adapter"))
                        if is_peft:
                            unwrapped.save_pretrained(tmp)  # adapter-only
                            print(f"[test_eval] (LoRA) adapter-only save -> {tmp} "
                                  f"(adapter_model.safetensors + adapter_config.json, PAS de merge)",
                                  flush=True)
                        else:
                            _save_model_for_vllm(unwrapped, tmp)  # full-ft: modèle complet
                        self.trainer_ref.processing_class.save_pretrained(tmp)
                        with open(os.path.join(tmp, ".best_info"), "w") as f:
                            f.write(f"step={state.global_step} pass_at_1={score:.4f}\n")
                        shutil.rmtree(self.best_dir, ignore_errors=True)
                        os.replace(tmp, self.best_dir)  # swap atomique (même FS home)
                        print(f"[test_eval] >>> NOUVEAU BEST {100*score:.0f}% "
                              f"(ancien {100*self.best_score:.0f}%) sauvé sur disque persistant: "
                              f"{self.best_dir}", flush=True)
                        self.best_score = score
                    except Exception as e:
                        shutil.rmtree(tmp, ignore_errors=True)
                        print(f"[test_eval] !!! SAUVEGARDE BEST ÉCHOUÉE step {state.global_step}: "
                              f"{e!r} — ancien best conservé, training continue.", flush=True)

                metrics["best_pass_at_1"] = max(score, self.best_score)
                self.trainer_ref.log({f"eval/{k}": v for k, v in metrics.items()})

        test_eval_cb = _TestEvalCB()
        callbacks.append(test_eval_cb)
        print(f"[test_eval] Éval test set tous les {args.eval_every} steps "
              f"({args.eval_items or 'tous les'} items).", flush=True)

    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=textcraft_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=textcraft_rollout_func,
        callbacks=callbacks if callbacks else None,
    )
    if test_eval_cb is not None:
        test_eval_cb.trainer_ref = trainer

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)
    print("[smoke] TRL+GRPO training run finished")

    # Sauvegarde finale garantie (sauf smoke). On matérialise les poids finaux à la racine
    # de out_dir comme un modèle HF COMPLET (pas juste l'adapter en LoRA) pour que l'éval
    # (start_vllm_server.sh + eval_vllm.py) et le chaînage --model-path fonctionnent direct.
    if not is_smoke and not args.save_best_only:
        final_model = trainer.accelerator.unwrap_model(trainer.model)
        _save_model_for_vllm(final_model, str(out_dir))  # full-ft: save direct ; LoRA: merged
        tokenizer.save_pretrained(str(out_dir))
        kind = "merged LoRA" if hasattr(final_model, "merge_adapter") else "full-ft"
        print(f"[train] Final model ({kind}) saved to {out_dir}", flush=True)
    elif args.save_best_only:
        print("[train] --save-best-only : pas de save du dernier modèle ; "
              "seul <run>_best (meilleur Pass@1) est conservé.", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
