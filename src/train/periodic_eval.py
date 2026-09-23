"""Test evaluation during training, and saving of the best model.

In plain words: every N updates, the current model plays the 100 test tasks with
exactly the protocol of the offline evaluation (30 turns, temperature 1.0), and the
pass@1 is logged with its breakdown by depth and by error class. When the score beats
the best so far, the model (the adapter, and optionally its optimizer) is saved to
saves/trl_grpo/<run>_best. These test episodes never reach the loss: the test set is
never learned.

Notes (FR) — Éval Pass@1 sur le test set PENDANT l'entraînement + sauvegarde du best.

`run_test_eval` réplique le protocole d'eval offline (src/eval/eval_textcraft.py) —
même dataset, même bootstrap de messages, cap 30 tours, temperature 1.0 — pour que
les scores soient comparables. Pure inférence via le moteur vLLM colocate de TRL
(poids du step courant, poussés par vllm_engine.sync_before_eval) : ces épisodes ne
passent jamais par rollout_func ni par la loss, le test set n'est donc jamais appris.

`TestEvalCallback` orchestre l'éval périodique et écrit le MEILLEUR checkpoint sur
le disque home persistant (pas /tmp volatil) — leçon des 3 coupures infra de juin.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from statistics import pstdev
from typing import Any

from transformers import TrainerCallback

from agentenv.envs import TextCraftEnvClient

# Source UNIQUE de la taxonomie d'erreurs : classify_error d'analyze_eval
# (src/analysis) plutôt qu'une copie des patterns (éviter toute dérive).
from src.eval.error_taxonomy import ERROR_PATTERNS, classify_error
# Bootstrap de messages partagé avec l'éval offline (règles + ack + obs, few-shot inclus).
from src.eval.textcraft_common import build_initial_messages
from src.train import vllm_engine
from src.train.data import DEFAULT_SYSTEM_PROMPT, ENV_SERVER_URL, REPO_ROOT, item_id_to_idx

EVAL_DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
EVAL_DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"
EVAL_MAX_ROUNDS = 30  # protocole d'éval, distinct de horizon_schedules.MAX_SIM_ROUNDS=20 (training)


def run_test_eval(llm: Any, tokenizer: Any, max_items: int = 0,
                  temperature: float = 1.0, max_tokens: int = 512,
                  fewshot_block: str | None = None,
                  fewshot_messages: list[dict] | None = None) -> dict[str, float]:
    """Pass@1 sur le test set via le moteur vLLM colocate (poids du step courant).

    llm : le moteur vLLM brut (vllm_engine.get_engine(trainer)), synchronisé au
    préalable par vllm_engine.sync_before_eval. Retourne les métriques globales
    + par depth + par type d'erreur (préfixées eval/ par le callback)."""
    from vllm import SamplingParams

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
        states.append(build_initial_messages(
            client, system_prompt=DEFAULT_SYSTEM_PROMPT,
            fewshot_block=fewshot_block,
            fewshot_messages=list(fewshot_messages) if fewshot_messages else None,
        ))
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
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
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


class TestEvalCallback(TrainerCallback):
    """Éval test set périodique + sauvegarde du MEILLEUR checkpoint.

    Synchronise lui-même les poids du step courant dans le moteur vLLM de TRL
    (sync_before_eval) avant d'évaluer — TRL ne synchronise que paresseusement
    au début du step suivant.

    Quand le Pass@1 test s'améliore (>= : à score égal on garde le checkpoint le
    plus RÉCENT → reprise moins coûteuse), le checkpoint est écrit dans
    saves/trl_grpo/<run>_best via .tmp + swap atomique : l'ancien best est
    préservé tant que le nouveau n'est pas complet.
      - LoRA    : adapter PEFT seul (~qq dizaines de Mo — le modèle fusionné de
        5.8 Go saturait le disque, cf. exp10.3) ; best = base + adapter, à fusionner
        via src/utils/merge_lora.py pour l'éval offline.
      - Full-ft : modèle HF complet (save_model_for_vllm), re-servable tel quel.
    """

    def __init__(self, eval_every: int, eval_items: int, best_init_score: float,
                 run_name: str, fewshot_block: str | None = None,
                 fewshot_messages: list[dict] | None = None,
                 eval_fn: Any = None, save_optimizer: bool = False,
                 delete_before_save: bool = False) -> None:
        self.eval_every = eval_every
        self.eval_items = eval_items
        # Mode gros modèle (7B full-FT, exp26) : le home (35 Go) ne peut pas
        # héberger DEUX modèles pendant le swap atomique .tmp → dir (pic 2×15 Go).
        # delete_before_save supprime l'ancien best AVANT d'écrire le nouveau :
        # pic disque = 1 modèle, au prix d'une fenêtre de quelques minutes sans
        # best complet sur le home (risque assumé, documenté runs/14_fullft_7b).
        self.delete_before_save = delete_before_save
        # Sauver aussi optimizer.pt avec le best (reprise à moments Adam cohérents,
        # leçon exp22.1). Full-FT 3B 8-bit : ~6.2 Go — l'ordre des écritures dans
        # on_step_end est pensé pour plafonner le pic disque (voir commentaires).
        self.save_optimizer = save_optimizer
        # Fonction d'éval interchangeable (même signature que run_test_eval) :
        # le mode plan (exp21, rollout_plan.run_plan_test_eval) évalue en
        # single-turn avec le protocole de SON training ; le save-best et le
        # logging eval/ restent identiques.
        self.eval_fn = eval_fn or run_test_eval
        self.best_score = best_init_score  # -1 => le 1er eval sauve toujours
        # Best TOUJOURS sur le disque home persistant, quel que soit --output-root.
        self.best_dir = str(REPO_ROOT / "saves" / "trl_grpo" / f"{run_name}_best")
        self.fewshot_block = fewshot_block
        self.fewshot_messages = fewshot_messages
        self.trainer_ref: Any = None  # injecté après la création du GRPOTrainer

    def on_step_end(self, targs: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return
        t0 = time.time()
        print(f"[test_eval] step {state.global_step} — éval test set en cours...", flush=True)
        try:
            # TRL synchronise les poids paresseusement (au début du step suivant) ;
            # à on_step_end le moteur tient encore les poids d'AVANT l'update → sync forcée.
            vllm_engine.sync_before_eval(self.trainer_ref)
            metrics = self.eval_fn(vllm_engine.get_engine(self.trainer_ref),
                                   self.trainer_ref.processing_class, max_items=self.eval_items,
                                   fewshot_block=self.fewshot_block,
                                   fewshot_messages=self.fewshot_messages)
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

        if score >= self.best_score:
            import shutil
            tmp = self.best_dir + ".tmp"
            swap_ok = False
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                if self.delete_before_save:
                    # Pic disque = 1 seul modèle (voir __init__) — le swap final
                    # os.replace reste valide (best_dir absent = simple rename).
                    shutil.rmtree(self.best_dir, ignore_errors=True)
                if self.save_optimizer:
                    # Supprimer l'ancien optimizer.pt AVANT d'écrire le nouveau modèle :
                    # plafonne le pic disque à ancien_modèle + nouveau_modèle + rien
                    # (au lieu de + ancien optimizer). Fenêtre de risque assumée : si le
                    # run meurt pendant l'écriture du modèle, le best précédent reste
                    # complet côté poids mais sans optimiseur.
                    old_opt = os.path.join(self.best_dir, "optimizer.pt")
                    if os.path.exists(old_opt):
                        os.remove(old_opt)
                unwrapped = self.trainer_ref.accelerator.unwrap_model(self.trainer_ref.model)
                is_peft = (hasattr(unwrapped, "merge_adapter")
                           and hasattr(unwrapped, "unmerge_adapter"))
                if is_peft:
                    unwrapped.save_pretrained(tmp)  # adapter-only
                    print(f"[test_eval] (LoRA) adapter-only save -> {tmp} "
                          f"(adapter_model.safetensors + adapter_config.json, PAS de merge)",
                          flush=True)
                else:
                    vllm_engine.save_model_for_vllm(unwrapped, tmp)  # full-ft: modèle complet
                self.trainer_ref.processing_class.save_pretrained(tmp)
                with open(os.path.join(tmp, ".best_info"), "w") as f:
                    f.write(f"step={state.global_step} pass_at_1={score:.4f}\n")
                shutil.rmtree(self.best_dir, ignore_errors=True)
                os.replace(tmp, self.best_dir)  # swap atomique (même FS home)
                print(f"[test_eval] >>> NOUVEAU BEST {100*score:.0f}% "
                      f"(ancien {100*self.best_score:.0f}%) sauvé sur disque persistant: "
                      f"{self.best_dir}", flush=True)
                self.best_score = score
                swap_ok = True
            except Exception as e:
                shutil.rmtree(tmp, ignore_errors=True)
                print(f"[test_eval] !!! SAUVEGARDE BEST ÉCHOUÉE step {state.global_step}: "
                      f"{e!r} — ancien best conservé, training continue.", flush=True)

            if self.save_optimizer and swap_ok:
                # APRÈS le swap du modèle (jamais dans le .tmp : ça doublerait le pic
                # disque). Écriture en .tmp + os.replace : un optimizer.pt présent est
                # toujours complet. Un échec ici ne perd ni le best ni le run.
                try:
                    import torch
                    t_opt = time.time()
                    opt_tmp = os.path.join(self.best_dir, "optimizer.pt.tmp")
                    torch.save(self.trainer_ref.optimizer.state_dict(), opt_tmp)
                    os.replace(opt_tmp, os.path.join(self.best_dir, "optimizer.pt"))
                    sz_gb = os.path.getsize(os.path.join(self.best_dir, "optimizer.pt")) / 1e9
                    print(f"[test_eval] optimizer.pt sauvé avec le best "
                          f"({sz_gb:.1f} Go, {time.time() - t_opt:.0f}s)", flush=True)
                except Exception as e:
                    try:
                        os.remove(os.path.join(self.best_dir, "optimizer.pt.tmp"))
                    except OSError:
                        pass
                    print(f"[test_eval] !!! SAUVEGARDE OPTIMIZER ÉCHOUÉE step "
                          f"{state.global_step}: {e!r} — best (poids) intact, training continue.",
                          flush=True)

        metrics["best_pass_at_1"] = max(score, self.best_score)
        self.trainer_ref.log({f"eval/{k}": v for k, v in metrics.items()})
