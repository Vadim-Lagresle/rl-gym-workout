"""Rollout single-turn « pure reasoning » (exp21) + prompts + éval périodique.

Paradigme : au lieu de la boucle multi-tour de rollout.py (generate → env.step →
observe → repeat, jusqu'à 20 tours), le modèle produit EN UNE SEULE complétion le
raisonnement PUIS la séquence d'actions complète (le « plan »). On parse les
actions directement dans le texte généré, on les rejoue TELLES QUELLES dans
TextCraft (aucune réparation, aucun feedback intermédiaire), et le reward sparse
0/1 de l'environnement est rétropropagé sur la complétion entière par GRPO.

Différence clé avec le pipeline exp16 (src/eval/single_turn/) : exp16 passait par
un SECOND appel LLM pour extraire les actions d'un plan en texte libre —
inutilisable en RL (le reward dépendrait de l'extracteur, pas des tokens
générés ; l'extracteur « informé » réparait même les plans). Ici le modèle doit
émettre les actions dans la syntaxe exécutable, le parsing est un simple regex
déterministe : le crédit revient à 100 % aux tokens du modèle.

Contrat TRL : trivial par rapport au multi-tour — completion_ids = LA complétion
unique, env_mask tout à 1 (aucune observation entrelacée), logprobs de vLLM.
La reward function reste rollout.textcraft_reward (sparse outcome, même contrat).

Branché dans train_grpo.py via --plan-mode (même patron que la variante SNIS).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentenv.envs import TextCraftEnvClient

from src.train import vllm_engine
from src.train.data import ENV_SERVER_URL, ITEM_TAG_RE, REPO_ROOT, TRAIN_PATH, item_id_to_idx

# Cap de replay : profondeur max ~4 avec multi-ingrédients → largement sous 50
# (même valeur que MAX_ACTIONS du replayer exp16).
MAX_PLAN_ACTIONS = 50

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Message de règles (1er tour user) — remplace le conversation_start multi-tour
# du serveur (qui décrit le protocole ReAct tour par tour, inadapté ici).
PLAN_RULES = """You are given a Minecraft crafting puzzle. You will receive a list of crafting commands and a goal.

Think step by step, then output the COMPLETE ordered sequence of actions that reaches the goal, from gathering base ingredients to the final craft. You only get ONE answer: your actions will be executed exactly as written, in order, with no feedback between steps.

Rules:
- "get <count> <item>" fetches ingredients from the environment. You can ONLY get base items, i.e. items that have no crafting command in the list.
- "craft <count> <item> using <count> <item>, <count> <item>, ..." must match one of the given crafting commands exactly (same ingredients, same counts). You must already have every ingredient in your inventory.
- A crafting command can be executed several times if you need more of an intermediate item.

Answer format — reasoning first, then exactly one action per line after the "Actions:" marker:
Thought: <your step-by-step reasoning>
Actions:
get <count> <item>
craft <count> <item> using <count> <item>
..."""

PLAN_ACK = ("OK. I will reply with my reasoning after \"Thought:\" followed by the "
            "complete ordered action sequence after \"Actions:\", one action per line.")


def load_plan_fewshot(fewshot_file: str, k: int) -> list[dict]:
    """Reformate les exemples résolus d'exp18/19 (mêmes recettes hors train/test)
    au format single-turn : user = tâche (recettes + goal, format identique à
    l'observation env), assistant = Thought + séquence d'actions complète."""
    with open(fewshot_file) as f:
        examples = json.load(f)
    if k > len(examples):
        raise SystemExit(f"--fewshot {k} > {len(examples)} exemples dans {fewshot_file}")
    messages: list[dict] = []
    for e in examples[:k]:
        task = ("Crafting commands:\n" + "\n".join(e["commands"])
                + f"\n\nGoal: craft {e['goal_str']}.")
        actions = "\n".join(st["action"] for st in e["steps"])
        messages.append({"role": "user", "content": task})
        messages.append({"role": "assistant",
                         "content": f"Thought: {e['thought']}\nActions:\n{actions}"})
    return messages


def build_plan_prompt_rows(max_items: int, system_prompt: str,
                           fewshot_messages: list[dict] | None = None) -> list[dict[str, Any]]:
    """Équivalent single-turn de data.build_prompt_rows : mêmes items d'entraînement,
    même marqueur <ITEM_IDX:n> (remplacé par l'observation env au rollout), mais
    règles PLAN_RULES et exemples single-turn (paires user/assistant fermées —
    pas d'observation à reporter dans le message porteur du marqueur)."""
    with TRAIN_PATH.open() as f:
        rows = json.load(f)
    if max_items > 0:
        rows = rows[:max_items]

    out: list[dict[str, Any]] = []
    for r in rows:
        idx = item_id_to_idx(r["item_id"])
        marker = (f"Now solve this new task:\n\n<ITEM_IDX:{idx}>"
                  if fewshot_messages else f"<ITEM_IDX:{idx}>")
        base = [{"role": "system", "content": system_prompt}] if system_prompt else []
        prompt = base + [
            {"role": "user", "content": PLAN_RULES},
            {"role": "assistant", "content": PLAN_ACK},
            *(fewshot_messages or []),
            {"role": "user", "content": marker},
        ]
        out.append({"prompt": prompt, "item_id": r["item_id"], "item_idx": idx})
    return out


# ---------------------------------------------------------------------------
# Parsing du plan → actions exécutables
# ---------------------------------------------------------------------------

# Une ligne d'action, tolérante aux décorations usuelles (numérotation, puces,
# "Step 3:", "Action:") mais STRICTE sur la commande elle-même : la séquence est
# rejouée telle quelle, on ne répare rien.
_ACTION_LINE_RE = re.compile(
    r"^(?:[-*•]\s*|\d+[.)]\s*|step\s*\d+\s*[:.)]\s*|action\s*:\s*)*"
    r"(get\s+\d+\s+.+?|craft\s+\d+\s+.+?\s+using\s+.+?|inventory)\s*[.;]*\s*$",
    re.IGNORECASE,
)


def parse_plan_actions(text: str) -> list[str]:
    """Extrait les commandes TextCraft de la complétion (une par ligne).

    Le prompt demande un bloc "Actions:" mais on scanne toutes les lignes : une
    ligne est une action ssi elle matche la syntaxe get/craft/inventory — le
    raisonnement en prose n'y ressemble jamais, et ça tolère un modèle qui
    oublie le marqueur."""
    actions: list[str] = []
    for line in text.splitlines():
        m = _ACTION_LINE_RE.match(line.strip())
        if m:
            actions.append(m.group(1).strip())
    return actions[:MAX_PLAN_ACTIONS]


# ---------------------------------------------------------------------------
# Replay déterministe dans l'environnement
# ---------------------------------------------------------------------------

def replay_actions(env: TextCraftEnvClient, actions: list[str]) -> tuple[float, int, int]:
    """Rejoue la séquence telle quelle. Retourne (reward, n_invalid, n_steps).

    Le préfixe "Action: " est requis par TextCraftEnvClient (il extrait le texte
    après le marqueur — même convention que le replayer exp16). On s'arrête à
    done (goal atteint) ; une action invalide ne stoppe PAS le replay : l'env
    renvoie une observation d'erreur et on continue, comme un agent têtu."""
    reward, n_invalid = 0.0, 0
    for i, cmd in enumerate(actions):
        out = env.step(f"Action: {cmd}")
        reward = max(reward, float(out.reward))
        low = out.state.lower()
        if "could not" in low or "error:" in low or "wrong item format" in low:
            n_invalid += 1
        if bool(out.done):
            return reward, n_invalid, i + 1
    return reward, n_invalid, len(actions)


# ---------------------------------------------------------------------------
# rollout_func GRPO single-turn
# ---------------------------------------------------------------------------

def plan_rollout_func(prompts: list[list[dict[str, str]]], trainer: Any) -> dict[str, Any]:
    """Une génération unique par prompt, puis replay exact du plan parsé."""
    tokenizer = trainer.processing_class
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    n = len(prompts)
    states = [list(p) for p in prompts]
    env_clients: list[TextCraftEnvClient] = []

    # Bootstrap identique à rollout.collect_episodes : marqueur → observation.
    for i in range(n):
        content = str(states[i][-1]["content"])
        m = ITEM_TAG_RE.search(content)
        if m is None:
            raise ValueError(f"Missing item marker in prompt[{i}]: {content!r}")
        env = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        env.reset(int(m.group(1)))
        states[i][-1]["content"] = content.replace(m.group(0), env.observe()).strip()
        env_clients.append(env)

    prompt_ids_all = []
    for i in range(n):
        rendered = tokenizer.apply_chat_template(states[i], tokenize=False, add_generation_prompt=True)
        prompt_ids_all.append(tokenizer.encode(rendered, add_special_tokens=False))

    active = list(range(n))
    if vllm_engine.get_engine(trainer) is not None:
        completion_ids, logprobs = vllm_engine.generate_round(
            trainer, tokenizer, states, active,
            max_tokens=trainer.args.max_completion_length,
        )
    else:
        completion_ids, logprobs = trainer._generate_single_turn(
            prompt_ids_all, images=None, multimodal_fields={}
        )

    # Parsing + replay parallèle (IO-bound HTTP ; chaque épisode reste séquentiel).
    plans = [parse_plan_actions(tokenizer.decode(list(ids), skip_special_tokens=True))
             for ids in completion_ids]

    def _replay(i: int) -> tuple[float, int, int]:
        if not plans[i]:
            return 0.0, 0, 0
        return replay_actions(env_clients[i], plans[i])

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(_replay, range(n)))

    for env in env_clients:
        try:
            env.close()
        except Exception:
            pass

    ids_out, lps_out, mask_out = [], [], []
    for j in range(n):
        ids = list(completion_ids[j])
        lps = list(logprobs[j][: len(ids)]) if logprobs is not None else [0.0] * len(ids)
        if not ids or ids[-1] != tokenizer.eos_token_id:
            # Tronqué par max_completion_length : on ferme le tour (même convention
            # que rollout.collect_episodes — loss active sur l'im_end forcé).
            ids = ids + im_end_ids
            lps = lps + [0.0] * len(im_end_ids)
        ids_out.append(ids)
        lps_out.append(lps)
        mask_out.append([1] * len(ids))  # single-turn : tout est sortie modèle

    rewards = [r for r, _, _ in results]
    invalids = [inv for _, inv, _ in results]
    n_actions = [float(len(p)) for p in plans]
    print(
        f"[rollout-plan] n={n} n_actions={[len(p) for p in plans]} "
        f"steps_executed={[s for _, _, s in results]} "
        f"episode_reward={rewards} invalid_steps={invalids}",
        flush=True,
    )
    return {
        "prompt_ids": prompt_ids_all,
        "completion_ids": ids_out,
        "logprobs": lps_out,
        "env_mask": mask_out,
        "episode_reward": rewards,
        "invalid_steps": invalids,
        # Contrat textcraft_reward (télémétrie pure) : ici = taille du plan parsé.
        "mean_n_actions_per_turn": n_actions,
    }


# ---------------------------------------------------------------------------
# Éval périodique single-turn (branchée sur TestEvalCallback via eval_fn)
# ---------------------------------------------------------------------------

EVAL_DATASET_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test.json"
EVAL_DEPTH_PATH = REPO_ROOT / "data" / "eval" / "textcraft_test_with_depth.json"


def run_plan_test_eval(llm: Any, tokenizer: Any, max_items: int = 0,
                       temperature: float = 1.0, max_tokens: int = 1024,
                       fewshot_block: str | None = None,
                       fewshot_messages: list[dict] | None = None) -> dict[str, float]:
    """Pass@1 single-turn sur le test set — MÊME protocole que le training
    (mêmes règles, mêmes exemples, replay exact), moteur vLLM colocate.

    Signature alignée sur periodic_eval.run_test_eval pour être passée telle
    quelle à TestEvalCallback(eval_fn=...). fewshot_block est ignoré (les
    exemples single-turn sont des tours de dialogue, pas un bloc)."""
    from vllm import SamplingParams

    with EVAL_DATASET_PATH.open() as f:
        items = json.load(f)
    if max_items > 0:
        items = items[:max_items]
    depth_map: dict[str, int] = {}
    if EVAL_DEPTH_PATH.exists():
        with EVAL_DEPTH_PATH.open() as f:
            depth_map = json.load(f)

    n = len(items)
    states, clients = [], []
    for it in items:
        client = TextCraftEnvClient(env_server_base=ENV_SERVER_URL, data_len=10000, timeout=60)
        client.reset(item_id_to_idx(it["item_id"]))
        obs = client.observe()
        task = f"Now solve this new task:\n\n{obs}" if fewshot_messages else obs
        states.append([
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": PLAN_RULES},
            {"role": "assistant", "content": PLAN_ACK},
            *(fewshot_messages or []),
            {"role": "user", "content": task},
        ])
        clients.append(client)

    prompts = [tokenizer.apply_chat_template(s, tokenize=False, add_generation_prompt=True)
               for s in states]
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    plans = [parse_plan_actions(out.outputs[0].text) for out in outputs]

    def _replay(i: int) -> tuple[float, int, int]:
        if not plans[i]:
            return 0.0, 0, 0
        return replay_actions(clients[i], plans[i])

    with ThreadPoolExecutor(max_workers=min(n, 32)) as executor:
        results = list(executor.map(_replay, range(n)))
    for c in clients:
        try:
            c.close()
        except Exception:
            pass

    rewards = [r for r, _, _ in results]
    solved = sum(1 for r in rewards if r >= 1.0)
    metrics: dict[str, float] = {
        "pass_at_1": solved / n,
        "solved": float(solved),
        "n_items": float(n),
        "mean_plan_actions": sum(len(p) for p in plans) / n,
        "mean_steps_executed": sum(s for _, _, s in results) / n,
        "errors_per_ep": sum(inv for _, inv, _ in results) / n,
        "parse_fail": float(sum(1 for p in plans if not p)),
    }
    depths = [depth_map.get(items[i]["item_id"]) for i in range(n)]
    for d in (1, 2, 3, 4):
        idx = [i for i in range(n) if depths[i] == d]
        if idx:
            metrics[f"pass1_d{d}"] = sum(1 for i in idx if rewards[i] >= 1.0) / len(idx)
    return metrics


# ---------------------------------------------------------------------------
# Selftest à sec (parsing uniquement — pas de serveur ni GPU requis)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """Thought: lapis block needs 9 lapis lazuli, which is a base item.
Actions:
get 9 lapis lazuli
craft 1 lapis block using 9 lapis lazuli"""
    assert parse_plan_actions(sample) == [
        "get 9 lapis lazuli",
        "craft 1 lapis block using 9 lapis lazuli",
    ], parse_plan_actions(sample)

    decorated = """Here is my plan:
1. get 8 terracotta
2) get 1 white tulip
- Action: craft 1 light gray dye using 1 white tulip
Step 4: craft 8 light gray terracotta using 8 terracotta, 1 light gray dye.
inventory
Then we are done. I will craft the item now."""
    assert parse_plan_actions(decorated) == [
        "get 8 terracotta",
        "get 1 white tulip",
        "craft 1 light gray dye using 1 white tulip",
        "craft 8 light gray terracotta using 8 terracotta, 1 light gray dye",
        "inventory",
    ], parse_plan_actions(decorated)

    prose = """Thought: I need to get the ingredients first. To craft the block I
will use nine lapis. Let me think about what to get and craft here.
There is nothing else to do."""
    assert parse_plan_actions(prose) == [], parse_plan_actions(prose)

    print("[rollout_plan] selftest parsing OK")
