"""
Mini-cycle d'un EnvClient TextCraft, SANS LLM.

But : valider à la main que `TextCraftEnvClient` parle bien au serveur HTTP, et
visualiser ce que renvoie chaque méthode (`reset`, `observe`, `step`).

Pré-requis :
1. Le serveur TextCraft tourne sur 127.0.0.1:36005, lancé hors sandbox depuis
   `AgentGym/agentenv-textcraft/` (cf. WORKLOG.md, section "Pièges").
2. Conda env `agentgym-rl` activé (la stack du trainer ; c'est l'env qui
   contient `agentenv` côté client).

Lancement :
    conda activate agentgym-rl
    python scratch/01_minicycle.py
"""

from agentenv.envs import TextCraftEnvClient


def banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def show(label: str, obj) -> None:
    print(f"[{label}] {obj!r}")


def main() -> None:
    banner("1. Connect to env server (this triggers POST /create)")
    client = TextCraftEnvClient(
        env_server_base="http://127.0.0.1:36005",
        data_len=1,
        timeout=60,
    )
    show("env_id", client.env_id)
    show("initial observation", client.observe())

    banner("2. Reset to task #0 (POST /reset)")
    client.reset(0)
    show("obs after reset", client.observe())

    banner("3. Step #1 - check inventory (action toujours valide)")
    out = client.step("Thought: Let me look at my inventory.\n\nAction: inventory")
    show("state", out.state)
    show("reward / done", (out.reward, out.done))

    banner("4. Step #2 - tenter de get 9 gold nuggets")
    out = client.step("Thought: I need gold nuggets.\n\nAction: get 9 gold nugget")
    show("state", out.state)
    show("reward / done", (out.reward, out.done))

    banner("5. Step #3 - tenter le craft cible")
    out = client.step(
        "Thought: Now I craft the goal item.\n\n"
        "Action: craft 1 gold ingot using 9 gold nugget"
    )
    show("state", out.state)
    show("reward / done", (out.reward, out.done))

    banner("6. Bilan final")
    show("env_id", client.env_id)
    show("final observation cached in client.info", client.info["observation"])


if __name__ == "__main__":
    main()
