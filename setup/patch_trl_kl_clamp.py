"""Borne haute optionnelle de l'estimateur KL k3 de TRL (patch idempotent, 15/09/2026).

Pourquoi : verl (stack du papier) borne k3 = exp(ρ) − ρ − 1 à [−10, 10] par token
(external/AgentGym-RL/verl/agent_trainer/ppo/core_algos.py:381) ; TRL ne borne pas
(trl/trainer/grpo_trainer.py, `per_token_kl = torch.exp(...) - (...) - 1`). Sans borne,
un seul token à ρ ≈ 18-45 (rejeu k3 du 15/09 : « Thought » redevenu rare, <|im_end|>
forcé après troncature) pèse 1e7 à 1e19 dans la perte et dicte le pas de gradient.

Ce script insère, juste après l'expression de TRL, une borne ACTIVÉE UNIQUEMENT si la
variable d'environnement TRL_KL_CLAMP est posée (train_grpo.py --kl-clamp la pose).
Sans la variable, TRL est inchangé. Appelé à la fin de setup_agentgym_rl_v2.sh
(l'env vit sur /tmp, purgé régulièrement) ; train_grpo.py vérifie la présence du
marqueur avant de lancer un run avec --kl-clamp.

Usage : <python de l'env> setup/patch_trl_kl_clamp.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# rl-gym-workout: borne k3 (TRL_KL_CLAMP)"
TARGET = (
    "            per_token_kl = (\n"
    "                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1\n"
    "            )\n"
)
PATCH = TARGET + (
    f"            {MARKER}\n"
    "            _kl_clamp = os.environ.get(\"TRL_KL_CLAMP\")\n"
    "            if _kl_clamp:\n"
    "                per_token_kl = torch.clamp(per_token_kl, max=float(_kl_clamp))\n"
    "                if not getattr(self, \"_kl_clamp_logged\", False):\n"
    "                    print(f\"[kl-clamp] borne k3 active : max={float(_kl_clamp)} par token (comme verl low_var_kl)\", flush=True)\n"
    "                    self._kl_clamp_logged = True\n"
)


def trl_file() -> Path:
    import trl.trainer.grpo_trainer as m
    return Path(m.__file__)


def main() -> int:
    check_only = "--check" in sys.argv
    f = trl_file()
    src = f.read_text()
    if MARKER in src:
        print(f"[patch-trl] déjà patché : {f}")
        return 0
    if check_only:
        print(f"[patch-trl] NON patché : {f}")
        return 1
    if src.count(TARGET) != 1:
        print(f"[patch-trl] expression cible introuvable ou multiple ({src.count(TARGET)}) dans {f} — TRL a changé, patch NON appliqué")
        return 2
    if "\nimport os\n" not in src:
        print(f"[patch-trl] `import os` absent de {f} — patch NON appliqué")
        return 3
    f.write_text(src.replace(TARGET, PATCH))
    print(f"[patch-trl] patch appliqué : {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
