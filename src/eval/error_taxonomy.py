"""The six kinds of mistakes TextCraft reports, and how to recognise them.

In plain words: when an action fails, the TextCraft server answers with a message
("Could not find a valid recipe", "Only one 'Action' is allowed"...). This file maps
each message to an error class, so that evaluations can count, per episode and per
depth, how many format errors (malformed action, several actions in one turn) and
planning errors (invalid recipe, missing ingredients, item not found) the model
makes. Used live by the periodic evaluation (train/periodic_eval.py) and offline by
analysis/analyze_eval.py.
"""

from __future__ import annotations

import re

# Patterns de classification des observations d'erreur
# Ordre IMPORTANT : classify_error garde le PREMIER pattern qui matche.
# Les patterns spécifiques doivent précéder les résiduels (other_error, generic_fail).
ERROR_PATTERNS = [
    ("format_error",   re.compile(r"could not execute", re.I)),          # action mal formée, parseur rejette
    ("recipe_wrong",   re.compile(r"could not find a valid recipe", re.I)),
    ("missing_items",  re.compile(r"could not find enough items", re.I)),  # quantité insuffisante
    ("item_not_found", re.compile(r"could not find", re.I)),             # "Could not find <objet>" : absent inventaire / mal nommé
    ("wrong_format",   re.compile(r"wrong item format", re.I)),
    ("multi_action",   re.compile(r"only one .action. is allowed", re.I)),  # plusieurs "Action:" dans une réponse
    ("other_error",    re.compile(r"error:", re.I)),                     # résiduel "error:"
    ("generic_fail",   re.compile(r"could not", re.I)),                  # résiduel "could not …"
]

# Description lisible de chaque type d'erreur (affichée dans chaque analyse).
ERROR_DESCRIPTIONS = {
    "format_error":   "« could not execute » — action mal formée, le parseur de l'env la rejette (syntaxe).",
    "recipe_wrong":   "« could not find a valid recipe » — aucune recette valide pour la cible.",
    "missing_items":  "« could not find enough items » — prérequis en quantité insuffisante (erreur de planif).",
    "item_not_found": "« could not find <objet> » — objet absent de l'inventaire / mal nommé.",
    "wrong_format":   "« wrong item format » — nom d'objet mal écrit.",
    "multi_action":   "« only one 'Action' is allowed » — plusieurs actions émises en une réponse (protocole).",
    "other_error":    "résiduel : contient « error: » sans matcher un cas ci-dessus.",
    "generic_fail":   "résiduel : contient « could not » sans matcher un cas ci-dessus.",
}

ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def classify_error(obs: str) -> str | None:
    """Retourne la catégorie d'erreur d'une observation, ou None si succès/neutre."""
    for label, pat in ERROR_PATTERNS:
        if pat.search(obs):
            return label
    return None


def extract_action(text: str) -> str:
    """Extrait la première ligne d'action d'un message assistant."""
    m = ACTION_RE.search(text)
    if not m:
        return ""
    return " ".join(m.group(1).strip().split())
