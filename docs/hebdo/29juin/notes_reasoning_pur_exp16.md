# Notes pour la semaine du 29 juin — Reasoning pur (exp16) & erreurs d'extraction

> ⚠️ **PARTIELLEMENT PÉRIMÉ — voir l'audit du 10 juillet**
> (`docs/hebdo/10juillet/audit_exp16_reasoning_pur.md`) :
> l'hypothèse « troncature max_tokens=1024 » de ce document est **réfutée**
> (les erreurs d'extraction sont de la dégénérescence en boucle) ; et les
> sweeps de température invalident tout verdict tiré d'un tirage isolé
> (bruit mono-tirage ±5 pts). Les chiffres corrigés font foi dans
> `docs/RESULTS.md` §(a)-(c).

Consolidation de l'analyse du 26/06 sur les expés "reasoning pur" single-turn
(plan texte libre → extraction JSON → replay TextCraft). À garder sous la main
pour la pres et pour le débogage de la phase d'extraction.

---

## 1. Résultats reasoning pur (pass@1 oracle, single-turn)

Protocole exp16 : le modèle génère **un** plan en texte libre (one-shot, pas de
boucle Thought/Action multi-tour), puis un 2ᵉ appel LLM convertit ce plan en
liste d'actions JSON, qu'on rejoue déterministiquement dans TextCraft.

| Depth (n items) | Qwen2.5-3B | Qwen3.5-4B |
|---|---|---|
| depth 1 (31) | 13/31 — 41.9 % | 24/31 — 77.4 % |
| depth 2 (41) | 7/41 — 17.1 % | 22/41 — 53.7 % |
| depth 3 (25) | 0/25 — 0 % | 8/25 — 32.0 % |
| depth 4 (3) | 0/3 — 0 % | 0/3 — 0 % |
| **Total /100** | **20/100** | **54/100** |
| Erreurs d'extraction | 11/100 | 2/100 |

- Runs : `runs/exp16_single_turn_reasoning/` (Qwen2.5-3B), `runs/exp16_qwen35_4b/` (Qwen3.5-4B).
- Plot : `docs/dashboard/09_reasoning_pur_par_depth.png` (script `src/analysis/plot_reasoning_pur.py`).

**À retenir** :
- Qwen3.5-4B double le score global (54 vs 20), gain surtout aux **depth 2-3**
  (débloque depth 3 à 32 % là où Qwen2.5-3B est déjà à 0 %).
- **depth 4 = mur universel (0/3)** pour les deux, cohérent avec toutes les autres expés.
- Comparaison clé pour la pres : en **multi-tour** (exp15, non-thinking),
  Qwen3.5-4B fait **77/100** vs **54/100** en reasoning pur single-turn
  → le single-turn reste nettement sous le multi-tour pour ce modèle.

---

## 2. Définition exacte d'une "erreur d'extraction"

Le pipeline fait **2 appels LLM** (même modèle) :

```
Phase 1 (collect_single_turn_plans.py) : modèle → PLAN texte libre  → plans/textcraft_N.json
Phase 2 (replay_single_turn_plans.py)  : modèle → JSON actions  [max_tokens=1024]
                                          → action_to_textcraft() → "Action: <cmd>"
                                          → TextCraftEnvClient.step() → replay_logs/textcraft_N.json
```

Une "erreur d'extraction" = champ `error` non-null dans le log de replay, levé dès
qu'une exception survient pendant la **phase 2** :
1. la sortie du parseur n'est pas du JSON valide (`json.loads` échoue) ;
2. pas de liste `actions` valide/non-vide ;
3. action non mappable en commande TextCraft.

**Cause réelle observée** : les 13 erreurs (11 + 2) sont **toutes** des erreurs de
décodage JSON (`Unterminated string`, `Expecting ',' delimiter`, `Expecting value`),
quasiment toutes vers le **char ~2600-2900 ≈ 1024 tokens** → le JSON de l'extracteur
est **tronqué** par `max_tokens=1024` (cf. `src/eval/replay_single_turn_plans.py:119`).
Ce n'est donc PAS l'entrée de TextCraft qui est tronquée : c'est le **JSON intermédiaire
produit par le 2ᵉ appel LLM**, coupé avant d'avoir pu être parsé → TextCraft ne reçoit
jamais rien pour ces items (`extracted_actions=[]`, reward=0). Problème de longueur de
génération, pas (que) d'incompétence — d'où l'écart 11 (Qwen2.5-3B) vs 2 (Qwen3.5-4B).

### Logs des items en erreur
- Qwen2.5-3B (11) — `runs/exp16_single_turn_reasoning/replay_logs/` :
  `textcraft_5` (d1) ; `140,144,158,159,161,164,171,177` (d2) ; `432,433` (d3).
- Qwen3.5-4B (2) — `runs/exp16_qwen35_4b/replay_logs/` :
  `textcraft_438` (d3), `textcraft_533` (d4).

---

## 3. Limite connue & fix proposé (à faire)

- **Limite** : sur le chemin d'erreur, le code remet `extraction_raw=""` → le JSON
  tronqué exact **n'est pas conservé** (on a le message d'erreur + le plan d'origine
  dans `plans/`, mais pas la sortie coupée de l'extracteur).
- **Fix proposé** (`src/eval/replay_single_turn_plans.py`) :
  1. sauvegarder `extraction_raw` **même en cas d'erreur** (debug des troncatures) ;
  2. monter `max_tokens` de l'extracteur (1024 trop bas pour les plans longs de depth 2-3).
- Après fix : re-rejouer uniquement les items en erreur (`--force-redo` ciblé) pour
  voir combien de "fausses" erreurs (pures troncatures) deviennent des succès/échecs réels.

---

## 4. Fichiers de code de référence

| Rôle | Fichier |
|---|---|
| Phase 1 — génère le plan (reasoning) | `src/eval/collect_single_turn_plans.py` |
| Phase 2 — extraction JSON + replay | `src/eval/replay_single_turn_plans.py` |
| Wrapper appel LLM (`max_tokens`, vLLM/HF) | `src/eval/llm_chat.py` |
| Client env TextCraft (`.step()`) | `external/AgentGym/agentenv/agentenv/envs/textcraft.py` |
