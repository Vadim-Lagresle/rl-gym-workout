# Audit exp16 (reasoning pur) — fixes et requantification (2026-07-08)

Suite de l'audit du pipeline « plan single-turn → extraction JSON → replay
déterministe » (exp16). Ce document consigne les fixes appliqués, les reruns,
et l'ablation « extracteur aveugle » qui requalifie les scores.

## 1. Rappel du pipeline et des biais identifiés

Deux appels LLM par item : (1) plan en anglais libre (temp 0.7, 1024 tokens),
(2) extraction JSON des actions (temp 0.0), puis replay sans LLM dans TextCraft.
Pass si le goal est crafté (reward 0/1 de l'env).

Biais identifiés à l'audit :
- **Sur-évaluation** : l'extracteur reçoit recettes + goal et la consigne
  « include every step needed » → il peut compléter/réparer un plan faux.
- **Sous-évaluation supposée** : 11 « erreurs d'extraction » (3B) ressemblaient
  à de la troncature JSON à max_tokens=1024 → items comptés 0 d'office.
- Défauts d'observabilité : `extraction_raw` perdu sur échec, `finish_reason`
  jamais lu, cache qui re-sert les erreurs.

## 2. Fixes appliqués (src/eval/)

| Fix | Fichier | Détail |
|---|---|---|
| Parsing JSON tolérant | `replay_single_turn_plans.py` | fence ```json``` → brut → sous-chaîne 1er `{`..dernier `}` |
| Budget extraction | idem | `--extract-max-tokens`, défaut 1024 → **2048** |
| Raw conservé sur échec | idem | exception porte `.raw` → `extraction_raw` rempli dans le log |
| Cache retente les erreurs | idem | `load_existing_replay` retourne None si `error` |
| Mode ablation | idem | `--blind` : prompt d'extraction **sans recettes ni goal**, consigne de traduction littérale |
| Warning troncature | `llm_chat.py` | vLLM : `finish_reason != stop` ; HF : génération au cap |

## 3. Résultat du rerun des items en erreur — hypothèse de troncature RÉFUTÉE

Rerun des 11 items 3B (et 2 items 4B) avec budget 2048 + parsing tolérant :
**aucun ne passe — les scores restent 20/100 (3B) et 54/100 (4B).**

Diagnostic (possible grâce au raw maintenant sauvé) : ce n'était pas de la
troncature récupérable mais de la **dégénérescence en boucle** à temp 0.0 —
le modèle répète indéfiniment les 2-3 mêmes actions jusqu'au cap, quel que
soit le budget. Exemples :
- 3B, `textcraft_5` (d1) : 28× `craft stick` / 27× `craft string` en alternance ;
- 4B, `textcraft_438` (d3) : 24× `get lime dye`.

Ces items correspondent aux plans les plus longs/verbeux de la phase 1 : un
plan confus fait dégénérer l'extracteur. **Ce sont des échecs légitimes du
modèle, correctement comptés 0.** La crainte de sous-évaluation par troncature
est levée.

## 4. Ablation « extracteur aveugle » — le biais de sur-évaluation quantifié

Même plans, même replay, mais le prompt d'extraction ne contient **ni les
recettes ni le goal** : l'extracteur traduit le plan littéralement (consigne
explicite de ne rien ajouter/corriger).

| Modèle | Extracteur informé | Extracteur aveugle | Δ (part « réparation ») |
|---|---|---|---|
| Qwen2.5-3B | 20/100 | **12/100** | −8 pts (40 % du score) |
| Qwen3.5-4B | 54/100 | **46/100** | −8 pts (15 % du score) |

Détail par depth (informé → aveugle) :
- 3B : d1 13→9/31, d2 7→3/41, d3 0→0, d4 0→0. Flips : 11 OK→FAIL, 3 FAIL→OK.
- 4B : d1 24→**25**/31 (+1 !), d2 22→15/41, d3 8→6/25, d4 0→0. Flips : 16
  OK→FAIL, 8 FAIL→OK, et **0 erreur d'extraction** en aveugle.

L'extracteur informé peut aussi **dégrader** un plan correct en le « réparant »
de travers (3 items 3B, 8 items 4B récupérés par la traduction littérale).
Lecture : la dépendance relative à l'extracteur-solveur est bien plus faible
pour le 4B (15 %) que pour le 3B (40 %) — les plans du 4B sont plus
auto-suffisants (get explicites, noms corrects), ce qui renforce la conclusion
« le 4B raisonne mieux » au-delà du simple écart de score.

Ce que l'extracteur informé répare (constaté sur les flips) :
1. **Étapes `get` implicites** — beaucoup de plans ne listent que les crafts ;
2. **Normalisation des noms** vers le vocabulaire exact des recettes ;
3. **Quantités / ordre** induits des recettes ;
4. **Discipline de schéma** (l'aveugle émet parfois des types hors schéma,
   ex. `confirm`).

## 5. Lecture corrigée des résultats exp16

- Le « reasoning pur » du plan seul se situe **entre le score aveugle (borne
  basse, traduction volontairement stricte) et le score informé (borne haute,
  extracteur-solveur)** : 3B ∈ [12, 20], 4B ∈ [46, 54].
- Les conclusions qualitatives tiennent : 4B ≫ 3B quel que soit l'extracteur,
  **mur depth 4 = 0/3 partout**, single-turn < multi-tour (77/100 exp15).
- Pour la présentation : nommer la métrique historique « plan + extraction
  informée », et citer l'encadrement plutôt qu'un chiffre unique.

## 6. Restes non traités (proposés)

- **Alignement de température** : plans générés à 0.7 vs baseline multi-tour à
  1.0 — regénérer les plans à 1.0 (ou pass@k, k=4-8) pour dé-bruiter le tirage
  unique. Non fait : change les plans, à décider séparément.
- Répétition dégénérée de l'extracteur : un `repetition_penalty` léger ou un
  garde-fou « stop après N actions identiques » réduirait les 8-11 erreurs,
  mais modifie la sémantique de l'éval — à discuter.

Scripts : `src/eval/replay_single_turn_plans.py` (`--blind`,
`--extract-max-tokens`), runs : `runs/exp16_blind_extraction_3b/`,
`runs/exp16_blind_extraction_4b/`.
