# Session 2026-07-08 — Audit exp16, ablation température, reasoning pur du best58

> ⚠️ **CORRIGÉ le 2026-07-09 par le balayage fin de température (§6 en bas).**
> Deux conclusions de cette session, tirées de **points isolés bruités**, sont
> **invalidées** par le balayage T=0→1 (11 tirages/modèle) :
> 1. la « pénalité de température −5/−6 » (§4) → en fait **aucun effet détectable** ;
> 2. « le RL n'a rien transféré à la planification, 19≈20 » (§4ter) → en fait
>    **best58 (moy 19.5) > base (moy 12.3), +7.2, p=0.002** : transfert réel mais modeste.
> Lire le §6 comme référence ; §3-4 gardés pour l'historique.

Suite directe de l'audit exp16 (voir `audit_exp16_reasoning_pur.md`). Trois
questions traitées dans cette session :

1. Les « erreurs d'extraction » d'exp16 cachaient-elles une sous-évaluation ?
   → **Non** (dégénérescence, pas troncature — audit doc §3).
2. Quelle part du score vient de l'extracteur informé ?
   → **3B : 40 %, 4B : 15 %** (ablation aveugle — audit doc §4).
3. (cette partie) **Température 0.7 vs 1.0** sur la génération de plans, et
   **reasoning pur du modèle RL best58** (exp10.8, 58/100 multi-tour).

## 1. Théorie & prédictions (écrites AVANT les runs)

Température = adoucissement de la softmax (`softmax(logits/T)`). T=0.7 aiguise
la distribution (tokens probables favorisés, sorties stéréotypées, meilleure
adhérence au format, moins de noms/quantités erratiques) ; T=1.0 échantillonne
la distribution brute (plus de diversité — utile en pass@k — mais chaque token
rare mal tiré peut casser un plan en pass@1).

Prédictions posées avant les expériences :

| # | Prédiction | Attendu |
|---|---|---|
| P1 | 3B temp 1.0 (vs 20 @ 0.7) | léger recul ou stable : 15-20 |
| P2 | 4B temp 1.0 (vs 54 @ 0.7) | stable : 50-55 (modèle fort = robuste au bruit) |
| P3 | best58 reasoning pur | nettement sous son 58 multi-tour : ~25-40 (le RL optimise le contrôle réactif, pas le plan one-shot) |
| P4 | caveat | ±4-5 pts de bruit binomial sur 100 items / 1 tirage → écart <5 pts non significatif |

## 2. Protocole

- Pipeline exp16 inchangé (audit-fixé) : plan single-turn → extraction JSON
  (temp 0, 2048 tokens, parsing tolérant) → replay déterministe TextCraft.
- Chaque config : replay **informé** + ablation **aveugle** (sauf 4B temp 1.0 :
  informé seul, backend HF lent).
- best58 = merge `exp10p7_step92_53pct` + adapter best exp10.8
  (`qwen25_3b_exp10p8_step368_58pct`, scratchpad). Le MÊME modèle sert de
  planificateur et d'extracteur (comme pour 3B/4B).

## 3. Résultats

| Config | Informé | Aveugle | Réf. temp 0.7 (informé/aveugle) |
|---|---:|---:|---|
| 3B temp 1.0 | **14/100** | 12/100 | 20 / 12 |
| 4B temp 1.0 | **49/100** | — (non lancé, HF lent) | 54 / 46 |
| best58 temp 0.7 | **19/100** | 16/100 | (multi-tour : 54) |
| best58 temp 1.0 | **14/100** | 9/100 | (multi-tour : 54) |

Détail par depth (informé) :
- 3B t1.0 : d1 13/31, **d2 1/41** (vs 7/41 à 0.7), d3 0, d4 0 — 10 err.
- 4B t1.0 : d1 22/31, d2 20/41, d3 7/25, d4 0/3 — 2 err. Profil préservé.
- best58 t0.7 : d1 14/31, d2 5/41, d3 0, d4 0 — 9 err. t1.0 : d1 10/31, d2 4/41.

## 4. Impact de la température (synthèse)

**Pénalité systématique de −5 à −6 pts en montant 0.7 → 1.0** (informé), même
sens sur les 3 modèles :

| Modèle | T=0.7 | T=1.0 | Δ |
|---|---:|---:|---:|
| 3B base | 20 | 14 | −6 |
| 4B | 54 | 49 | −5 |
| best58 (RL) | 19 | 14 | −5 |

- **Mécanisme** : pass@1 = 1 seul tirage, sans feedback. À T=1.0 (distribution
  brute) un token « queue de distribution » (mauvais nom d'item, quantité fausse)
  casse tout le plan ; T=0.7 aiguise la softmax → sorties plus stéréotypées, moins
  d'accidents fatals.
- **Signature : effet porté par les plans longs.** depth 1 insensible (3B 13→13),
  c'est depth 2+ qui s'effondre (3B **7→1**). Plus le plan a de tokens, plus la
  proba qu'au moins un soit fatal explose à T haute.
- **Conséquence méthodo** : le « single-turn 20 vs multi-tour 18 » était **flatté
  par la température** (plans à 0.7 vs multi-tour à 1.0). À T alignée (1.0), le
  single-turn 3B tombe à **14 — SOUS le multi-tour (18)**. Même classe de biais
  que le 58→54 (chiffres non comparables car un paramètre variait en douce).
- **Interaction extracteur** : à T=1.0 l'extracteur informé répare moins
  (3B : écart informé−aveugle 8→2 pts) — un plan erratique n'est pas rattrapable.
- **Caveat** : ±4-5 pts de bruit binomial (100 items, 1 tirage) ; chaque Δ est
  limite-significatif isolément, la **cohérence 3/3** le rend crédible. Effet
  mesuré sur la génération de plans one-shot ; non testé en multi-tour.

## 4bis. Commentaire des prédictions (posées avant les runs)

| # | Prédit | Observé | Verdict |
|---|---|---|---|
| P1 (3B t1) | 15-20 | 14 | direction juste, d2 s'effondre plus que prévu |
| P2 (4B t1) | 50-55 | 49 | quasi exact — modèle fort = robuste au bruit |
| P3 (best58 reasoning pur ≪ 58) | 25-40 | **19** | direction très juste, mais encore trop optimiste : gain nul vs base |
| P4 (bruit ±4-5) | — | Δ −5/−6 cohérents 3/3 | validé |

## 4ter. best58 : le RL ne transfère pas à la planification (+ correction 58→54)

- best58 **re-évalué en multi-tour = 54/100** (pas 58). Le « 58 » du step 368
  était un pic isolé d'une courbe mono-tirage bruitée (voisins 40/49/58/49) →
  *winner's curse*. Le vrai niveau est bas-50. **Le merge est correct** (54 ≫ base
  18) : le 19 en reasoning pur est un vrai résultat, pas un bug de pipeline.
- best58 reasoning pur = **19 ≈ base 3B (20)** alors que multi-tour ~3× la
  baseline → le GRPO a optimisé la **politique réactive** (agir/observer/corriger),
  pas le raisonnement one-shot. Pour depth 3-4 : injecter la planification
  (CoT tour 1, SFT plans, distillation) plutôt que l'attendre du RL outcome-only.
- Vérifications anti-bug passées : bon chemin modèle dans les plans, plans 100 %
  distincts de la base, safetensors intègre (6,17 Go), 0 fuite de format RL.

## 5. Récap de la session du matin (2026-07-08)

- **exp10.8 terminé : 58/100 (step 368), nouveau record projet** — run complet
  400 steps sans collapse, hypothèse LR÷3 validée. Configs/INDEX à jour.
- **Audit exp16** : code décrit et vérifié de bout en bout (scores recomptés
  identiques), 6 fixes appliqués à `replay_single_turn_plans.py`/`llm_chat.py`
  (parsing JSON tolérant, budget extraction 2048, raw sauvé sur échec, cache
  qui retente les erreurs, mode `--blind`, warnings de troncature).
- **Hypothèse troncature réfutée** : les erreurs d'extraction sont de la
  dégénérescence en boucle (échecs légitimes) → 20/100 (3B) et 54/100 (4B)
  confirmés.
- **Ablation extracteur aveugle** : 3B 12/100, 4B 46/100 → encadrement du
  « reasoning pur » : 3B ∈ [12,20], 4B ∈ [46,54] ; la dépendance à
  l'extracteur-solveur est 40 % (3B) vs 15 % (4B).
- **Cette étude température + best58** (§3-4).

Runs : `exp16_3b_temp1`, `exp16_blind_3b_temp1`, `exp16_4b_temp1`,
`exp16_best58_temp07`, `exp16_blind_best58_temp07`, `exp16_best58_temp1`,
`exp16_blind_best58_temp1`.

---

## 6. Balayage fin de température T=0.0→1.0 (2026-07-09) — RÉFÉRENCE

Objectif : trancher §4 (effet température) et §4ter (transfert RL) avec un
échantillonnage propre au lieu de points isolés. Reasoning-pur informé, 1 tirage
par température, pas de 0.1. base 3B et best58 complets (11 T) ; Qwen3.5-4B
partiel T=0.0→0.6 (backend HF ~30 min/config, arrêté pour libérer le GPU).
Runs : `runs/exp16_sweep_{base,b58,4b}_t00…t10/`.

| T | base 3B | best58 | 4B (partiel) |
|---:|---:|---:|---:|
| 0.0 | 11 | 21 | 56 |
| 0.1 | 15 | 15 | 54 |
| 0.2 | 15 | 18 | 54 |
| 0.3 | 11 | 24 | 52 |
| 0.4 | 11 | 18 | 52 |
| 0.5 | 14 | 24 | 52 |
| 0.6 | 11 | 18 | 57 |
| 0.7 | 10 | 21 | — |
| 0.8 | 13 | 18 | — |
| 0.9 | 14 | 20 | — |
| 1.0 | 10 | 17 | — |
| **moy** | **12.3** | **19.5** | **53.9** (T≤0.6) |
| σ / min-max | 1.9 / 10-15 | 2.7 / 15-24 | 1.9 / 52-57 |

### Conclusion 1 — aucun effet de température détectable (corrige §4)
Chaque modèle reste dans une bande de bruit plate de 0 à 1 (base 10-15, best58
15-24, 4B 52-57), sans tendance ; le greedy (T=0) n'est ni meilleur ni pire.
La « pénalité −5/−6 de 0.7→1.0 » du §4 venait de comparer deux tirages isolés
(base 0.7=20 était chanceux ; refait ici = 10). **Un pass@1/100 mono-tirage a
~±5 de bruit run-à-run**, trop pour résoudre un pas de 0,1 en température.

### Conclusion 2 — le RL a transféré à la planification (corrige §4ter)
best58 (moy 19.5) > base (moy 12.3) à **10 configs/11** (1 égalité, 0 défaite),
écart moyen **+7.2**, **test des signes p=0.002**. L'ancien « 19≈20 » comparait
un tirage haut de la base à un tirage bas de best58. Transfert **réel mais
modeste** : +7 en single-turn vs +36 en multi-tour (18→54) → le gain GRPO reste
surtout réactif, mais pas *nul* en planification.

### Note 4B
Partiel (7/11) mais déjà informatif : ~54 de moyenne, même bande plate. Balayage
arrêté à T=0.6 pour libérer le GPU (nouvelle expé). À compléter (T=0.7→1.0) plus tard.

### Meta-leçon méthodo
Le bruit mono-tirage (~±5 sur 100 items) fausse toute lecture d'un point unique :
best sélectionné (58 vs 54 re-éval), comparaison de températures, comparaison de
deux modèles proches. Remède pour les prochaines comparaisons fines : multi-seed
ou pass@k, pas un tirage isolé.
