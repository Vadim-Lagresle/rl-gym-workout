# Session 2026-07-16/17 — exp18 : few-shot prompting (éval pure)

Référence complète : `docs/RESULTS.md` §Exp 18 · artefacts : `runs/0_baselines/exp18_fewshot/`
(config.yaml = protocole + verdict, un sous-dossier par k).

## 1. Question et dispositif

Combien de points le **prompt seul** apporte-t-il au Qwen2.5-3B base, sans toucher aux poids ?
On injecte k exemples de tâches résolues dans le prompt, k ∈ {1, 3, 5, 10, 20, 30, 50}.

**Source des exemples — zéro contamination.** L'univers TextCraft compte 544 items
craftables (140 d1, 280 d2, 113 d3, 11 d4) ; train (374) + test (100) sont disjoints et
n'en couvrent que 474. Les **70 restants** (0 d1 / 18 d2 / 45 d3 / 7 d4) servent
d'exemples. Chaque solution est dérivée de l'arbre de recettes (logique min-depth de
l'env) puis **validée par replay dans un TextCraftEnv réel : 70/70 reward=1**
(`src/utils/build_fewshot_examples.py` → `data/eval/textcraft_fewshot_examples.json`).
Ordre déterministe entrelacé d2,d3,d2,d3,d4 à préfixes emboîtés (exemples(3) ⊂ exemples(10)).

## 2. Résultat principal (format dialogue)

| k exemples | 0 (réf) | 1 | 3 | **5** | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| Pass@1 /100 | 18 | 25 | 30 | **31** | 30 | 30 | 20 | 22* |

*k=50 : 99 items (1 crash isolé). Par depth :

| k | d1 (/31) | d2 (/41) | d3 (/25) | d4 (/3) |
|---|---|---|---|---|
| 0 | 13 | 5 | 0 | 0 |
| 5 | **25** | 5 | 1 | 0 |
| 20 | 20 | **10** | 0 | 0 |

Lectures :
1. **+13 pts sans training** (18 → 31, +72 % relatif) — autant que les premiers runs
   GRPO full-FT (exp7.x : 15-22 après des centaines de steps). Toute comparaison
   future d'un run RL devrait se faire aussi contre cette baseline promptée.
2. Le gain est un gain de **fiabilité d1-d2** ; **depth ≥ 3 reste à 0-1** quel que
   soit k → le mur de capacité (diagnostic oracle) n'est pas contournable par prompting.
3. Courbe en cloche : plateau k=3-20 (~30), chute à k≥30 (8-14k tokens d'exemples,
   dilution du contexte). **k=5 = point d'équilibre** (1 055 tokens).

## 3. Ablation involontaire : bloc vs dialogue (leçon de prompt engineering)

Première version : exemples collés en **bloc texte** dans le message de règles
(paires `Action:`/`Observation:` enchaînées). Résultat : **5/100 (k=1), 8/100 (k=3)**
— PIRE que zéro exemple. Diagnostic sur transcripts : le modèle imite le motif et
**hallucine lui-même les observations de l'env** dans ses messages (~740 messages
multi-actions, 228 sans action) → désynchronisation complète.

Correctif : les exemples deviennent de **vrais tours user/assistant** (format ReAct,
un message assistant = un Thought + UNE action). Même contenu, +18 à +26 pts d'écart
entre les deux formats. Témoins conservés : `exp18_fewshot/format_bloc_k01/k03`.

> **Règle à retenir : les démonstrations d'agent s'injectent en tours de dialogue,
> jamais en bloc texte.**

## 4. ⚠ Perte constatée : le best58 n'est plus reconstructible (2026-07-17)

En voulant refaire l'expérience avec le meilleur modèle RL (exp10.8, 54/100 re-éval) :
- le modèle mergé `qwen25_3b_exp10p8_step368_58pct` vivait sur un scratchpad /tmp
  (volatil) — disparu ;
- les `_best` de toute la lignée sont des **adapters seuls**, chacun ancré sur le
  merge précédent : 10.8 → `…exp10p7_step92_53pct` → 10.7 → `…exp10p5_step368_51pct`
  → 10.5 → `…exp10p3_step414_35pct` (exp10.3) ;
- **le maillon initial (merge 35 % d'exp10.3) a été supprimé de `models/` et
  l'adapter exp10.3 n'a jamais été conservé** → chaîne brisée, aucune copie nulle
  part (home, /tmp, overlays, backups vérifiés).

Conséquence : les études « best58 + few-shot » et « oracle best58 » sont impossibles
en l'état. **Leçon d'hygiène : après chaque merge de best, copier le modèle fusionné
sur le disque persistant (ou pousser les adapters + base de CHAQUE maillon), sinon le
best du projet ne survit pas à un nettoyage de /tmp.** Si un backup externe du 35 %
(ou de n'importe quel maillon) réapparaît, la reconstruction est scriptée
(`src/utils/merge_lora.py`, 3 merges).

## 5. Pass@10 few-shot (oracle N=10 × 100 items, base 3B) — LE mur d3 se fissure

`eval_oracle.py` accepte désormais `--fewshot` (mêmes exemples, format dialogue).
Artefacts : `exp18_fewshot/oracle_base_k05` et `oracle_base_k20` (passes.jsonl + JSON).

| Config | pass@1 | pass@10 | d1@10 | d2@10 | d3@10 | d4@10 |
|---|---|---|---|---|---|---|
| zero-shot (réf. N=20) | 10.3 % | 37.8 % | 84 % | 29 % | **0 %** | 0 % |
| few-shot k=5 | 27.7 % | 60.0 % | 100 % | 66 % | 8 % | 0 % |
| few-shot k=20 | 27.0 % | **61.0 %** | 100 % | 63 % | **16 %** | 0 % |

Lectures :
1. **pass@10 : +23 pts** (37.8 → 60-61 %). d1 sature à 100 %, d2 double (29 → 66 %).
2. **Résultat central : depth 3 passe de 0 % (même à pass@20 zero-shot !) à 8-16 %
   pass@10.** Le « mur de capacité d3 » du diagnostic oracle était donc en partie
   un artefact du prompt zero-shot : avec des démonstrations, le modèle échantillonne
   parfois des solutions d3 (~2-4 items /25). **Il existe désormais des graines GRPO
   à depth 3** — la condition qui manquait à tout le programme RL sur d3.
3. k=20 > k=5 pour l'exploration profonde (d3 16 % vs 8 %) alors que leurs pass@1
   sont équivalents : les exemples supplémentaires n'aident pas la fiabilité mais
   élargissent la couverture. d4 reste à 0 (3 items seulement).
4. Implication directe pour la suite : **RL initialisé avec prompt few-shot**
   (k≈5-20) — départ pass@1 ~28-31 au lieu de 18, signal non nul à d3, et l'écart
   pass@1 ↔ pass@10 (28 → 60) est exactement la marge que GRPO sait convertir.

## 6. À refaire quand un backup du best58 réapparaît (demande Vadim, bloquée)

Sweep few-shot pass@1 du best58 pour k ∈ {3, 8, 12, 17, 20, 25, 30} + oracle
pass@10 (zero-shot et meilleur k). Tout est prêt (`--fewshot` sur les deux
scripts d'éval) — il ne manque que le modèle (cf. §4).
