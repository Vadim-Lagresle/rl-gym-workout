# Plan d'expériences jusqu'à la soutenance (8 septembre)

> Rédigé le 19/08/2026. Une seule B200 → tout passe par la file `runs/queue/`
> (runner flock-protégé). Vitesse constatée : ~30-40 min/epoch LoRA r8-r16,
> ~46 steps/epoch → **~2 runs de 20 epochs par jour**.
>
> **Jalons durs** : gel des expériences GPU le **2/09** ; figures les 2-3/09 ;
> rédaction 2-7/09 ; soutenance 8/09.

## Recette de référence (héritée d'exp25, sauf mention contraire)

```
--lora-r 8 --lora-alpha 32
--num-generations 8 --gradient-accumulation-steps 64
--entropy-coef 0.001
--learning-rate 3e-6 --beta 0.01
--moving-anchor-every-epochs 4
--max-completion-length 512 --max-items 0
--max-rounds-schedule '30:0'
--vllm-max-len 32768 --eval-every 47 --eval-items 100
--save-steps 94 --save-total-limit 1
--output-root /tmp/trl_grpo_runs --use-vllm-inprocess
```

> **Amendement 24/08 (verdict bench exp25.1 appliqué, décision Vadim)** : tous
> les jobs de phase 2 (43b-47) passent à `--num-generations 16` (64 traj/step
> constant → 4 prompts/step) et `--vllm-gpu-util 0.5` — config D du bench,
> 46.9 s/it vs 57.1 pour N=8/0.17 (18 % plus rapide ET meilleure estimation
> d'avantage). La baseline exp25 reste N=8 → le contrôle homogène de la phase 2
> devient exp36 (ci-dessous).

---

## Phase 0 — exp25 : validation longue durée de l'ancre mobile (EN COURS)

| | |
|---|---|
| Run | `exp25_r8_anchor4ep` (reprise 200 ep, job 40) |
| Objectif | atteindre **60/100 en éval** pour valider le setup ; courbe de stabilité longue durée = figure maîtresse |
| Critère d'arrêt | 60 atteint (projection : ep ~90-120, **21-22/08**) OU point de décision le **23/08 matin** si plateau |
| Livrable | recette stabilisée pour toutes les phases suivantes + baseline "GRPO uniforme" de la phase 2 (courbe réutilisée telle quelle, coût zéro) |

## Phase 0.5 — exp25.1 : étude d'optimisation GPU (job 40b, ~1 h, juste après exp25)

Mesuré le 19/08 : un run LoRA n'utilise que 68 Go sur 183 (`--vllm-gpu-util
0.17` est un réglage fossile de l'époque full-FT) et le KV cache actuel ne
loge que ~35 des 64 épisodes concurrents. Bench de 5 configs × 8 steps réels
(gpu-util 0.17/0.4/0.6 ; N=8 vs N=16 à 64 traj/step constant) : s/it, KV,
préemptions. Design et règles de décision :
`runs/13_moving_anchor/exp25.1_gpu_throughput/README.md`.

**La conclusion de cette expé influence tout le reste du plan** : le gpu-util
retenu est appliqué aux jobs des phases 1-2, et si N=16 coûte < ~30 % de temps
à trajectoires constantes, un bras `exp36_n16` devient candidat prioritaire de
la phase 2 (meilleure estimation d'avantage GRPO et groupes mixtes plus
fréquents sur les items difficiles — réponse directe au plafonnement observé
sur exp25).

## Phase 1 — Ablation d'ancre (2 runs, lancement dès la fin d'exp25)

Isoler la contribution de l'ancre mobile vs le β. Recette de référence, 20 epochs.

> **Numérotation (décision 19/08)** : les expériences FINALES (post-exploration)
> commencent à **exp30**. exp30-31 = phase 1, exp32-35 = phase 2.

| Run | Delta vs référence | Question | Deadline fin |
|---|---|---|---|
| `exp30_r8_fixedanchor` | ancre FIXE (pas de `--moving-anchor-*`), β 0.01 | l'ancre mobile explique-t-elle la montée d'exp25, ou r8+β0.01 suffisait ? | **24/08** |
| `exp31_r8_anchor_b0001` | ancre mobile 4 ep, `--beta 0.001` | l'ancre mobile sauve-t-elle le régime β faible (qui collapsait en ancre fixe, exp24) ? | **25/08** |

Lecture : avec exp25 (mobile+0.01) et exp24 r16 (fixe+0.001 / fixe+0.01), on a
les 4 cases du carré ancre × β → figure 1 du rapport.

## Phase 2 — Curriculums et transfert (4 runs, le CŒUR)

Recette de référence (amendée du verdict phase 1 si besoin), 20 epochs chacun.
Baseline = exp25 (gratuite).

| Run | Delta vs référence | Question | Deadline fin |
|---|---|---|---|
| `exp32_horizon` | `--max-rounds-schedule '10:0,20:X,30:Y'` (paliers ScalingInter, X/Y à caler sur 20 ep) | l'horizon progressif accélère-t-il ? | **26/08** |
| `exp33_depth` | curriculum depth 1→4 par paliers calendaires | ordonner par difficulté aide-t-il le transfert d3-d4 ? | **morte purge 28/08 (ep ~22, stagnation)** → remplacée par exp33.1 |
| `exp33.1_depth_auto` (ajouté 28/08) | paliers depth AUTO-DÉCLENCHÉS : reward ≥ 0.8 sur 1 ep → palier suivant, sinon cap 10 ep | corrige les 2 défauts d'exp33 — VALIDÉ : d≤2 dès l'ep 2.14 au succès | **morte purge 29/08 (ep ~44.5), BEST 72** → exp33.2 |
| `exp33.2_from72` (ajouté 30/08) | reprise du best 72 (merge chaîne+best, pattern exp25.2), reprise AU PALIER 4 comme si le run n'avait pas été interrompu, 40 ep (job 44c) | consolider/dépasser 72 sur le dataset complet | dès relance file |
| `exp34a_auto_depth` | autocurriculum ALP par depth (EK-Online-ALP au sens MAGELLAN, voir `docs/MAGELLAN_ANALYSE.md`) : échantillonnage ∝ LP roulant par depth + plancher ε=0.2 — **à coder** (stats par depth dans `rollout.py` + sampler dans `data.py`) | la machine dose-t-elle mieux que le design humain ? prédiction : désinvestit d4 (support nul) | **28/08** |
| `exp34_magellan` (statut 31/08) | port complet en ligne — smoke GPU échoué (double bug : CheckpointError adapter/checkpointing + adapters SR jamais entraînés, tête seule) → FIX + selftest durci, **rejouée en job 47 après exp35**, 80 ep | inchangée | après exp35 |
| `exp34c_probe_offline` | sonde MAGELLAN OFFLINE (pas de GPU de course) : tête MLP de compétence entraînée sur les embeddings de checkpoints successifs d'exp25 + couples (tâche, succès) loggés ; partition induite vs partition par depth (détail : `docs/NOTES_RAPPORT.md` §curriculum machine) | la depth est-elle la vraie structure de compétence du modèle ? de-riske 34b | **27/08** (parallèle) |
| `exp34b_auto_learned` | sampler à LP APPRIS en ligne (MAGELLAN allégé : tête seule + embeddings de la politique courante, pas de 2e adapter) — **conditionnel** : si 34c concluant ET calendrier tenu après phase 1 | le curriculum découvert bat-il le curriculum par depth ? ouverture méta-learning | **31/08** (si go) |
| `exp35_budget1024` | `--max-completion-length 1024` (hyperparamètre supplémentaire demandé : budget de sortie par tour) | plus de place pour raisonner par tour aide-t-il depth 3 ? (lien test-time compute) | **29/08** |
| `exp36_n16` (ajouté 24/08) | recette exp25 STRICTE ('30:0', pas de curriculum), SEUL delta N=16 (+gpu-util 0.5) — job 43z, **promu juste après exp32** (décision Vadim 24/08 soir, vu la montée d'exp32) | effet propre de N (vs exp25 N=8) + lève le double-delta d'exp32 v2 (horizon ET N) ; on reste à N=16 : N=32 = 2 items/update à 64 traj/step, GA=128 doublerait le s/it | **fait — COLLAPSE 27/08 (54†)** → exp36.1 (ancre /12 ep, job 48) |
| `exp37_fewshot_n16` (ajouté 27/08) | GRPO simple + `--fewshot 10`, N=16, ancre /12 ep (alignée exp36.1) — job 49, DERNIER de la file | le few-shot amorce-t-il ET tient-il la durée à N=16 ? risque : sur-imitation des recettes des exemples ; repli si non couru : phrase « amorce sans garantie » | si temps (avant gel 2/09) |

Code à écrire AVANT (pendant qu'exp25 tourne, dry-run CPU obligatoire) :
sampler autocurriculum (design § MAGELLAN_ANALYSE.md), stats de succès par
depth, jobs 43-46 dans la file.

## Phase 3 — Analyse transfert et taxonomie d'erreurs (pas de GPU, en parallèle)

| Tâche | Contenu | Deadline |
|---|---|---|
| Métriques par depth | pass@1 par depth au fil des epochs pour chaque régime (relire les logs d'éval existants, `analyze_eval.py`) | 29/08 |
| Matrice de transfert | entraîné sur depth ≤ d → perf par depth d' (évals finales des 4 régimes + baseline) | 30/08 |
| Taxonomie d'erreurs | action invalide / ingrédient manquant / boucle / horizon dépassé, par depth × régime, comparée au GRPO sans curriculum | 31/08 |

## Phase 4 — Stretch : transfert vers UN autre environnement (décision le 29/08)

| | |
|---|---|
| Env candidat | BabyAI ou SciWorld (vérificateur exact, même pattern `agentenv-*`) |
| Contenu | adaptation rollout/reward (~1-2 j) + 2 runs courts : baseline vs meilleur curriculum |
| Go/no-go | **29/08** selon l'avancement des phases 1-2 ; fin au plus tard **1/09** |
| Si no-go | une ligne "en cours" honnête dans le rapport §5 |

## Phase 5 — SNIS guidée, analyse OFFLINE (pas de GPU, si temps)

| | |
|---|---|
| Contenu | sur les trajectoires déjà loggées : scorer les segments par LLM-as-judge (API), mesurer ESS et qualité des recombinaisons candidates — sans réentraîner |
| Fenêtre | 26/08 → 1/09, en parallèle des runs GPU |
| Livrable | §4.6 du rapport (piste exploratoire chiffrée, pas un résultat RL) |

## Remarque — « reprise exacte » après purge (backlog, ajouté 30/08)

Le warm-start actuel (merge chaîne+best → init + ancre, adapter frais) déplace
l'ancre KL sur le best et remet les moments Adam à zéro. La reprise fidèle :
base = cycles SEULS (ancre au dernier ré-ancrage), adapter best INJECTÉ
(`set_peft_model_state_dict`, mécanique du restore intra-run), `optimizer.pt`
chargé (moments cohérents avec l'adapter injecté), `--moving-anchor-initial-cycle`
calé. Toutes les briques existent ; manque le câblage au lancement pour LoRA.
À implémenter si temps — détail : session hebdo 30/08.

## Remarque — transfert mono-tour → multi-tour (backlog, non planifié)

Étude de transfert single-turn → multi-turn (lignée Plan-Mode/exp21 et pipeline
exp16 `src/eval/single_turn/`) : pré-alignement RL en plan-mode (rollout unique,
pas d'allers-retours serveur, donc rapide) puis mesure de ce qui se transfère au
contrôle interactif — et le sens inverse (le RL multi-tour améliore-t-il la
planification mono-tour ? acquis partiel : +7 pts, à consolider). **La priorité
reste le curriculum (phases 1-2)** ; on ne planifie cette étude que si le
curriculum avance bien — décision au même jalon que la phase 4 (29/08).

## Déjà acquis (rédaction seule, aucun run à refaire)

- **Oracle pass@k par depth** : 3B et 7B nus, d4 = 0 % @20 → figure "mur de capacité".
- **Few-shot** : lignée exp10.8/exp22 (gain, format, amorçage).
- **Inter-générations** : Qwen3.5-4B zero-training > objectif papier.
- **7B full-FT** : abandonné (bug sync vLLM / embeddings non liés) → une phrase en annexe C.
- **Grille LoRA exp24** : les 2 cases "ancre fixe" du carré de la phase 1.

## Calendrier récapitulatif

```
19-21/08  exp25 grimpe | coder : autocurriculum, jobs 41-44, dry-runs CPU
21-23/08  exp25 atteint 60 (ou décision 23/08 matin) → phase 1 (exp30-31)
24-25/08  fin phase 1 → lancement phase 2 (exp32-35, file automatique)
26-29/08  phase 2 | en parallèle : phase 3 (analyse) + phase 5 (SNIS offline)
29/08     go/no-go phase 4 (2e environnement)
30/08-1/09 stretch phase 4 | bouclage phases 3 et 5
2/09      GEL GPU — dernières évals, figures
2-7/09    rédaction (plan : docs/PLAN_RAPPORT.md)
8/09      soutenance
```

Marge : ~1,5 jour de GPU non alloué entre le 29/08 et le 2/09 (absorbe un
crash, un rejeu, ou la phase 4).

## Ajout 03/09 (gel) — premier run à lancer si le GPU redevient disponible
**exp39_g16_8tasks** : recette exp36 (LoRA r8/α32, LR 3e-6, β0.01, ancre /4, G=16, 30 tours, sans
curriculum) avec `--gradient-accumulation-steps 128` → 8 tâches × 16 = 128 traj/pas, 47 pas/époque.
Question : le collapse d'exp36 vient-il de G=16 ou du passage de 8 à 4 tâches par pas ? Compare à
exp25 (8×8, stable 110 ép.) et exp36 (4×16, collapse ép. 10). Verdict en 10-15 époques.
