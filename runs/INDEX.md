# Index des runs

Registre expérimental versionné. **Aucun run n'est supprimé.** Depuis 2026-07-16,
les runs sont classés par **famille de méthode** (dossiers numérotés dans l'ordre
de l'histoire du projet). Les chiffres de référence détaillés et leur analyse sont
dans [`../docs/RESULTS.md`](../docs/RESULTS.md).

Conventions de contenu d'un run :
- **entraînement** : `config.yaml` (hyperparamètres + courbe d'éval + verdict) ; les checkpoints vivent dans `saves/` (gitignoré)
- **éval multi-tour** : `eval_logs/` (100 JSON) + souvent `analysis.txt/csv`
- **exp16 single-turn** : `plans/` (phase 1) + `replay_logs/` (phase 2)
- **oracle** : `oracle_passk.json` et/ou `passes.jsonl` + plots

Statuts : **référence** (comparable, à jour) · **config-only** (jamais lancé) ·
**incomplet** (interrompu) · **legacy** (conservé pour traçabilité, ne pas réutiliser).

---

## 0_baselines/ — modèles évalués sans training (multi-tour, 100 items)

| Run | Modèle | Statut | Pass@1 |
|---|---|---|---|
| `exp1_baseline` | Qwen2.5-3B-Instruct | référence | **18/100** |
| `exp15_qwen35_4b` | Qwen3.5-4B (HF `--no-thinking`, vLLM incompatible) | référence | **77/100** (> papier sans RL) |
| `exp10_qwen0.5b_baseline` | Qwen2.5-0.5B — ⚠ malgré le nom, tentative GRPO full-FT | référence | **0/100** (trop petit pour apprendre) |
| `exp11_llama1b_baseline` | Llama-3.2-1B | config-only | — |
| `exp12_smollm2_baseline` | SmolLM2-1.7B | config-only | — |
| `exp13_gemma3_baseline` | Gemma-3-1B | config-only | — |
| `exp14_deepseekr1_baseline` | DeepSeek-R1-Distill-Qwen-1.5B | config-only | — |
| `exp_baseline_vllm_test` | smoke test 10 items | local, gitignoré | 2/10 |
| `exp18_fewshot/` (k01…k50 + format_bloc_*) | Qwen2.5-3B + k exemples held-out (format dialogue) | référence | **31/100 @ k=5** ; pass@10 **61%** @ k=20 (d3 : 0→16%) — courbes dans son config.yaml |

Les 4 config-only se lancent d'un coup via `src/train/run_benchmark_baselines.sh`.

## 1_api/ — bornes hautes SOTA via API externes

| Run | Modèle | Statut | Pass@1 |
|---|---|---|---|
| `exp_gemini_gemini_3_5_flash` | Gemini 3.5 Flash | référence | **99/100** |
| `exp_gemini_baseline` | Gemini 2.5 Flash | config-only | — |
| `exp_deepseek_v4_pro_max` | DeepSeek V4 Pro Max | config-only | — |
| `exp_kimi_k2_6` | Kimi K2.6 (Moonshot) | config-only | — |

## 2_grpo_fullft/ — réplication recette papier (GRPO full-FT, TRL+vLLM, B200)

| Run | Contenu | Statut | Pass@1 |
|---|---|---|---|
| `exp7_b200_fullft` | N=8, lr 1e-6, 564 steps | référence | 15/100 |
| `exp7.1_b200_fullft_12ep` | continuation → 12 epochs (4 488 steps) | référence | 16/100 |
| `exp7.1_ckpt1269` / `exp7.1_ckpt1598` | évals intermédiaires | checkpoint | 15 / 17 |
| `exp7.2_ckpt550` | run env_mask, incomplet | checkpoint | 15/100 |
| `exp7.3_ckpt400` | éval ckpt400 du run exp7.3_grpo_pur_we6 (le run d'entraînement n'a pas de dossier ici — best **32/100**, sert de modèle à `7_oracle/oracle_best32`) | checkpoint | 22/100 |
| `exp23_verl_repro` | recette verl du 75 traduite au plus près sur TRL : full-FT, LR 1e-6 + beta 0.001 constants, VRAI PPO (buffer 256 traj → 4 updates clippées eps 0.2, `steps_per_generation`), bonus entropie 0.001, max_rounds 30 au train, zero-shot, 30 epochs | terminé (2026-08-05) — **best 40/100** @ ep ~27, courbe test ENCORE ascendante à l'arrêt | apprentissage sain mais sous les runs LoRA few-shot (49-54) en 30 ep → budget prolongé dans exp23.1 |
| `exp23.1_verl_100ep` | exp23 identique prolongée à **100 epochs** (~4400 steps) + nouveau `--save-best-optimizer` : optimizer.pt (8.1 Go, Adam 8-bit) sauvé avec chaque best test sur le home | mort step 4236/4400 (epoch 96, purge /tmp week-end) — **best 58/100 @ ep 87.6, NOUVEAU BEST DU PROJET**, montée quasi monotone sans collapse KL, plateau bruité 48-58 sur les 15 dernières évals | budget x3.3 suffit à dépasser 54-58 (LoRA) ; reste à re-confirmer le 58 indépendamment |
| `exp23.2_warmstart58_50ep` | continuation du best 58 d'exp23.1 pour 50 epochs via NOUVEAU `--warm-start-dir` : poids du best injectés dans la politique, **ancre KL restée Qwen nu** ; incident : Adam réinitialisé (optimizer.pt supprimé avant chargement) | mort step 1292/2200 (epoch ~29, purge /tmp) — **best 64/100 @ ep 25.6, NOUVEAU BEST DU PROJET** (+6 vs parent), sauvé avec optimizer.pt ; dernières évals 54-64 encore vivantes | le "plateau" 48-58 d'exp23.1 n'en était PAS un — la rampe continue |
| `exp23.3_warmstart64_50ep` | continuation du best 64 d'exp23.2, 50 epochs — cette fois poids ET moments Adam chargés (vérifié dans le log AVANT le ménage disque) ; ancre KL = Qwen nu (KL 0.070 dès step 1) ; best seulement si >64 | terminé (2026-08-10, 2200/2200 steps, fin propre) — **max 63, jamais ≥64 → AUCUN best écrit** ; démarrage 49-56 malgré Adam chargé, fin 55-63 | le 64 était un pic bruité (évals voisines 54-59) : le creux post-restart = régression vers le vrai niveau, PAS un problème d'Adam ; vrai niveau ~58→~60 en 50 ep — rampe réelle mais lente ; leçon : seuil best trop haut = 50 epochs sans persistance |
| `exp23.4_final2200_150ep` | continuation du modèle FINAL d'exp23.3 (step 2200, /tmp non purgé — pas le pic 64), **150 epochs** (~6600 steps, ~60 h), Adam neuf (aucun état disponible) ; ancre KL = Qwen nu ; **seuil best abaissé à 0.55** pour toujours avoir un point de reprise best+optimizer sur le home | mort step ~3170/6600 (epoch ~72, purge pod nuit 10-11/08) — **best 69/100 @ ep ~69.4, NOUVEAU BEST DU PROJET**, sauvé complet (poids + optimizer.pt) ; dernières évals 56-69, toujours ascendant sous fort bruit | lignée exp23 depuis scratch : ~247 epochs cumulées → 69/100 (papier : 75) ; le seuil 0.55 a rendu la purge indolore ; suite = grille LoRA exp24 |

## 3_scalinginter/ — curriculum sur le budget d'interaction (max_rounds)

| Run | Schedule | Statut | Pass@1 |
|---|---|---|---|
| `exp8_scalinginter_b200` | step-based 10→20→30, N=16 — `eval_logs/` = ckpt200 (ex-dossier `_ckpt200`, fusionné) | incomplet (mort step 214/300) | 19/100 @ckpt200 |
| `exp8.1_scalinginter_b200` | epoch-based 6→11→…→25, N=16, 10 ep | config-only (`results: ~`) | — |
| `exp8.2_paper_batch_b200` | batch papier 256 traj/step, N=8, 20 ep | terminé (résultats dans config, pas d'eval_logs) | best 20/100 @step187 |
| `exp32_horizon` | PHASE 2 curriculum horizon : recette exp25 (r8/α32, LR 3e-6, β0.01, ancre mobile /4 ep) + horizon progressif `--max-rounds-schedule-epochs '10:0,20:15,30:30'` + **N=16** rollouts/prompt à 64 traj/step constant (4 prompts/step) + `--vllm-gpu-util 0.5` (verdict bench exp25.1), départ Qwen nu, 200 ep (arrêt manuel) | **terminé 26/08 17:17 (arrêt au plateau, ep ~80/200 — N=16 : 93 steps/epoch —, 2 j 00 h GPU)** — **82/100 (steps 6110 et 7238) : RECORD DU PROJET, objectif papier 75 DÉPASSÉ** ; montée 67 @ ep ~29 → 82 @ ep ~66, plateau final 75-82, 20 ré-ancrages sans collapse, ~23 s/step (épisodes courts du début de curriculum). ⚠ double delta vs exp25 (horizon ET N) → exp36 isole N. Best+optim, ckpt-7426 et 20 ancres sur le home. (v1 job 43, paliers 0/8/20 N=8, avortée le 24/08 avant la 1re éval) | **82/100** |

## 4_curriculum/ — curriculum de difficulté par depth

| Run | Contenu | Statut | Pass@1 |
|---|---|---|---|
| `exp9_curriculum_depth` | 4 stages d1→d2→d3→d3+4 (`run_curriculum_staged.sh`) — config + eval_logs fusionnés (ex-`_final`) | référence | **14/100** (régression) |
| `exp33_depth` | PHASE 2 : recette exp25 + N=16 + paliers depth CALENDAIRES `'1:0,2:12,3:25,4:37'` (sampler pondéré, dataset entier), 80 ep | **morte purge pod 28/08 ~01:48 (ep ~22)** — leçon double : 8 ep PERDUES à reward=1.0 saturé sur d<=1 (aucun gradient, illustration empirique de la limite des paliers statiques → rapport §4.3), puis STAGNATION sur d<=2 (reward médian figé 0.25 pendant 10 ep, éval 26-29, entropie saine 0.3→0.49). Non relancée telle quelle → exp33.1. Best 31 @ step 282 + ancres sur le home | 31/100† |
| `exp33.1_depth_auto` | idem exp33 mais paliers AUTO-DÉCLENCHÉS (règle 28/08) : palier suivant si reward train moyen 1 epoch ≥ 0.8, sinon cap 10 ep (`DepthAutoScheduleProvider`) | **morte purge pod 29/08 ~23:06 (ep ~44.5/80)** — la règle a fait exactement son travail : palier d≤2 dès l'ep 2.14 (succès, reward 0.801 — les 8 ep saturées d'exp33 économisées), d≤3 à 12.14 et d≤4 à 22.14 (caps), puis ~22 ep sur le dataset complet. **Best 72/100 @ step 4042** (+optimizer) + 11 cycles d'ancre sur le home. → reprise exp33.2 | **72/100** |
| `exp33.2_from72` | reprise d'exp33.1 depuis le best 72 reconstruit (Qwen3B ⊕ cycles ≤ step 4042 ⊕ adapter best → init + ancre, adapter r8 frais, pattern exp25.2), `--best-init-score 0.72`, **reprise AU PALIER 4** (`--depth-auto-start-stage 4`, décision finale Vadim 30/08 : continuité exacte du run tué, le curriculum avait fini sa gradation à l'ep 22.14), 40 ep ⚠ durée fixée à 40 ep par Claude SANS consulter Vadim (consigne = 80) — incident consigné, règle « tout demander » gravée en mémoire | **terminé 31/08 11:23 (fin propre 40/40 ep)** — **best 80/100 @ step 3290 (~ep 35)** (+optimizer), plateau final 74-75 sur les 5 dernières évals. Lignée depth-auto : 72 → **80**, à 2 pts du record exp32 (≈ égalité au bruit près). ⚠ Sémantique de reprise : ancre repositionnée sur le best, moments Adam à zéro (cf. session 30/08) | **80/100** |
| `exp34_magellan` | autocurriculum MAGELLAN en ligne (port complet : tête SR + adapters LoRA dédiés r16, sampler ∝ LP prédit, ε 1.0→0.2), recette exp25 N=16, 80 ep. 1er essai 31/08 : smoke ÉCHOUÉ (double bug du port : `CheckpointError` adapter r8/r16 sous gradient checkpointing + adapters SR jamais entraînés car `requires_grad=False` avant backward ; fix + tests 12, 4-6 recalibrés) ; rejouée job 47 le 02/09 | **tuée 03/09 08:47 (COLLAPSE)** : 15→**45 (step 611, ep ~6.5)** puis 32→18→0-2 dès step 940 (ep ~10). Même signature qu'exp36, plus tôt : entropie 0.94 (ep 2) → 0.66 (ep 6-7) → 0.19 (ep 7-8) → 0.01 ; KL 0.019 → 133 → 10¹²⁺. Sampler final p(d1)=0.79, p(d2)=0.18, p(d3)=0.03, **p(d4)=0.0006** : la prédiction pré-enregistrée « désinvestit d4 » est confirmée, MAIS le LP (\|SR−SR retardé\|) a explosé sur d1 quand le collapse a fait osciller les prédictions → concentration sur d1 → distribution de tâches rétrécie → collapse accéléré (rétroaction positive LP × instabilité). Lecture : l'autocurriculum suppose un apprenant stable ; il n'est pas une source d'entropie | 45/100 |
| `exp36.1_anchor12` | contrôle N=16 sans curriculum, 2e essai : recette exp36 à l'identique, SEUL delta ancre mobile /4 → /12 ep (référence plus ancienne, donc plus entropique sur la fenêtre critique ep 8-12), 80 ep | **en cours** (job 48, démarré automatiquement le 03/09 08:47 après le kill d'exp34). Pronostic écrit avant le verdict : les collapses G16 (exp36/34) sont entropie-d'abord à KL ~0.003, mécanisme que l'ancre ne voit pas → attendu : ne l'empêche pas, au mieux le retarde | — |
| `exp38_buffer256_n16` | OPTION A (décision Vadim 03/09) : mécanique de mise à jour du PAPIER (collecte 256 traj = 16 prompts × 16, 4 pas clippés ε0.2, `--steps-per-generation 256`) sur la recette LoRA + ancre mobile /4 — SEUL delta vs exp36 ; même nombre de pas (~93) et de trajectoires par epoch → compute égal ; pré-vol : smoke GPU buffer + ré-ancrage au milieu d'une collecte ; 80 ep | **en file** (job 49, après exp36.1). Question : la région de confiance du clipping et la collecte 4× plus large ralentissent-elles le sharpening G16 ? Signaux : `clip_ratio/*` ≠ 0 dès le 2e pas de chaque collecte, entropie, KL | — |
| `exp39_g16_8tasks` | DÉCONFONDRE G × tâches-par-pas : recette exp36 à l'identique (LoRA r8/α32, LR 3e-6, β0.01, ancre /4, G=16, 30 tours, sans curriculum), SEUL delta `--gradient-accumulation-steps 128` → 8 tâches × 16 = 128 traj/pas (46 pas/ép., comme G=8). exp25 = 8×8 stable, exp36 = 4×16 collapse ép. 10 : lequel des deux facteurs ? 3 lancements (purges Coder 04/09 11:08 et 19:45, disque plein 04/09 14:44), reprise exacte du ckpt-94 le 06/09 13:24 ; ckpt périodiques sur le HOME | **VERDICT 07/09 08:30 (ép. 22, pas 1016) : ÇA TIENT.** KL médiane 0.001 constante, entropie 1.05 → 0.93 (décroissance lente type G=8, pas la chute 1.0→0.59→0.10 d'exp36), pass@1 38→**49** (steps 893 et 940), aucun signe de collapse à l'époque où exp36 (ép. 10) et exp34 (ép. 7-8) étaient mortes. ⚠ FACTEUR CONFONDU relevé par Vadim 07/09 : l'ancre est définie en ÉPOQUES (4) → exp39 ré-ancre tous les 184 pas (comme exp25) contre 372 pour exp36. Même nombre de trajectoires par cycle (4×374×16) mais MOITIÉ de pas de gradient entre deux ancres, donc région de confiance plus serrée par pas. exp39 diffère donc d'exp36 sur DEUX points : 8 tâches/pas au lieu de 4, ET ré-ancrage 2× plus fréquent en pas. On ne peut PAS attribuer la stabilité au seul nombre de tâches. Run propre à faire : 8 tâches × 16 avec `--moving-anchor-every-epochs 8` (= 372 pas comme exp36). Plus lent que les curriculums à époque égale (49 vs ~60 à l'ép. 22) : le curriculum reste un accélérateur ET une protection ; la recette « 8 tâches/pas + ancre tous les 184 pas » protège aussi, sans qu'on sache lequel des deux facteurs agit. Run toujours en cours (80 ép. prévues, arrêt manuel) | 49/100 (en cours) |

## 5_lora_warmstart/ — la lignée du meilleur modèle du projet (LoRA + warm-start)

Lignée : exp10.3 (35 %) → 10.5 (51 %) → 10.7 (53 %) → **10.8 (54/100 re-éval, 58 = pic bruité)**.
Tous config-only (courbe d'éval dans le YAML, checkpoints dans `saves/`).

| Run | Base | LR | Statut | Best |
|---|---|---|---|---|
| `exp10.5_resume35_lr_div1.5` | resume LoRA 35 % | 3.33e-6 | référence | 51/100 (step 368) puis collapse |
| `exp10.6_resume35_lr2.2e-6` | resume LoRA 35 % | 2.2e-6 | legacy (abandonné step ~42) | — |
| `exp10.7_warmstart51` | merge 51 % | 2.2e-6 | référence (interrompu step 163) | 53/100 (step 92) |
| `exp10.8_warmstart53_lr_div3` | merge 53 % | 7.33e-7 | **référence — best du projet** | 58 pic / **54 re-éval** |
| `verif_best58_multitour` | re-éval indépendante du best 10.8 mergé | référence | **54/100** (le chiffre honnête) |

## 6_snis/ — recombinaison SNIS des tours (rollout_func expérimentale)

| Run | Contenu | Statut | Best |
|---|---|---|---|
| `exp17_snis_v1` | GRPO+SNIS v1, M=32/G=8, full-FT | incomplet (EngineDeadError vLLM step ~53) | 15/100 |

## 7_oracle/ — pass@k best-of-N (marge exploitable par le RL)

| Run | Modèle | Contenu | Résultat |
|---|---|---|---|
| `oracle_baseline` | Qwen2.5-3B base, N=20 | `oracle_passk.json` | pass@1 10.3 %, pass@20 47 % |
| `oracle_best32` | best RL exp7.3 (32/100), N=20 | + **`ANALYSIS.md`** (lecture clé) + plots | pass@1 32 %, pass@20 65 % |
| `oracle_qwen35_4b` | Qwen3.5-4B, N=18 | `passes.jsonl` + plots | pass@1 ~80 % |
| `oracle_smoke`, `oracle_smoke_hf` | smoke tests 5 items | — | — |

Conclusion clé (ANALYSIS.md) : depth 2 = mur de *fiabilité* (RL-able, +60 pts de marge) ;
depth 3-4 = mur de *capacité* (pass@20 ≈ 0 → aucune graine pour GRPO).

## 8_single_turn_exp16/ — planification single-turn (« reasoning pur »)

41 dossiers organisés en `core/` (6 runs à température unique), `blind/` (5 ablations
extracteur aveugle), `sweep_base|b58|4b/` (30 dossiers de balayage T=0.0→1.0).
**Voir [`8_single_turn_exp16/README.md`](8_single_turn_exp16/README.md)** — indispensable :
les sweeps ont invalidé certains verdicts écrits dans les configs de `core/`.

Chiffres à retenir (extracteur informé) : 3B base ≈ 12 (moy sweep) · best58 ≈ 19.5
(transfert RL réel mais modeste, p=0.002) · 4B ≈ 54 · depth 4 = 0 partout.

## 10_fewshot_rl/ — RL initialisé par prompt few-shot (suite directe d'exp18)

| Run | Méthode | Statut | Best |
|---|---|---|---|
| `exp19_fewshot_rl_k10` | GRPO LoRA depuis base, k=10 exemples/rollout, LR 7.33e-7 | interrompu step 847 (infra) | **43/100** @ step 400 (départ 30) |
| `exp19.1_warmstart43` | warm-start merge exp19, LR 7.33e-7 | arrêté step ~580 (ablation LR : polissage ≈ immobile) | ~33-37 |
| `exp19.2_scratch_5e-6` | LR 5e-6 (régime exp10.3) + k=10, depuis zéro | arrêté step 557 (collapse KL ×3500 après le pic) | **45/100 @ step 300** (best honnête du projet côté few-shot) |
| `exp20_staged_lr_v2` | recette exp19.2 + paliers LR/beta ÷3 / 3 epochs, 21 epochs — 1er run long stack v2 | terminé (966 steps, pas de collapse) | **58/100 @ step 800** (re-éval indépendante à faire) |
| `exp22_reward_adaptive_lr` | recette exp20 mais LR ADAPTATIF au reward train (coupe ÷3 si moyenne roulante 1 ep sous le best pendant 3 ½-epochs), 100 epochs | arrêté step 759 (les coupes figeaient la dérive post-pic : test 49 → 22-26) | **49/100 @ step 200** |
| `exp22.1_restore_best` | exp22 + restore du BEST TRAIN à chaque coupe (adapter sauvé sur disque à chaque best de moyenne roulante, rechargé + reset Adam à la coupe) | mort step 2158 (purge /tmp) — restore OK mais 7 coupes sur BRUIT, LR≈0 dès epoch 28 | **53/100 @ step 850** |
| `exp22.2_rank64_median` | rang LoRA 16→64 + critère durci (réf médiane 5 checks, eps 0.02≈2σ, palier ≥3 ep, plancher 1e-7 avec arrêt, optimizer sauvé avec le best) | arrêté step 671 — LR 5e-6 trop agressif pour r=64 : spike KL ~12k epoch 3, test 47→25, train plafonné à 0.44 < plafond r=16 | **47/100 @ step 50** |
| `exp22.3_rank64_lr_div4` | ablation exp22.2 : SEUL le LR initial change, 5e-6 → 1.25e-6 (÷4) — départ stable pour tester l'hypothèse capacité r=64 | arrêté step 1322 (epoch ~28.7) — départ STABLE (zéro spike KL) mais test décroche après epoch ~11, train plafonne à 0.478 = niveau r=16 | **49/100 @ steps 350-500** |
| `exp22.4_rank64_staged_div2` | paliers CALENDAIRES LR+beta ÷2 / 4 epochs (coupe pendant la montée, pas après le décrochage), départ 1.25e-6, r=64, 24 epochs | arrêté step ~407 (epoch ~8.8) — même décrochage post-pic qu'exp22.3 : couper tôt ne suffit pas, la coupe consolide l'état courant (= la dérive) | **50/100 @ step 250** |
| `exp22.5_stage10ep_restore` | paliers calendaires LONGS (10 ep) ÷2 + RESTORE du best du palier — nouveau `StagedBestRestoreCallback` — **ZERO-SHOT** (bras 1 de l'ablation few-shot) | mort epoch ~27/50 (purge pod nuit ven→sam) — mécanique paliers+restore OK sur 2 frontières ; test 18 → pic 41, train best 0.369 | **41/100 @ steps 400-500** |
| `exp22.6_stage10ep_fewshot` | bras 2 de l'ablation : recette exp22.5 à l'identique + `--fewshot 10` (SEULE différence) | mort step 516/2300 (epoch 11.2, purge pod 03-05/08) — ablation lisible : few-shot 49 test / 0.468 train vs zero-shot 41 / 0.369 → les exemples valent ~+8 pts, et le mur train ~0.47 few-shot réapparaît | **49/100 @ step 250** |
| `exp37_fewshot_n16` | GRPO simple (pas de curriculum) + `--fewshot 10`, recette N=16 + ancre mobile /12 ep (alignée exp36.1), 80 ep | en file (job 49, ajouté 27/08, DERNIER de la file — si temps avant gel) — le few-shot amorce-t-il et tient-il la durée à N=16 ? Risque à surveiller : sur-imitation des recettes des exemples (`err_recipe_wrong`). Repli si non couru : phrase « amorce/accélère sans garantie » dans le rapport — **parquée 03/09** (`queue/skipped/49_exp37_fewshot_n16.sh`) au profit d'exp38 : plus le temps de l'exploiter avant la soutenance du 08/09 | — |

## 11_reasoning_rl/ — RL single-turn « pure reasoning » (plan complet en une complétion)

Le modèle émet en UNE complétion le raisonnement + la séquence d'actions complète
(`--plan-mode`, `src/train/rollout_plan.py`) ; la séquence est parsée par regex
(pas d'extracteur LLM, contrairement à exp16) et rejouée telle quelle dans l'env.
Éval périodique single-turn (même protocole que le training).

| Run | Méthode | Statut | Best |
|---|---|---|---|
| `exp21_pure_reasoning` | GRPO LoRA depuis zéro, k=10 single-turn, LR 7e-6 + paliers ÷3 / 3 epochs, 21 epochs | terminé (966 steps, pas de collapse) | **37/100 single-turn** (re-éval best step 800 ; pic périodique 47 bruité) · **9/100 multi-tour** (interférence : ne sait plus faire 1 action/tour) — d3-d4 inchangés (4 %/0 %) |

## 12_lora_grid/ — recherche d'hyperparamètres LoRA (blog « LoRA Without Regret »)

Grille rangs × (LR, beta) sur la recette exp23 (zero-shot, vrai PPO, buffer 256),
15 epochs/run (révisé 12/08, après le collapse du 1er run à 100 ep), α=32 fixe
(convention du blog, remplace notre α=2r).
Design et grille de lecture : `runs/12_lora_grid/exp24_grid/README.md`.
Exécution séquentielle par `scripts/run_queue.sh` (jobs dans `runs/queue/`).

| Run | r | LR | beta | Statut | Best |
|---|---|---|---|---|---|
| `exp24_r64_lr1e-5_b0.001` | 64 | 1e-5 | 0.001 | **collapse** — pic 24 @ step 94-141 puis 0/100 dès step 658, tué ep ~17 : le « 10× full-FT » du blog ne transfère pas sur notre RL multi-tour batch 256 | 24/100 |
| `exp24_r16/r32_lr1e-5_b0.001` | 16/32 | 1e-5 | 0.001 | parqués (`runs/queue/skipped/`) — même collapse attendu | — |
| `exp24_r64_lr3e-6_b0.001` | 64 | 3e-6 | 0.001 | **mort purge pod 12/08 @ ep 10.6** (GPU idle 2 j avant détection) — plateau 15→25→19, à peine > baseline 18, loin de la barre full-FT ~30 ; lecture acquise, non rejoué | 25/100 |
| `exp24_r16_lr3e-6_b0.001` | 16 | 3e-6 | 0.001 | terminé 15/08 — monte à **35** @ ep ~12 (**> barre full-FT 30-32**, meilleur run de la grille, best adapter sauvé) puis **collapse** (2-15/100 en fin de run) | 35/100 |
| `exp24_r32_lr3e-6_b0.001` | 32 | 3e-6 | 0.001 | terminé 15/08 — pic 31 @ ep ~6 puis **collapse** (2-16/100 sur toute la 2e moitié) | 31/100 |
| `exp24_r64_lr3e-6_b0.01` | 64 | 3e-6 | 0.01 | terminé 15/08 (fin propre) — **stable 21-29, fin 27, zéro collapse** : l'ancre β0.01 protège mais plafonne sous la barre full-FT | 29/100 |
| `exp24_r16_lr3e-6_b0.01` | 16 | 3e-6 | 0.01 | terminé 17/08 (rejeu complet) — montée monotone 7→39 (**39 @ ep ~7, RECORD LoRA zero-shot**, best sauvé), puis instabilité tardive (10 @ ep ~12, remonte 24-25) : β0.01 ne protège pas totalement r16 | 39/100 |
| `exp24_r32_lr3e-6_b0.01` | 32 | 3e-6 | 0.01 | parqué (`queue/skipped/`, 17/08 — priorité à exp25/exp26) | — |
| `exp24_r{64,16,32}_lr1e-6_b0.001` | 64/16/32 | 1e-6 | 0.001 | parqués (`queue/skipped/`) | — |

**Lecture au 17/08** : à chaleur α·LR égale (9.6e-5), β=0.001 monte plus haut (31-35)
mais collapse systématiquement ; β=0.01 ne collapse jamais mais plafonne (~27-29).
LoRA peut égaler le full-FT à budget égal (35 vs 30-32) si protégé du collapse.

Barre de comparaison : full-FT (exp23) ≈ 30-32/100 à epoch 15.
Vitesse saine : ~60-75 steps/h (15 ep ≈ 9-11 h) ; un collapse ralentit à ~30 st/h.

## 9_legacy_pre_b200/ — ère A100/verl et premiers essais TRL (ne pas réutiliser)

| Run | Stack | Pass@1 |
|---|---|---|
| `exp2_grpo_v2` | TRL LoRA, reward shaping bugué | 18→14 |
| `exp3_grpo_v3` | TRL LoRA, fix v2 (régresse) | 8 |
| `exp4_grpo_v4_scalinginter` | TRL LoRA + ScalingInter sparse | 14 |
| `exp6_verl_4gpu` | verl 4×A100 full-FT (+ courbes PNG dans le dossier) | **38** (meilleur pré-B200) |
| `prototypes/` | scripts d'exploration archivés | — |

---

## 13_moving_anchor/ — ancre KL mobile pour LoRA (merge-and-restart)

Hypothèse (lecture exp24) : l'ancre KL FIXE (Qwen nu) freine d'autant plus que la
politique progresse → plafond β0.01 ; l'ancre mobile re-base la référence sur la
politique courante toutes les N epochs (`--moving-anchor-every-epochs`,
`schedules.MovingAnchorCallback` — TRL `sync_ref_model` refuse PEFT).
Design : `runs/13_moving_anchor/exp25_r8_anchor4ep/config.yaml`.

| Run | Config | Statut | Best |
|---|---|---|---|
| `exp25_r8_anchor4ep` | r8/α32, LR 3e-6 constant, β 0.01, ré-ancrage /4 ep, batch 64 on-policy, 15 ep | terminé 18/08 (fin propre) — 12→**35 au step FINAL (658)**, encore ascendant à l'arrêt, zéro collapse, 3 ré-ancrages neutres (pas de déstabilisation post-ancre) ; égale le record de la grille (35) avec r8 et SANS l'instabilité | 35/100 |
| `exp25_r8_anchor4ep` (suite 200 ep) | reprise exacte du checkpoint-690 (adapter+optimizer, epoch 15) sur base ancrée cycle 3, `--moving-anchor-initial-cycle 3`, jusqu'à 200 epochs — toutes choses égales | mort purge /tmp 20/08 à l'epoch ~111.5 — montée 35→58→**65 (@ step 4794, RECORD LoRA du projet)**, 27 ré-ancrages sans collapse ; best (adapter) + chaîne d'ancres sauvés sur le home, optimizer perdu (le job ne passait pas `--save-best-optimizer` — règle corrigée le 20/08 pour tous les jobs suivants) | **65/100** |
| `exp25.2_from65` | reprise depuis le BEST 65 reconstruit (Qwen3B ⊕ cycles 1-26 ⊕ adapter best, merge → init + ancre KL), adapter r8 frais, moments Adam à zéro (purgés /4 ep de toute façon), `--best-init-score 0.65 --save-best-optimizer`, 60 epochs — objectif 70 | terminé 21/08 (fin propre, 60 ep, 20/08 16:44 → 21/08 12:36) — 66→**73 (steps 2491/2632/2726, RECORD du projet)**, plateau final 66-73, 15 ré-ancrages sans collapse ; best + optimizer + chaîne d'ancres sur le home | **73/100** |

### Phase 1 (plan 19/08) — ablation ancre × β (exp30/31/31.1, recette exp25 sinon)

| Run | Config (delta vs exp25) | Statut | Best |
|---|---|---|---|
| `exp30_r8_fixedanchor` | ancre FIXE (pas de `--moving-anchor-*`), β 0.01 | terminé (job 41, 21/08) — pic **39 @ step 329 (ep ~7)** puis décrochage 37→26 (collapse ep ~11-13) : sans ré-ancrage, β0.01 ne suffit pas | 39/100 |
| `exp31_r8_anchor_b0001` | ancre mobile /4 ep, β 0.001 | arrêté manuellement 24/08 à l'ep ~60/150 (job 42 → `skipped/`, redéfini en 31.1) — montée régulière jusqu'au plateau **49-53**, 15 ré-ancrages sans collapse : l'ancre mobile SAUVE le régime β0.001 (qui collapsait en ancre fixe, exp24) mais plafonne sous exp25 (β0.01 → 65-73) | 53/100 |
| `exp31.1_r8_fixedanchor_b00001` | ancre FIXE + β 0.0001 (KL quasi désactivée) | **arrêté 24/08 16:47 (divergence)** à l'ep ~9/200 (run 09:47→16:47) — 13→16→22→24→32→**34 (step 282, ep ~6)** puis DIVERGENCE KL runaway : kl 0.2 (ep 6) → 906 (ep 7) → 1572 (ep 8) → 2×10¹⁴ (ep 9), entropie 1.3→0.02, générations détruites (8k tokens/completion, 20 erreurs format/ep à l'éval). Best+optimizer et ckpt-376 sauvés sur le home | 34/100 |

| `exp36_n16` | recette exp25 STRICTE ('30:0', sans curriculum), SEUL delta **N=16** (4 prompts/step, 64 traj/step constant) + gpu-util 0.5, 80 ep | **tué 27/08 09:00 (COLLAPSE)** après 15 h 40 — montée saine 12→**54 (step 1222)** PLUS RAPIDE qu'exp25 à trajectoires égales, puis effondrement d'entropie ep 8-12 (1.0→0.10) et KL runaway 10⁸-10¹² malgré les ré-ancrages /4 ep (l'ancre suit la déterminisation au lieu de la freiner). **1er verdict du contrôle : N=16 seul ≠ le 82 d'exp32 — le curriculum d'horizon agit en RÉGULARISEUR d'exploration** (entropie exp32 stable 0.6-0.7 aux mêmes epochs). Best 54+optim et ckpt-1410 (post-collapse) sur le home | 54/100† |
| `exp36.1_anchor12` | idem exp36, SEUL delta : `--moving-anchor-every-epochs 4 → 12` — ancre plus ancienne = référence entropique sur toute la fenêtre critique (ep 8-12), la KL redevient un frein à la déterminisation ; β/entropy-coef inchangés (un seul delta, décision Vadim 27/08) | en file (job 48, FIN de file — passera après exp33/34/35 si le calendrier respire) — objectif : un contrôle N=16 sans curriculum qui SURVIT | — |

**Lecture phase 1 (24/08)** : le carré ancre × β est complet — fixe+β0.001 (exp24 : pic 35
puis collapse), fixe+β0.01 (exp30 : pic 39 puis collapse ep ~11), fixe+β0.0001 (exp31.1 :
divergence ep ~7-9), mobile+β0.001 (exp31 : 53 stable), mobile+β0.01 (exp25/25.2 : **73**
stable). **L'ancre mobile est nécessaire ET aucune valeur de β ne sauve l'ancre fixe** ;
β0.01 + mobile reste la recette. → Phase 2 (curriculums, exp32-35) lancée le 24/08.

## 14_fullft_7b/ — réplication de la recette exp23 sur Qwen2.5-7B

Vérifier les chiffres du papier sur le 7B + étude pass@20 test AVANT/APRÈS RL
(la frontière de support bouge-t-elle, ou le RL ne fait-il que fiabiliser ?).
Design : `runs/14_fullft_7b/exp26_7b_fullft_150ep/config.yaml`.

| Run | Config | Statut | Best |
|---|---|---|---|
| `oracle_qwen25_7b_base` | pass@20 test, 7B nu (borne avant) | terminé 18/08 — **pass@1 33.2 %, pass@20 68 %** ; par depth (@20) : d1 94, d2 76, d3 32, **d4 0/3 items** — le mur depth 4 persiste à 7B | — |
| `exp26_7b_fullft_150ep` | recette exp23, 150 ep | **TUÉ 18/08 (bug, pas un résultat)** : buffer 1 sain (reward train 0.40 !) puis génération détruite dès le buffer 2 (0 action valide, 14k tokens/traj, KL→10⁷, 828 s/step). Le smoke on-policy était sain → suspect : 1re sync de poids post-updates en mode buffer 256 sur le 7B (embeddings NON liés, contrairement au 3B). **Famille ABANDONNÉE le 18/08** (soutenance < 1 mois : priorité LoRA 3B + curriculum) — repro tué, artefacts 7B purgés | — |
| `oracle_exp26_7b_best` | pass@20 test, best exp26 (borne après) | abandonné avec la famille (18/08) | — |

---

## Repères transverses

| Référence | Valeur |
|---|---|
| Baseline Qwen2.5-3B | 18/100 |
| **Best du projet** (exp32 curriculum horizon + N=16, ancre mobile — DÉPASSE l'objectif papier) | **82/100** |
| Best sans curriculum (exp25.2 LoRA r8 + ancre mobile) | 73/100 |
| Best full-FT (exp23.4, lignée verl-sur-TRL ~247 epochs) | 69/100 |
| Best LoRA few-shot (exp10.8 mergé, re-éval) | 54/100 |
| Papier AgentGym-RL-3B (objectif) | 75/100 |
| Qwen3.5-4B sans training | 77/100 |
| Gemini 3.5 Flash | 99/100 |
