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
| `exp23_verl_repro` | recette verl du 75 traduite au plus près sur TRL : full-FT, LR 1e-6 + beta 0.001 constants, VRAI PPO (buffer 256 traj → 4 updates clippées eps 0.2, `steps_per_generation`), bonus entropie 0.001, max_rounds 30 au train, zero-shot, 30 epochs | en cours (2026-08-05) | teste si le mur ~0.47-0.50 vient du principe LoRA |

## 3_scalinginter/ — curriculum sur le budget d'interaction (max_rounds)

| Run | Schedule | Statut | Pass@1 |
|---|---|---|---|
| `exp8_scalinginter_b200` | step-based 10→20→30, N=16 — `eval_logs/` = ckpt200 (ex-dossier `_ckpt200`, fusionné) | incomplet (mort step 214/300) | 19/100 @ckpt200 |
| `exp8.1_scalinginter_b200` | epoch-based 6→11→…→25, N=16, 10 ep | config-only (`results: ~`) | — |
| `exp8.2_paper_batch_b200` | batch papier 256 traj/step, N=8, 20 ep | terminé (résultats dans config, pas d'eval_logs) | best 20/100 @step187 |

## 4_curriculum/ — curriculum de difficulté par depth

| Run | Contenu | Statut | Pass@1 |
|---|---|---|---|
| `exp9_curriculum_depth` | 4 stages d1→d2→d3→d3+4 (`run_curriculum_staged.sh`) — config + eval_logs fusionnés (ex-`_final`) | référence | **14/100** (régression) |

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

## 11_reasoning_rl/ — RL single-turn « pure reasoning » (plan complet en une complétion)

Le modèle émet en UNE complétion le raisonnement + la séquence d'actions complète
(`--plan-mode`, `src/train/rollout_plan.py`) ; la séquence est parsée par regex
(pas d'extracteur LLM, contrairement à exp16) et rejouée telle quelle dans l'env.
Éval périodique single-turn (même protocole que le training).

| Run | Méthode | Statut | Best |
|---|---|---|---|
| `exp21_pure_reasoning` | GRPO LoRA depuis zéro, k=10 single-turn, LR 7e-6 + paliers ÷3 / 3 epochs, 21 epochs | terminé (966 steps, pas de collapse) | **37/100 single-turn** (re-éval best step 800 ; pic périodique 47 bruité) · **9/100 multi-tour** (interférence : ne sait plus faire 1 action/tour) — d3-d4 inchangés (4 %/0 %) |

## 9_legacy_pre_b200/ — ère A100/verl et premiers essais TRL (ne pas réutiliser)

| Run | Stack | Pass@1 |
|---|---|---|
| `exp2_grpo_v2` | TRL LoRA, reward shaping bugué | 18→14 |
| `exp3_grpo_v3` | TRL LoRA, fix v2 (régresse) | 8 |
| `exp4_grpo_v4_scalinginter` | TRL LoRA + ScalingInter sparse | 14 |
| `exp6_verl_4gpu` | verl 4×A100 full-FT (+ courbes PNG dans le dossier) | **38** (meilleur pré-B200) |
| `prototypes/` | scripts d'exploration archivés | — |

---

## Repères transverses

| Référence | Valeur |
|---|---|
| Baseline Qwen2.5-3B | 18/100 |
| **Best du projet** (exp10.8 mergé, re-éval) | **54/100** |
| Papier AgentGym-RL-3B (objectif) | 75/100 |
| Qwen3.5-4B sans training | 77/100 |
| Gemini 3.5 Flash | 99/100 |
