# Index des runs

Statut de chaque sous-dossier de `runs/`. **Aucun run n'est supprimé** : ce dossier est le
registre expérimental versionné. Les chiffres Pass@1 de référence sont dans
[`../docs/RESULTS.md`](../docs/RESULTS.md).

Légende statut :
- **actif/référence** — run valide, utilisé pour comparaison
- **checkpoint** — eval d'un checkpoint intermédiaire (pas de `config.yaml` propre)
- **legacy/bugué** — conservé pour traçabilité, ne pas réutiliser tel quel
- **config-only** — `config.yaml` présent mais **jamais lancé** (0 eval_logs)
- **prototype** — scripts d'exploration archivés

| Run | Statut | Stack | Note |
|---|---|---|---|
| `exp1_baseline` | actif/référence | vLLM (eval_baseline.py) | Baseline Qwen2.5-3B sans training |
| `exp7_b200_fullft` | actif/référence | TRL+vLLM B200 | Réplication recette papier, full FT |
| `exp7.1_b200_fullft_12ep` | actif/référence | TRL+vLLM B200 | Continuation exp7 → 12 epochs |
| `exp10_qwen0.5b_baseline` | actif/référence | TRL+vLLM | Itération rapide petit modèle (0.5B) |
| `exp10.5_resume35_lr_div1.5` | actif/référence | TRL+vLLM B200 | Reprise LoRA depuis 35% (exp10.3), LR/1.5 → **best 51/100** (step 368), puis collapse |
| `exp10.6_resume35_lr2.2e-6` | legacy | TRL+vLLM B200 | Variante LR 2.2e-6 depuis 35% — **abandonné step ~42** au profit du warm-start |
| `exp10.7_warmstart51` | actif/référence | TRL+vLLM B200 | Warm-start depuis 51% (merge exp10.5) → **best 53/100** (step 92), interrompu step 163 |
| `exp10.8_warmstart53_lr_div3` | actif/référence | TRL+vLLM B200 | Warm-start depuis 53% (merge exp10.7), LR/3 (7.33e-7) → best step 368 : **58/100 (pic éval train), 54/100 (re-éval indépendante)**, run complet 400 steps |
| `exp16_blind_extraction_3b` | actif/référence | vLLM (replay --blind) | Ablation extracteur aveugle exp16 3B → **12/100** (vs 20 informé) |
| `exp16_blind_extraction_4b` | actif/référence | HF (replay --blind) | Ablation extracteur aveugle exp16 4B → **46/100** (vs 54 informé) |
| `exp_gemini_gemini_3_5_flash` | actif/référence | API Gemini (eval_gemini.py) | Borne haute SOTA (API externe) |
| `exp7.1_ckpt1269` | checkpoint | TRL+vLLM | Eval intermédiaire (step 1269) — voir exp7.1_b200_fullft_12ep |
| `exp7.1_ckpt1598` | checkpoint | TRL+vLLM | Eval intermédiaire (step 1598) |
| `exp7.2_ckpt550` | checkpoint | TRL+vLLM | Eval partielle (env_mask), run incomplet |
| `exp2_grpo_v2` | legacy/bugué | TRL | Reward shaping bugué (count_actions sur completion concaténée) |
| `exp3_grpo_v3` | legacy/bugué | TRL | Tentative fix v2, régression supplémentaire |
| `exp4_grpo_v4_scalinginter` | legacy | TRL | ScalingInter sparse 0/1, remplacé par exp7-8 |
| `exp6_verl_4gpu` | legacy (verl) | verl 4×A100 | Framework abandonné — courbes : `training_curves.png`, `training_losses.png` |
| `exp8_scalinginter_b200` | config-only | TRL+vLLM | Planifié, jamais lancé |
| `exp9_curriculum_depth` | config-only | TRL+vLLM | Planifié (nécessite textcraft_train_with_depth.json) |
| `exp11_llama1b_baseline` | config-only | benchmark | Baseline Llama-3.2-1B, jamais lancé |
| `exp12_smollm2_baseline` | config-only | benchmark | Baseline SmolLM2-1.7B, jamais lancé |
| `exp13_gemma3_baseline` | config-only | benchmark | Baseline Gemma-3-1B, jamais lancé |
| `exp14_deepseekr1_baseline` | config-only | benchmark | Baseline DeepSeek-R1-Distill-1.5B, jamais lancé |
| `exp_deepseek_v4_pro_max` | config-only | API (eval_openai_compat.py) | Planifié, jamais lancé |
| `exp_gemini_baseline` | config-only | API Gemini | Planifié, jamais lancé |
| `exp_kimi_k2_6` | config-only | API (eval_openai_compat.py) | Planifié, jamais lancé |
| `prototypes/` | prototype | divers | Scripts d'exploration archivés (mini-cycle, skeletons, smoke tests) |

> **Note** : `exp_baseline_vllm_test/` (smoke test 10 items) est local et gitignoré — non listé ici.
