# Checklist rédaction v1 — 27/08 (ordre de revue conseillé : §4 → §3.3 → §1-2 → §5/conclusion → abstract → biblio/annexes)

## §4 Expériences — LE chantier (réécrire « acquis/protocole » → résultats établis)
- [x] 4.1 (v2 03/09) : splits exacts (train 374 : d1=109/d2=221/d3=43/d4=1 ; test 100 : 31/41/25/3 ; few-shot 70) ; tableau hyperparamètres (recette exp25 : r8/α32, LR 3e-6, β0.01, ancre /4 ep, N=8→16, GA 64) ; méthode compute-contrôlé (¶ prêt dans RESULTS.md) + caveat epoch N=8 vs N=16 (46 vs 93 steps)
- [x] 4.1 (v2 03/09) : **retirer les promesses non tenues** : « ≥3 graines », « FLOPs », ablation LoRA r∈{8,16,32} → reformuler honnêtement (mono-graine, bruit éval ±4-5 pts, coût en tokens/GPU-h)
- [x] 4.1 (v2 03/09) : REFAIRE le tableau de nomenclature (exp23/24/25/30-32/36, plus les vieux noms Shaping/Baseline+)
- [ ] 4.2 : garder env_mask + reward hacking (acquis, bons) ; réécrire LoRA (blog ne transfère PAS : grille exp24, collapses) ; ajouter full-FT exp23 (69) ; **ancre mobile + carré ancre×β complet** (tableau prêt, session 24/08) ; virer « attendre exp23.1 » et « Baseline+ 60 % » (dépassés : 73, 82)
- [ ] 4.3 curriculums : 4 régimes réels (horizon/depth/MAGELLAN/budget) + 2 baselines (exp25 N=8, exp36 N=16) ; **exp32 = 82 (> papier 75)** ; **exp36 collapse → curriculum = régularisateur** (¶ prêts, session 27/08 + RESULTS) ; note méthodo hyperparamètres égaux ; exp33/34/35 = « résultat au gel 2/09 » ; confronter H1-H3 aux résultats
- [ ] 4.3 : sous-section MAGELLAN (fonctionnement du sampler appris — base : nos échanges + MAGELLAN_ANALYSE.md)
- [ ] 4.3.2 transfert : garder (23 pts, +7.2) ; rouges stats (modèle = Qwen2.5-3B, test, n) ; Plan-Mode « protocole » → reclasser en non couru/perspective
- [ ] 4.4 few-shot : rouges k à unifier (5/10/20) ; protocole croisé → non couru
- [ ] 4.5 inter-générations : Qwen3-4B jamais évalué → compléter ou retirer la ligne
- [ ] Tableau récapitulatif : refaire → 18 / 31 (few-shot) / 54 (exp10.8) / 69 (full-FT) / **73** (exp25.2) / **82** (exp32) / 77 (Qwen3.5 nu) / 99 (Gemini) + pass@20 oracle par depth
- [ ] Figures : 7 placeholders avec légendes définitives (checklist du PLAN_RAPPORT) ; corriger les légendes « 3 graines » ; courbes produites au gel 2-3/09

## §3 Cadre technique
- [ ] 3.2 : rouge OpenReview → vérifier statut ICLR 2026 (web) ; option : ajouter l'asymétrie de moyens (~50 auteurs vs mono-GPU)
- [ ] 3.3 : rouges → 544 recettes, répartition par depth, splits (chiffres ci-dessus) ; **critique dataset (d4=1 au train vs 3 au test)** ; Table 3 → reporter les chiffres du papier (PDF) ; ajouter les 3 questions ouvertes (plan §3.3)

## §1-2 (passe mécanique, texte quasi fini)
- [ ] 1.1 : sharpening = hypothèse À TESTER, pas acquis (1 phrase) ; citer deepseekr1_2025 + openai_o1_2024 (déjà en biblio)
- [ ] 1.2 : note rouge notation (récompenses → R_i, unifier N/G, « annexe 1 » → A, micro-style, titre) ; couper DPO (jamais utilisé) ; **contributions → les 3 blocs du plan** (guide bonnes pratiques / étude curriculums / analyse inter-curriculums)
- [ ] 2.5 : GARDER (sert 4.5 : pass@k, attribution modèle de base) — révision du plan assumée
- [ ] 2.6 SNIS : réduire à un ¶ + renvoi annexe B

## §5 + Conclusion
- [ ] §5 : ajouter travail futur n°1 = re-stratifier le train set (confound d4)
- [ ] Conclusion : réécrire en rapport FINAL (résultats : 82, collapse instructif, carré ancre×β ; ce qui reste ouvert : d4, exp33-35) — plus de « la fin du stage visera à »

## Abstract (EN DERNIER)
- [ ] Réécrire avec résultats ; retirer « protocole qui structurera la fin du stage »

## Biblio + Annexes
- [ ] foster2026autocurriculum : ID placeholder → résoudre ou retirer ; qwen35 URL ; herakles à recouper ; ajouter Cui 2025 (entropy collapse) et Portelas 2020 si cités
- [ ] Annexe A : renuméroter « Annexe 1 » → A ; Annexe B : intégrer docs/hebdo/10juillet/snis_derivation.tex ; Annexe C : à rédiger (purges /tmp, bug 7B, variance éval — sources : docs hebdo)

## Répartition proposée
Claude drafte les fragments LaTeX de §4 (dans docs/rapport/, un fichier par sous-section) pendant que Vadim fait la passe mécanique §1-2 sur Overleaf ; puis 3.3, §5/conclusion, abstract. Vérif web (OpenReview, biblio) par Claude en fin de journée.
