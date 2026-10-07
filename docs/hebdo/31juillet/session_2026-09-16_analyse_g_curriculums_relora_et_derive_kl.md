# Session 2026-09-16 — analyse et interprétation : exp43 (borne k3), G=8 vs G=16, curriculums, ré-ancrage, dérive KL, ReLoRA

Suite de la session du 15/09 (rejeu k3, mécanisme du collapse). Journée d'analyse sur les logs
existants, sans nouveau run lancé à part la file. Toutes les figures citées sont dans
`docs/rapport/figures/` (script `plot_*.py` + PDF/PNG) et copiées dans `docs/slides/figures/`.

## 1. État d'exp43 (G=16 sans curriculum + borne k3 à 10), lancé le 15/09 15:57

- Époque 22 en fin de journée, 5 ré-ancrages, KL max 0,3 sur tout le run, épisodes stables à 550-650
  tokens, aucune boucle. A passé l'époque 10 (mort d'exp36) et l'époque 15 (mort d'exp40).
  → Le maillon « k3 non borné → divergence » de la chaîne du 15/09 est confirmé.
- Pass@1 : 31 (ép. 8), 42 (12), 54 (16), 56 (20), best 63 (ép. 19), dernières évals 55/51/61/61.
  Reward train 0,47 (ép. 12) → 0,64 (ép. 20).
- MAIS l'entropie s'est effondrée plus tôt et plus fort qu'exp36 : 1,11 (ép. 2) → 0,11 (ép. 4), contre
  1,0 pour exp36 au même point. Interprétation : borner k3 met le gradient KL à zéro sur les tokens
  où la politique s'éloigne le plus, donc plus rien ne freine le sharpening ; la KL non bornée d'exp36
  le freinait jusqu'à exploser. Prix : creux des ép. 4-10 (pass@1 25-31 pendant qu'exp36 était à 35-45).
- Régime actuel : entropie 0,33, ~12 tokens par tour depuis l'ép. 12. Horizon à l'ép. 20 était à 0,6 et
  a fait ses gains des ép. 30-66 en descendant lentement de 0,6 à 0,2. exp43 a déjà consommé cette
  marge → question ouverte du plateau (55-63 ou continue ?), verdict entre les ép. 30 et 60.

## 2. Analyse — l'effet de G (rollouts par tâche) : figure `fig_g8_vs_g16` (+ variante `_curricula`)

Piège de l'axe « époque » : une époque à G=16 contient 2× plus de trajectoires (374 tâches × 16) et coûte
≈ 2× le GPU d'une époque à G=8. Les figures tracent pass@1 et reward train par époque ET par
trajectoires cumulées (≈ compute).

| Run | traj/pas | pass@1 à 120k traj | à 180k | reward à 120k | à 180k |
|---|---|---|---|---|---|
| exp41 · G=8, 8 tâches, ancre /8 ép. | 64 | **58** | **60** | **0,62** | **0,65** |
| exp43 · G=16, 4 tâches, k3 borné | 64 | 57 | — | 0,64 | — |
| exp31 · G=8, β0,001 | 64 | 48 | — | 0,50 | 0,56 |
| exp25 · G=8, 8 tâches, ancre /4 ép. | 64 | 45 | 49 | 0,50 | 0,56 |
| exp39 · G=16, 8 tâches (128 traj/pas) | 128 | 45 | 50 | 0,52 | 0,54 |

Conclusions :
- **Rien ne montre que G accélère l'apprentissage à compute égal.** exp39 vs exp41 (seul delta = G, mêmes
  8 tâches/pas) : à 180k trajectoires, 50 contre 60. Par époque exp39 paraît en avance jusqu'à l'ép. 20
  (il a vu 2× plus de trajectoires), puis exp41 le dépasse dès l'ép. 25 (57 vs 51 à l'ép. 40).
- exp43 est le seul G=16 rapide à compute égal, mais il cumule 3 différences (4 tâches/pas, borne,
  entropie effondrée tôt → politique piquée à fort reward train). Ce n'est pas un effet de G.
- Seulement deux valeurs de G testées : aucune extrapolation.
- **Surprise : l'ancre.** exp41 vs exp25 = un seul delta (ré-ancrage /8 ép. au lieu de /4) : 60 contre 49 à
  180k traj. Chaque ré-ancrage réinitialise l'adaptateur et purge Adam → coût de réchauffage de plusieurs
  époques (confondu avec « référence 2× plus âgée », cf. §4).

## 3. Analyse — curriculums vs sans curriculum (figure `fig_g8_vs_g16_curricula`)

- À compute égal, jusqu'à ~120k trajectoires, exp43, exp41 et les trois curriculums sont indiscernables
  (55-58). Ensuite les curriculums continuent vers 70-80 alors que les runs sans curriculum plafonnent à
  60-66. **La différence n'est pas la vitesse de départ, c'est le plateau.** exp43 est au point de
  bifurcation.
- « G=16 stabilisé vaut un curriculum » reste vrai par époque, mais faux à compute égal en fin de run.
- Le collapse n'était pas une raison de faire des curriculums (la borne k3 suffit à l'éviter) ; reste
  la question accélération / plafond, tranchée par exp44 (Horizon à G=8) et la suite d'exp43.
- Profondeur (exp33.1) : le pic de reward train à 0,9 aux ép. 0-2 est le palier depth≤1 (109 tâches)
  saturé ; passage auto à depth≤2 (330 tâches) à l'ép. 2,14 → reward retombe à 0,4 (figure
  `fig_depth_reward_early`). Le run s'arrête à l'ép. 44,5 par purge pod (29/08), repris en exp33.2 → 80.
  **Règle adoptée : les reprises post-purge sont tracées en pointillé fin, décalées de l'époque et des
  pas du parent, sur toutes les figures** (fait sur les 4 figures concernées).
- Budget (exp35) : tombe à ~20 tokens par tour à l'ép. 16 et y reste 25 époques sans casser — contre-exemple
  à « tours courts = collapse », à comprendre (non élucidé).

## 4. Analyse — le ré-ancrage et les moments Adam

Ce que fait `MovingAnchorCallback` (schedules.py:718-738), VÉRIFIÉ dans les logs des 10 runs à ancre
mobile (exp25 → exp43, curriculums inclus, 2 à 27 cycles chacun) : fusion de B·A dans les poids de base,
réinitialisation de l'adaptateur EN PLACE (A kaiming, B=0), purge des états Adam des params LoRA.
- Les moments ne sont pas purgés « pour retirer la part KL » (ils sont des moyennes glissantes du gradient
  total, indécomposables ; et la KL vaut 1e-3 au ré-ancrage, gradient négligeable). Ils sont purgés parce
  que A est retiré au hasard : un moment attaché à un paramètre re-tiré n'a plus de sens.
- Option 1 (quelques lignes) : ne pas retirer A, mettre seulement B à 0, garder tous les moments —
  politique inchangée, moments de B toujours valides ; reste le démarrage lent de B=0.
- Option 2 (patch TRL moyen, un seul endroit) : ne plus toucher à l'adaptateur ; référence = copie FIGÉE
  de l'adaptateur (second adaptateur PEFT « ref »), rafraîchie toutes les N époques ; remplacer
  `disable_adapter()` par `set_adapter("ref")` dans le calcul des ref-logprobs. Aucun reset, Adam continue,
  politique de rang 8 pour toujours = **vraie LoRA à référence mobile**.

## 5. Analyse — la dérive KL à référence FIXE (figure `fig_kl_drift_fixed_anchor`)

KL médiane par demi-époque, échelle log ; runs zero-shot à LR constant (few-shot retirés).

| Run | KL ép. 2 | ép. 4 | ép. 6 | ép. 8 | ép. 10 | facteur / 2 ép. |
|---|---|---|---|---|---|---|
| full-FT, LR 1e-6 (exp23.1) | 0,0017 | 0,0029 | 0,0044 | 0,0060 | 0,0082 | ×1,4-1,5 (linéaire) |
| LoRA r8, LR 3e-6 (exp30) | 0,0013 | 0,0037 | 0,021 | 0,12 | 0,38 | ×5,6, ×5,8, ×3,2 |
| LoRA r16, LR 3e-6 (exp24) | 0,0011 | 0,0028 | 0,014 | 0,30 | mort | ×5, ×21 |
| LoRA r64, LR 3e-6 (exp24) | 0,0012 | 0,040 | 0,83 | mort | | ×21 |
| LoRA r64, LR 1e-5 (exp24) | 0,097 | mort | | | | |

- **Vocabulaire** : « géométrique » = exponentielle (×constante par intervalle) ; le full-FT ajoute une
  constante par intervalle (linéaire). On dira exponentielle.
- **LoRA ou LR ?** Panneau (b), KL vs LR cumulé Σlr (= budget de déplacement par paramètre sous Adam) :
  jusqu'à Σlr ≈ 0,9e-3, LoRA r8 et full-FT sont au même niveau (~0,04) ; ensuite le full-FT s'aplatit et la
  LoRA décolle. Ce n'est pas que LoRA dérive plus vite dès le départ, c'est la FORME : concave pour le
  full-FT, convexe pour LoRA. Le LR fixe la vitesse (r64 : 3e-6 → 1e-5 multiplie la KL à l'ép. 2 par 80 ;
  le rang aussi : r8/r16/r64 → 0,004/0,003/0,04 à l'ép. 4), pas la forme.
- **Manque** : aucune LoRA zero-shot à LR 1e-6 à ancre fixe (jobs parqués en août) — le run qui clorait
  la question LR (~15 h GPU). Les LoRA few-shot à bas LR (exp22.x) montraient la même explosion mais
  ne sont pas la recette.
- **Pourquoi LoRA dériverait exponentiellement (hypothèse, à présenter comme telle)** : le delta est un
  produit b·a ; chaque pas déplace a et b de ~lr ; le produit bouge de ~(a+b)·lr, proportionnel à la taille
  actuelle des facteurs qui grandissent → vitesse de dérive proportionnelle au chemin déjà parcouru =
  exponentielle. En full-FT, w bouge de lr par pas → linéaire. Second effet additif : en LoRA tous les pas
  sont alignés dans un sous-espace de rang 8 (la sortie bouge plus pour le même mouvement par poids) —
  c'est la version juste de l'intuition « moins de poids → plus de dérive » (le pas par poids ne dépend pas
  du nombre de poids sous Adam). α/r (2 à 4) est un multiplicateur constant, il ne change pas la forme.
  Vérifiable en traçant ‖B·A‖ au fil des pas d'exp30 depuis ses checkpoints.
- **Pourquoi c'est fatal** : β·KL = 1e-5 en régime sain vs 3e-3 pour le terme de politique ; le frein
  n'engage qu'à KL ≈ 0,3. Face à une dérive linéaire il a le temps ; face à une exponentielle (0,01 → 1 en
  2 ép.) il arrive après. Course perdue, pas instabilité numérique.

## 6. Analyse — borne k3 vs ancre mobile : complémentaires, pas substituables

k3 a deux régimes : ρ petit → k3 ≈ ρ²/2, gradient de rappel proportionnel à l'écart (informatif) ;
ρ > ~2,4 (k3 > 10) → borné, gradient nul (la borne ignore, elle ne freine pas). À référence fixe en
LoRA, la majorité des tokens franchirait la borne en quelques époques : KL loguée saturée ~10, sans
information, régularisation ≡ β=0 pour l'essentiel. **L'ancre maintient le gros des tokens dans le
régime quadratique où la KL parle ; la borne neutralise la queue.** Le papier (verl, full-FT, dérive
linéaire sous 0,03) n'a besoin que de la borne. Test à peu de frais : exp30 (ancre fixe, mort ép. 11)
+ borne.

## 7. Interprétation — ce qu'on a vraiment entraîné : ReLoRA, pas LoRA

- Le modèle après k cycles = Qwen ⊕ B₁A₁ ⊕ … ⊕ B_kA_k, rang jusqu'à 8k (160 après 20 cycles pour
  Horizon). C'est **ReLoRA** (Lialin et al. 2023, merge-and-restart avec reset de l'optimiseur), soit du
  full fine-tuning par tranches de rang 8. Le registre, le rapport (36 mentions) et le deck (18 mentions)
  disent « LoRA » : à requalifier en **trois régimes** — full-FT réf. fixe (réplication), LoRA réf. fixe (grille
  d'août, collapse), ReLoRA réf. mobile (exp25 et suivants, tient 80 ép.). La comparaison « LoRA vs
  full-FT » ne vaut que pour les runs à ancre fixe.
- **Stockage** : la chaîne de cycles (`<run>_anchors/cycleN/`, 58 Mo/cycle en fp32, 1,2 Go pour Horizon)
  est l'adaptateur de rang 160 écrit par blocs ; concaténer [B₁…B₂₀]·[A₁;…;A₂₀] donne le même volume ;
  la seule autre forme exacte est le modèle fusionné dense (5,8 Go). Pas de factorisation exacte plus
  petite (SVD tronquée = approximation). Un adaptateur seul est inutilisable sans sa chaîne ;
  `merge_anchor_chain.py` resomme Qwen ⊕ cycles ⊕ adaptateur. Full-FT pour comparaison : 5,8 Go modèle +
  6,2 Go Adam 8 bits = 12 Go par best.
- **Pourquoi ce choix** : besoin d'une référence mobile (dérive exponentielle à réf. fixe) → TRL calcule la
  référence PEFT en désactivant l'adaptateur (référence ≡ base) et `sync_ref_model` refuse PEFT →
  contournement le moins invasif sans patcher TRL = déplacer la base en fusionnant, ce qui impose de
  redémarrer l'adaptateur. **Phrase-fil pour les slides (validée par Vadim)** : « Ce n'est pas une
  contrainte mathématique, c'est une limite d'implémentation de TRL avec PEFT, plus notre choix du
  contournement le moins invasif. » ReLoRA n'a PAS guidé le choix : identifié après coup le 16/09 (les
  docs du 17/08 citent le billet Thinking Machines « LoRA Without Regret », Schulman et al. 2025, pour le
  warmup implicite de B=0, pas ReLoRA). À dire dans cet ordre.
- **Ce que ReLoRA dit de nous** (résumé du papier vérifié sur le PDF, hors de ce doc) : leurs cycles font
  5 000 pas (nos 372 sont côté « trop fréquent » de leur résultat négatif Online ReLoRA, cohérent avec
  exp41 > exp25) ; leur reset des moments à 99 % est le mécanisme qui force de NOUVEAUX sous-espaces (chez
  nous, effet de bord) ; leur gain est un gain de pré-entraînement depuis zéro où la capacité manque — en
  RL sur modèle pré-entraîné, Thinking Machines argue qu'un rang 1 suffit → prédiction : la vraie LoRA r8
  à référence mobile devrait égaler exp25 ; si elle plafonne plus bas, la capacité accumulée (ou le
  redémarrage lui-même) compte.

## 8. Runs proposés (découlant de ce qui précède)

En file derrière exp43 (guetteur `scripts/after_exp43_start_queue.sh` → runner) : **exp44 Horizon à G=8**
(job 57, = exp25 + un seul delta, le calendrier 10/20/30 ; répond à « le curriculum accélère-t-il ? » à G
fixe et à « G compte-t-il ? » à curriculum fixe), puis la reprise d'exp41 (job 58). **MAGELLAN parqué**
(`queue/skipped/59_exp42…`) : on se concentre sur l'analyse de l'existant. Candidats après jeudi, non
lancés : (a) vraie LoRA r8 à référence mobile /4 ép. (option 2, patch des ref-logprobs TRL) vs exp25 —
sépare « référence mobile » de « capacité/redémarrage » ; (b) la même à r64 si (a) plafonne ; (c) exp30 +
borne k3 — la borne remplace-t-elle l'ancre ? ; (d) LoRA zero-shot LR 1e-6 à ancre fixe — clore la
question LR vs LoRA. Correction du deck/rapport (3 régimes, ReLoRA) à faire avant jeudi sur validation.

## Fichiers touchés aujourd'hui
- figures : `plot_g8_vs_g16.py`, `plot_g8_vs_g16_curricula.py`, `plot_kl_drift_fixed_anchor.py`,
  `plot_depth_reward_early.py`, `plot_collapse_lengths_families.py`, `plot_collapse_lengths_all.py`
  (+ reprises en pointillé), `extract_lengths.py` (jeux `families`/`all` + KL + reprises) et leurs PDF/PNG.
- file : `59_exp42_magellan_horizon.sh` → `queue/skipped/`.
- mémoire Claude : `project_relora_story.md`.
