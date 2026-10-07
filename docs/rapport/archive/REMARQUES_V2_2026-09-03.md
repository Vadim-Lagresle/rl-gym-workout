# Remarques de Vadim sur la v1 complète (03/09) + remarques générales de Claude

> À relire avant chaque section de la passe v2. Ordre : §4 → §3 → §2 → intro/conclusion → annexes.
> Source de vérité du texte v1 + remarques : `RAPPORT_V1_AVEC_REMARQUES.tex` (collé en entier
> le 03/09, 1367 lignes). Index des remarques rouges : `grep -n 'textcolor{red}' RAPPORT_V1_AVEC_REMARQUES.tex`.

## Remarques rouges de Vadim (fond), par section

### Abstract (à rédiger EN DERNIER, mais cadrage à retenir partout)
- « étude expérimentale, partiellement exploratoire ».
- Pas de cadrage négatif (« ne pouvant prétendre à cette échelle »). Dire : ce qui nous
  intéresse est la stabilisation de l'apprentissage dans ces environnements et son
  optimisation (efficience).
- Dire quelque part que l'horizon est d'adapter cela à des environnements d'achat
  (agents shoppers Criteo) ; le retour d'expérience « pourra permettre à Criteo de gagner
  du temps pour le développement de ses agents shoppers ».
- LoRA (efficience) à placer EN PREMIER, curriculum (stabilisation) ensuite.
- Deux travaux principaux à mettre en avant : optimisation via LoRA, stabilisation via
  curriculum. Puis : « nous proposons une analyse de nos résultats en essayant de quantifier
  les capacités de transfert et l'utilité des curriculums au-delà de la simple
  stabilisation, vers des perspectives méta-apprentissage et apprentissage continu ».
- Pas « contribution » (prétentieux). Pas d'agents autotéliques dans l'abstract (c'est
  l'ouverture).
- Noter quelque part qu'AgentGym (2024, prédécesseur d'AgentGym-RL) ne faisait PAS de RL,
  jugé instable à l'époque : accroche historique pour la question de stabilité.
- Phrases trop longues avec trop de virgules : interdit, surtout dans l'abstract.
- La phrase « gestion des observations, reward hacking, instabilités LoRA, clés de
  reproductibilité » est trop longue et complexe pour un abstract.

### Remerciements (nouveau paragraphe à ajouter)
Patrick, Alberto, Alain pour la supervision depuis le début ; Sylvain pour les agents
autotéliques ; Flavian et Imad pour la découverte de DeepShopper ; Otmane pour les
discussions autour du SNIS.

### Introduction
- 1.1 : il manque une référence pour le SFT en instructions (FLAN / T0 / InstructGPT-SFT :
  à vérifier dans biblio_refs.tex).
- 1.2 : ajouter un cours de Stanford sur les LLMs en plus de CS336 (CS324 ou CS25, vérifier).
- 1.2 : reformuler la fin du paragraphe « on parle d'apprentissage par renforcement… » en :
  « l'algorithme renforce la distribution de sortie du modèle vers une distribution cible
  qu'il contient déjà grâce au pré-entraînement ».
- Contributions : « guide de bonnes pratiques » = prétentieux et faux → « rapport
  d'exploration organisé par constats de stabilisation ». « Ablations » = prétentieux.
  Définir ce qu'est un curriculum AVANT de l'utiliser (petit paragraphe avec quelques
  articles). Trop d'énumérations, on se perd : réécrire.
- Annonce du plan : les benchmarks « comme outil de définition des objectifs de l'IA
  agentique » plutôt que « matière première de l'entraînement ».

### État de l'art (reçu jusqu'à 2.4 partiel)
- 2.4 : figure illustrative demandée (batchs de tâches aléatoires vs organisés par
  difficulté).
- 2.1-2.3 : pas de remarque rouge reçue ; passe de style seulement (phrases longues).

### Section 3 (cadre technique)
- Ajouter quelques figures du papier (l. 287). Table 3 « archi moche » : reprendre le design du
  papier (zone grise, best en gras) ou capture (l. 334, 399).
- Dire ici (pas en 4.2) pourquoi on a répliqué à seed/train set/hyperparamètres identiques avant
  d'ouvrir : comprendre et reproduire d'abord ; sans cela on n'aurait pas trouvé env_mask (l. 444).
- Assumer pleinement le déséquilibre du dataset (d4 = 1 au train) comme limitation non étudiée
  (l. 397) ; « contribution free » : refaire un split avec toutes les données au train (l. 397).
- Accroche AgentGym 2024 : pas de RL car instable à l'époque (abstract).
- Remplacer « guide de stabilisation » (l. 296) et « cœur de notre contribution » (l. 300).

### Section 4 — intro et 4.1 Protocole
- Intro §4 : stop « guide » ; « étude de stabilisation et ce qu'on a vraiment à partager : LoRA »
  (l. 385). Transfert mono/multi SORT de 4.3 → ouverture/4.5 en paragraphe court (l. 387). Lire
  Yue et al. 2025 pour vérifier ce qu'on lui fait dire (l. 387, 620).
- 4.1 : énumérations trop riches, renvoyer aux sections amont par liens (l. 401). G=16 « permis par
  la mémoire libérée par LoRA » (l. 401). Chiffres du banc (18 %) à vérifier (l. 401).
- Justifier « toutes les 47 mises à jour » (l. 403) → 47 ≈ une epoch à G=8 (374×8/64 = 46,75) ;
  deux évals par epoch à G=16.
- « top-p 1,0 » pas compris (l. 403) → expliquer : pas de troncature nucleus, on échantillonne la
  distribution complète comme à l'entraînement.
- Calcul d'incertitude pas compris (l. 403, 405) → expliquer en clair : 100 items, proportion p,
  écart-type sqrt(p(1−p)/100) ≤ 5 points ; et appuyer sur NOTRE transparence (max = statistique
  d'extrême, on donne la plage des évals voisines) : « on déconstruit des métriques peu
  transparentes ». Le papier ne rapporte aucune incertitude (à vérifier, l. 532, 597).
  Vadim craint que ça décrédibilise le 82 vs 75 → RÉSOUDRE : headline = plateau (moyenne des
  dernières évals) et max en second ; 82 atteint deux fois ; plateau 75-82 > 73 (exp25.2).
- « journaliser » (l. 405) → « enregistrer dans les logs » ; « multi-graine » (l. 405) = seeds
  (graine aléatoire du run) : un seul run par configuration ; à dire comme travail futur.
- Oracle pass@k (l. 405) pas compris → définir avant d'utiliser ; renvoyer à 4.5.
- Compute contrôlé (l. 407) : « très intéressant, à bosser » → développer.
- Tableau hyperparamètres : le \ref{tab:hyperparams} ne pointe sur rien (l. 453) → créer le
  tableau ou retirer la référence.

### 4.2 Stabilisation
- Figures pour chaque partie (l. 438) ; « l'échec » LoRA → on va PLUS LOIN que le papier (l. 440).
- Masking : intro trop story-telling, pas de « plusieurs semaines » ; schéma explicatif du masque ;
  VÉRIFIER que seuls les tokens agent sont rétropropagés (l. 444) ; « 17, en dessous de 18 ».
- Reward shaping : retrouver les shapes exacts testés (l. 446, runs/0_baselines + hebdo) ; le lien
  de causalité « une action par tour → abandon des tâches composées » n'est pas clair → expliquer
  ou retirer ; « quelle correction » (bonus) → préciser. Ouverture scientifique demandée (l. 446-448) :
  reward de bon sens = souvent mauvaise idée, il faut comprendre la propagation du signal ; Dr GRPO
  (regard critique, VÉRIFIER la section) ; magic numbers du reward shaping (Shop-R1, à citer,
  bibitem à créer) ; littérature sparse vs dense ; DeepSeek-R1 encourage les rewards par règles ;
  « reward gate » (0 tant que tous les tests ne passent pas) : retrouver le papier (Self-Challenging
  Agents ?). Pourquoi sparse chez nous : cas standard des envs et benchs maths, encouragé par les
  constructeurs ; la reco (rewards non sparses) demanderait un travail dédié non fait.
- Full-FT : dire qu'on a analysé leur code et reproduit à hyperparamètres égaux ; le nombre
  d'epochs du papier est donné pour le 7B, pas le 3B (l. 454). « Pas de full-FT > 70 ? » → NON :
  max full-FT 3B = 69 (exp23.4) ; exp26 7B avortée (bug). Appuyer la stabilité : reward train/test
  jamais aussi peu bruité (schéma) (l. 456).
- LoRA : motivation = full-FT trop long pour tester toutes nos expés + LoRA Without Regret comme
  source d'idées (théorie de l'information), PAS « promettait » ni « argument solide » (l. 460-462) ;
  « ne transfère pas NÉCESSAIREMENT » ; leur récompense : vérifier si dense. Ajouter : LoRA libère
  la mémoire → G=16 → signal d'avantage plus précis (l. 460, 475, 500). Les 4 hypothèses : un
  tiret chacune, plus de matière, appuyées par les courbes KL/collapse (l. 465) → utiliser
  l'analyse KL transverse (docs/RESULTS.md) : dérive linéaire full-FT vs géométrique LoRA, rang
  accélère, β retarde d'une fenêtre, point de non-retour 0,05-0,1, β·KL vs terme de politique.
- Ancre mobile : ne pas prétendre à l'universalité ; tests de robustesse sur les autres envs
  AgentGym-RL à venir (l. 469). « Si le rappel est soit trop fort soit trop faible, il faut
  déplacer la référence » (l. 471). Chercher des papiers « moving anchor / reference reset » →
  candidat : ProRL (Liu et al. 2025, reference policy reset + KL control ; soutient aussi que le RL
  étend les frontières pass@k, contre Yue) — VÉRIFIER et bibitem. Dire que l'optimiseur (moments
  Adam des params LoRA) est aussi remis à zéro (l. 471). Lien TRPO → région de confiance (l. 471) :
  présenter TRPO en annexe A (bibitem Schulman 2015).
- « β=0,01 reste préférable » : Vadim doute (l. 473) → CORRIGER (indiscernable à epoch égale).
- Signaux avant-coureurs : Cui 2025 « je sais pas de quoi on parle » → définir la loi entropie/
  performance avant (l. 500) ; G16 arrive sans intro → l'introduire dans LoRA (l. 500) ; graphes
  indispensables ; étude de corrélation signaux ↔ collapse (matrice) et courbes d'erreurs (l. 500)
  → à faire si temps, sinon dire non fait.
- LR : ton trop familier (l. 502) ; graphe si matière, sinon la dernière phrase suffit.
- Paragraphe final « tips » → conclusion de partie qui rappelle les résultats (l. 504-505) ;
  « investiguer la propagation du signal avant tout reward shaping » ; pas « recettes du mono-tour ».

### 4.3 Curriculums
- « Source d'entropie » interdit (l. 515) → écrire le mécanisme (PLAN §4.3 pt 5).
- Idée centrale à poser : DUALITÉ du curriculum = suite du travail de stabilisation ET question
  de recherche (méta-apprentissage, exploration, battre le pass@k, transfert) ; « trop intéressant
  mais risqué côté méta-apprentissage, et au pire ça stabilise » (l. 515). Ajouter : comportements
  généralisables (vitesse, généralisation, limitation des erreurs).
- Transfert mono/multi : retiré de 4.3 → ouverture (méta-apprentissage, LeCun/Dupoux) (l. 515).
- Une puce par régime de curriculum (l. 519, 522). Définition du curriculum tirée de papiers +
  souligner : pas seulement ordonner les tâches, aussi limiter les capacités du modèle (l. 520).
  Profondeur = « définition la plus proche du curriculum » (l. 524).
- Horizon : dire « nombre de tours », hypothèse : restreindre les tours pousse à réussir les tâches
  résolubles en peu de tours (faciles et/ou moyennes de façon efficiente) → analyser solved par
  depth (l. 522). Budget : même idée sur les tokens. Conclusion/ouverture : combiner les curriculums
  (l. 522).
- Contrôle-G16 « n'a pas tenu, témoin de la stabilité du curriculum » (l. 520). « Bras » interdit.
- Profondeur : mécanisme changé → passage au palier suivant si reward train moyen ≥ 0,8 sur une
  epoch, sinon 10 epochs max (exp33.1 ; vérifié : --depth-auto-threshold 0.8, --depth-auto-max-epochs 10) (l. 524).
- Autocurriculum : expliquer son utilité (difficulté intrinsèque inconnue, non alignée sur la
  nôtre ; reco : difficulté dure à définir ; Toolathlon utilise le nb de tours d'Opus) (l. 527).
  MAGELLAN : Vadim ne comprend pas la présentation (l. 528) → paragraphe court + ANNEXE D dédiée
  (présentation honorable : c'est un vrai travail de recherche) ; « prédiction pré-enregistrée »
  à reformuler (c'est nous, avant le run : désinvestir d4) ou retirer.
- Résultats : mettre à jour au 03/09 (exp33.1/33.2 72/80, exp34 45 collapse, exp35/35.1 78/80,
  exp36.1 en cours, exp38 en file) (l. 532, 538). Rappeler 80 epochs vs ~250 du full-FT (l. 532).
  Commencer par le run contrôle (l. 534). Pas d'enchaînements par virgules avec acronymes (l. 534).
  Contrôle : « pas eu le temps d'aller plus loin ; le collapse est en soi un résultat qui illustre
  la stabilisation par curriculum » (l. 534). Fin de paragraphe pas claire (l. 536).
- « Deux analyses de transfert » : lesquelles ? (l. 540) → acquisition par depth + élimination
  des erreurs (analyses A/B du plan) ; nommer.

### 4.4 Compute contrôlé
- Pas « màj » (l. 560, 588) ni « caveat » ni « bras ». Le fait que le curriculum raccourcit les
  épisodes est un AVANTAGE à mettre en avant, pas une gêne (l. 560). Heures GPU : pertinent ?
  Tours d'environnement : justifier (l. 560) → tokens générés = coût dominant ; tours = coût du
  serveur d'env (pertinent pour envs réels/API payantes) ; heures GPU = coût facturé.
- « Propose déjà des résultats et interprétations » (l. 560) → extraire les chiffres au gel.
- Tableau dépasse (l. 567) ; valeurs « à extraire au gel » (l. 576).
- Effet de G : « ouvre vers des pistes d'études » (l. 597). Bruit d'éval : cf. 4.1 (l. 597).

### 4.5 Expériences complémentaires (Vadim a déplacé le transfert ici)
- Réintroduire la section (l. 610-611) : mesures sans entraînement, ou non refaites dans le setup
  final, ou hors cadre central ; inclure le transfert.
- pass@1 nu : 18 (une passe, notre harnais) vs 10,3 (oracle, moyenne de 20 passes) vs 14 (papier)
  → TRANCHER avant d'écrire (vérifier configs runs/0_baselines et 7_oracle : mêmes tours/prompt ?)
  (l. 622).
- Dire que les gros modèles font mieux ; 3B choisi pour le compute et pour Criteo (latence
  d'inférence de la reco) (l. 622).
- Modèle full-FT intermédiaire : motiver (« même sans le meilleur modèle, l'hypothèse est
  dépassée ») (l. 624). Énumérations trop rapides, lecteurs non familiers (l. 624).
- Few-shot : « gain rapide à étudier » (l. 634).
- Inter-générations : « cheveu dans la soupe » → dire le message : modèles récents = entraînement
  agentique massif = easy win ; on est resté sur 2.5 pour la comparaison honnête avec le papier ;
  les conditions de stabilisation restent valables, les modèles récents seront sans doute moins
  instables (l. 638).
- Transfert mono/multi : rappeler le papier « en env un LLM fait mieux qu'en full planning »
  (Kambhampati) ; expliquer la question (mono aide multi ? vice versa ? méta-apprentissage entre
  les deux pour réduire le coût ; motivé par le coût mono-tour et la littérature SCPO/West-of-N de
  l'offre de stage) ; Plan-Mode « on savait que ça ne marcherait pas, testé car gratuit » ; format
  court comme SNIS + annexe (l. 646-653).
- Tableau récap : mettre à jour.

### Section 5 et conclusion
- §5 : ajouter méta-apprentissage (LeCun/Dupoux), combinaison de curriculums, agents autotéliques
  (ouverture), re-stratification, tests de robustesse ancre mobile sur les autres envs, exp36.1/38.
- Conclusion : pas « ne pouvant prétendre », pas « contribution », pas « guide » ; mettre à jour
  (exp34 collapse, exp33/35 résultats) ; retirer le transfert comme résultat principal.

### Annexes
- Annexe A : renommer « Annexe 1 » → A ; ajouter TRPO (court) ; « Berkley » → Berkeley.
- Annexe B : garder, court.
- Annexe C : C.3 dit « LoRA r=16 … 54 % (pic 58) » → périmé (r8, ancre mobile, 73/82) ; mettre à
  jour ; ajouter purges /tmp, bug 7B, variance d'éval (plan).
- Annexe D (nouvelle) : MAGELLAN.

## Réponses rapides aux questions posées dans les remarques
- 47 : une epoch à G=8 (374 items × 8 / 64 = 46,75 pas). À G=16, deux évals par epoch.
- top-p 1,0 : pas de troncature nucleus ; on échantillonne la vraie distribution, comme à l'entraînement.
- Incertitude : proportion sur 100 items → écart-type ≤ 5 points ; max d'une courbe = statistique
  d'extrême biaisée vers le haut. Le papier ne donne pas d'incertitude (à vérifier).
- Multi-graine = plusieurs seeds par configuration. Non fait (un run par config).
- Journaliser = enregistrer dans les logs (wandb + fichiers). Mot à remplacer.
- Bras = configuration d'une grille. Mot interdit.
- Full-FT > 70 : non, 69 (exp23.4) est le max 3B.
- exp33.1 : palier suivant si reward train moyen ≥ 0,8 sur une epoch, sinon 10 epochs max.
- « Source d'entropie » : remplacer par le mécanisme (groupes mixtes, covariance de Cui).
- Prédiction pré-enregistrée : nous, avant le lancement d'exp34, avions écrit que le sampler
  désinvestirait d4 (MAGELLAN_ANALYSE.md §4). Confirmé (p(d4) = 0,0006).
- Papiers à vérifier/ajouter : ProRL (reference reset), TRPO, Shop-R1, DeepSeek-R1 (rewards par
  règles), Kambhampati (déjà), référence SFT instructions, second cours Stanford LLM.

## Remarques générales de Claude (à garder en tête)

1. **Le défaut n°1 de la v1 est le rythme** : phrases à 3-5 virgules qui empilent incise,
   énumération et subordonnée. Règle mécanique : plus de deux virgules = couper. Les
   énumérations longues passent en puces (autorisées désormais) ou en phrases séparées.
2. **Vocabulaire d'humilité partout** : travaux et non contributions ; retour d'expérience
   et non guide ; comparaisons et non ablations ; « indiquent » et non « démontrent » ;
   « sur un run » quand c'est un run.
3. **Cadrage** : positif, orienté « ce qu'on étudie » ; horizon applicatif Criteo (agents
   shoppers) dit une fois dans l'abstract et une fois dans l'intro ou §5, pas plus.
4. **Ordre des travaux** : LoRA/efficience → curriculum/stabilisation → analyse (transfert,
   au-delà de la stabilisation, ouverture méta/continual). Autotélique = ouverture §5 ; le
   port MAGELLAN reste un résultat de §4.3 (négatif, expliqué), pas une bannière.
5. **Fond à faire passer dans §4 (déjà dans le plan)** : mécanisme du collapse (groupes
   mixtes, covariance Cui, ressource d'entropie, corollaire G=16 : PLAN §4.3 pt 5) ; deux
   modes de collapse (PLAN §4.2 pt 5) ; β=0.01 NON supérieur à β=0.001 (correction) ;
   exp34 = rétroaction LP × instabilité ; justifications chiffrées LoRA / r8 / β / ancre.
6. **Chiffres** : aucun de mémoire ; tout depuis runs/INDEX.md ou docs/RESULTS.md. Les
   statuts au gel : exp36.1 en cours, exp38 en file, résultats à insérer s'ils arrivent
   avant le 8/09, sinon « en cours au moment de la rédaction ».
7. **Le plan reste la référence** : comparer chaque passage au PLAN_RAPPORT.md, signaler
   les écarts, ne pas réinventer la structure.

## Faits vérifiés dans les papiers (03/09, agents de lecture ; PDF AgentGym-RL extrait dans le scratchpad)

### AgentGym-RL (arXiv 2509.08755 v1, 39 p.)
- **Aucune incertitude rapportée** : pas de seed, d'écart-type, d'intervalle, de « ± », ni de nombre
  de runs par configuration (implicitement 1). Un seul nombre par cellule dans toutes les tables.
  → Notre traitement du bruit d'évaluation est un PLUS de rigueur, à dire ainsi.
- Éval TextCraft (Annexe B.3) : « maximum number of interactions to 20 turns », température 1.0 et
  8 trajectoires par query sont décrits pour le RL ; le décodage à l'ÉVAL (greedy/sampling, nombre
  de passes) n'est PAS précisé. « Pass@1 » n'apparaît nulle part ; ils disent « success rate ».
  100 items déduits de la Table 3 (31/41/25/3) ; le 33,33 en depth 4 = 1 item sur 3.
- Hyperparamètres TextCraft : GRPO, lr 1e-6, KL 1e-3, T 1.0, n=8, 20 tours max ; donnés PAR
  ENVIRONNEMENT, communs 3B et 7B. Non trouvés : batch, PPO epochs, clip, longueur max, nombre de
  steps/epochs. Seul indice : axe x de la Fig. 6 = 0-300 steps pour TextCraft (modèle non précisé).
  → Notre affirmation « leur nombre d'epochs est donné pour le 7B » est FAUSSE : aucun nombre
  d'epochs n'est donné du tout. À écrire : « le papier ne rapporte ni steps ni epochs ; la seule
  indication est l'axe d'une figure (≈300 steps) ».
- Incohérence interne : Table 6 GRPO-7B TextCraft = 83,00 vs Table 3 AgentGym-RL-7B = 89,00 (3B
  cohérent à 75). La phrase « surpasses the base model by 30 points » ne correspond à aucune ligne.
- **ScalingInter : motivation en une intuition, pas de théorie.** Citation (§3.3) : « beginning with
  a large number of interaction turns often leads the model into redundant reasoning and unproductive
  actions, ultimately causing training collapse […] Conversely, constraining the number of
  interactions to remain consistently small tends to narrow exploration ». Formalisme réduit à une
  contrainte K_t ≤ h_t avec un calendrier monotone, ∆ et δh jamais instanciés. Paliers pour
  TextCraft : NON DONNÉS (« we set the transition points […] according to the total optimization
  steps »). Analyse du mécanisme : UNE figure (Fig. 7), sur Deep Search seulement (reward + nombre
  de tours vs steps pour max 5 / max 10 / ScalingInter). Aucune courbe d'entropie ni de KL, aucune
  ablation du calendrier, rien sur TextCraft au-delà de la Table 3. Effet sur TextCraft 7B :
  89 → 91 (+2, dont depth 4 : 0 → 1 item sur 3). Pas de ScalingInter-3B rapporté.
  → Réponse à Vadim : ils restent EN SURFACE. Notre mécanisme (groupes mixtes, covariance de Cui,
  entropie) est une contribution d'analyse que le papier ne fait pas.
- Stabilité : discutée qualitativement (collapse à 10 tours sur Deep Search, « high variance, credit
  assignment difficulties, overfitting to spurious behaviors » en légende de figure). Entropie et KL
  ne sont mentionnées que comme outils de logging (Annexe A), jamais comme résultat.
- Déséquilibre du dataset par depth : non mentionné. Ils constatent seulement le « performance
  cliff » à depth 4.
- AgentGym 2024 « sans RL car instable » : PAS sous cette forme. Le papier dit seulement que
  l'ancien AgentGym « supports only a limited set of training methods based on supervised
  fine-tuning » ; l'argument d'instabilité vise les AUTRES travaux RL multi-tour (« they struggle
  with optimization stability and efficiency »). → Formuler : « AgentGym (2024) ne proposait que du
  SFT et de l'imitation filtrée (AgentEvol) ; AgentGym-RL justifie le passage au RL en ligne en
  soulignant l'instabilité des tentatives antérieures de RL multi-tour ».

### Références secondaires (vérifiées, bibitems ajoutés à biblio_refs.tex)
- Shop-R1 (ICLR 2026, arXiv 2507.17842) : reward hiérarchique 0,5 (JSON valide) + 0,3 (type
  d'action) + 0,2 / 0,1 (attributs) + ROUGE-L × facteur DARS = 1000, α = 0,005, β = 0,001 ; AUCUNE
  ablation sur les valeurs des coefficients, seulement présence/absence des composantes. Parfait
  pour « magic numbers ».
- DeepSeek-R1 (§2.2.2 v1) : rewards par règles (accuracy + format) ; « we do not apply the outcome
  or process neural reward model […] because […] the neural reward model may suffer from reward
  hacking ». PRM = « inevitably leads to reward hacking » (§4.2).
- Dr. GRPO : NE parle PAS de reward shaping. Uniquement biais de longueur (1/|o_i|) et biais de
  difficulté (normalisation par std). Ne pas le citer pour le shaping ; citable pour la
  normalisation de l'avantage.
- TRPO (ICML 2015) : contrainte dure de KL vers la politique précédente ; PPO la remplace par le
  clipping. Bibitem `schulman2015trpo`.
- SFT instructions : FLAN (ICLR 2022) `wei2022flan`. Cours : CS324 `cs324` ; CS329A existe
  (automne 2025, Chowdhery et Mirhoseini) `cs329a`.
- « Reward gate 1 ssi tous les tests passent » : SWE-Master (arXiv 2602.03411) et le blog DeepSWE ;
  SWE-RL ne convient pas (similarité de patch) ; Self-Challenging = reward binaire de vérification.

### Yue et al. 2025 (NeurIPS 2025 oral, Best Paper runner-up ; arXiv 2504.13837 v5) — VÉRIFIÉ
- Thèse : « RLVR mainly sharpens the distribution within the base model's prior rather than
  expanding beyond it » ; « base models achieve higher pass@k score when k is large » ; « the
  reasoning capability boundary often narrows as RLVR training progresses ». v5 dit « rarely
  elicit » (avant : « does not elicit »).
- Budgets : k jusqu'à 1024 (AIME/AMC), 128 (MATH500, code, visuel), 64 (LiveCodeBench) ; croisement
  « tens or hundreds ». Éval T=0,6 top-p 0,95 ; entraînement T=1,0. Sans terme KL (comme DAPO) ;
  ablation KL 0,001 : même pass@1, pass@128 plus bas.
- Steps courts : leurs checkpoints à 150/300/450 steps GRPO. Réserve verbatim : « We leave the
  question of whether scaling RLVR training can eventually surpass the base model to future
  investigation. » Conclusion appelle « multi-turn agent-environment interaction ».
- Ce que notre §2.3/4.5 lui fait dire est FIDÈLE. Nuances à écrire : (i) domaines mono-tour maths/
  code, pas agentique ; (ii) RL court (≤ 450 steps) ; (iii) k ≫ 20 chez eux, notre pass@20 est un
  budget faible ; (iv) leur propre réserve sur le RL long.

### ProRL, Liu et al. 2025 (NVIDIA, NeurIPS 2025 poster ; arXiv 2505.24864) — VÉRIFIÉ
- **Reference policy reset** (§2.3.1) : « Periodically, we hard-reset the reference policy π_ref to a
  more recent snapshot of the online policy π_θ, and reinitialize the optimizer states. » Déclenché
  sur stagnation/dégradation de la validation (pas de période fixe) ; 8 runs séquentiels, resets aux
  runs 2 et 6-7. Motivation : « the KL term may increasingly dominate the loss, leading to
  diminishing policy updates » ; KL gardée « for both stability and sustained entropy » (β non donné).
  → C'est notre ancre mobile, à la période près (nous : tous les 4 ep, fixe ; eux : sur signal). À
  citer en 4.2 : « mécanisme indépendamment proposé par ProRL ».
- **Pass@k** : « RL-trained models consistently outperform base models across a wide range of pass@k
  evaluations, including scenarios where base models fail entirely regardless of the number of
  attempts » ; contredit Yue explicitement : « these conclusions may stem from methodological
  constraints rather than fundamental limitations » ; « limited amount of RL training, typically no
  more than hundreds of steps ». Nuance : leur base est DeepSeek-R1-Distill-Qwen-1.5B (déjà
  distillé), > 2k steps, 16k GPU-h, GRPO+DAPO, n=16, T=1,2.
  → Pour §4.5 : notre observation (pass@1 82 > pass@20 47 ; frontière étendue +14 d2 +2 d3) va dans
  le sens de ProRL ; à présenter comme cohérent avec, pas comme preuve.
- Autres ancres mobiles vérifiées : TR-DPO (Gorbatovski 2024, arXiv 2404.09656 : soft/hard update de
  π_ref, cadre DPO offline) ; EMA Policy Gradient (Zhang & Ba 2026, arXiv 2602.04417 : ancre EMA,
  testée sur GRPO agentique Qwen-3B) ; suite ProRL « Scaling Up RL » (arXiv 2507.12507).

### Compléments du 03/09 (après lecture des papiers, décisions Vadim)
- **§3, motivation « transparence »** : le papier ne donne ni trace de run, ni courbe TextCraft, ni
  calendrier de curriculum ; ses scripts fixent 30 epochs, 32 prompts × 8 par collecte (~1 400 pas).
  Une motivation de notre étude est d'ajouter de la transparence sur ces entraînements RL : pour
  questionner les résultats, pour transmettre des constats et bonnes pratiques, pour ouvrir de
  nouvelles questions.
- **Dr. GRPO, angle retenu** : le biais de longueur (division par |o_i|) relève du même point que
  le shaping : un objectif d'apparence raisonnable induit un comportement néfaste si l'on ne
  détaille pas le calcul. Le citer ainsi (biais d'objectif), pas comme shaping.
- **Bruit d'évaluation, DÉCISION FINALE (Vadim 03/09)** : un paragraphe TRÈS COURT en 4.1, qui
  assume complètement ses limites (p différente par tâche, borne haute, aléa du décodage seulement,
  pas la variabilité inter-runs) : montrer qu'on s'est posé la question et que ça mériterait plus
  d'étude. Ensuite : NE PLUS INVOQUER le bruit à chaque résultat. Ne le mobiliser que s'il sert
  vraiment le propos (ex. 18 vs 10,3 ; plateau vs max) ; l'éviter dès qu'il floute l'analyse.
  Headline = plateau (moyenne des dernières évals), max en second. Zéro-shot de référence = 10,3.
