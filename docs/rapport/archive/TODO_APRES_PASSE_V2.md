# TODO rapport — à reprendre APRÈS la passe v2 (créé le 03/09/2026)

> Tâches données par Vadim au fil de la passe, à ne PAS faire tout de suite.
> On les reprend une fois la passe v2 terminée (ordre : §4 → §3 → §2 → intro/conclusion → annexes).

- [ ] **Plus de figures, exploiter toute la batterie de métriques.** On n'a utilisé que
  l'entropie, la KL, le pass@1 (global et par profondeur) et les erreurs. Restent
  sous-exploités : nombre moyen de tours par épisode (`eval/mean_rounds`, et par
  profondeur `rounds_mean_d*`), tokens traités (`num_tokens`), longueur des générations
  (`completions/mean_length`), récompense d'entraînement et son écart-type par groupe
  (`reward`, `reward_std`), fraction de groupes sans gradient (`frac_reward_zero_std`),
  norme du gradient. Les tracer ET les interpréter, par régime.
- [ ] **§4.1 : remarque en rouge sur l'explication des numéros `.1`, `.2`** (exp33.1, 35.1…) :
  expliquer dans le protocole ou la nomenclature ce que signifient les suffixes (reprises
  après purge, variantes de règle), ou les faire disparaître du texte.
- [ ] **Figures illustratives et schémas (non extraites des logs).** Reprendre quelques
  figures du papier AgentGym-RL (dont on a le PDF dans le scratchpad de la session du
  03/09) pour illustrer §3 ; produire des schémas maison pour l'agent et son environnement
  (boucle observation/action), le jeu TextCraft (arbre de recette, exemple d'épisode déjà
  en tcolorbox à enrichir), la profondeur des tâches (arbre par depth 1 à 4), et la liste
  des environnements de l'écosystème AgentGym. Voir aussi la figure "batchs aléatoires vs
  organisés par difficulté" demandée en §2.4 (remarque rouge du 03/09) et le schéma du
  masquage déjà fait (`fig:env_mask`, §4.2.1) comme modèle de style (tcolorbox).
- [ ] **Commenter explicitement le cas Qwen3.5-4B (78,5 % sans entraînement, 97 % de pass@18).**
  C'est la limite la plus visible du rapport : le score final du RL (82) est presque
  atteint sans RL par un modèle d'un an plus récent. À traiter de front, probablement en
  4.5.3 « Générations de modèles » et rappelé en §5/conclusion : (i) la valeur du travail
  est la compréhension du mécanisme d'entraînement (stabilité, curriculum), pas le chiffre ;
  (ii) le RL sur un modèle récent partirait de 78 et non de 10, avec des groupes GRPO mixtes
  dès le départ et probablement moins d'instabilité (à dire comme hypothèse) ; (iii) pour
  Criteo, la question « quel modèle de départ » compte autant que « quel algorithme », et
  les conditions établies ici (masquage, récompense, ancre, curriculum) restent nécessaires
  quel que soit le modèle ; (iv) proposer la mesure manquante : un run RL court sur Qwen3.5
  pour voir s'il gagne encore et s'il collapse.
- [ ] **4.3.3 : reprendre la formulation exacte de Cui et al. 2025** (arXiv 2505.22617) : « la
  variation d'entropie est pilotée par la covariance entre la probabilité d'une action et la
  variation de ses logits ; sous gradient de politique, cette variation est proportionnelle à
  l'avantage ». Éviter d'écrire directement « covariance log-probabilité / avantage » sans ce
  pas intermédiaire. Mentionner aussi leur loi empirique R = −a·e^H + b (performance plafonnée
  par l'épuisement de l'entropie).
- [ ] **Ajouter au rapport (4.3.3 et/ou conclusion) l'explication exploration/exploitation du
  collapse à G=16 et de la protection par curriculum**, formulée ainsi : GRPO n'apprend que sur
  les groupes mixtes ; à G=16 plus de tâches difficiles sont mixtes dès le début ; dans un tel
  groupe, la réussite courte et routinière (exploitation) est renforcée, les quinze échecs longs
  et pleins d'actions rares (exploration) sont punis token par token, bonnes actions comprises,
  parce qu'il n'y a qu'une récompense finale et pas de modèle de processus ; l'entropie baisse à
  chaque pas et rien ne la recharge ; au plancher la politique est déterministe et la KL explose.
  Le curriculum change QUI produit du gradient : tâches difficiles à 0/16 → non punies,
  exploration préservée ; échecs courts sur les tâches faciles ; au palier suivant les réussites
  viennent d'actions rares → l'entropie remonte. Curriculum = régulateur du compromis
  exploration/exploitation, par le moment où chaque tâche a le droit de produire du gradient.
- [ ] **Préparer la réponse jury sur le budget d'époques** (30 ép. / 90k traj chez eux vs 80 ép.
  G16 / 480k traj chez nous ; réplication exacte à 40 % à leur budget, 69 à 250 ép. ; 75 dépassé
  par Horizon à l'ép. 37 = 220k traj ≈ 2,5× leur budget) et l'assumer dans le texte (4.2.2/4.4).
- [ ] **Récit à faire passer dans 4.2.2 / 4.4 / §5 (décision Vadim 03/09) :** le papier est une
  boîte noire pour le 3B (pas d'intuition derrière les chiffres, pas de configuration, pas de
  garantie sur le nombre d'époques). Nous avons **isolé la difficulté** en régime strictement
  on-policy, un mécanisme à la fois (64 traj, 1 pas, ratio ≡ 1). Retirer l'argument « un seul
  GPU » pour justifier la collecte de 64 : exp23 a tourné à 256 sur ce GPU. Les arbitrages
  d'optimisation restent à explorer maintenant que la compréhension est là, et exp38 est écrite :
  (i) collecte 256 en 4 pas clippés sur la recette LoRA ; (ii) 8 tâches × 8 par pas (comme eux)
  vs 4 tâches × 16 (nous) à 64 trajectoires constantes : variance inter-tâches vs précision de
  l'avantage par tâche ; (iii) mesurer aussi la vitesse par trajectoire (64 vs 256) sur un GPU.
- [ ] **Facteur confondu G × tâches-par-pas (remarque Vadim 03/09, PRIORITÉ si GPU dispo).** exp25
  (8 tâches × 8) vs exp36 (4 tâches × 16) changent DEUX choses à 64 traj/pas. Le collapse G16 peut
  venir du nombre de tâches par pas (moins de diversité par mise à jour) et non de G. Run à
  prévoir : recette stabilisée, LoRA r8, G=16, `--gradient-accumulation-steps 128` (8 tâches ×
  16 = 128 traj/pas), sans curriculum, 30 tours, 80 ép. (arrêt manuel) ; verdict collapse/non en
  10-15 ép. Dans le rapport : reformuler 4.3.1 (« isole l'effet du curriculum », pas de G) et
  4.4 (« effet de G seule » → réserve explicite), phrases à insérer :
  * 4.3.1, remplacer « c'est le contrôle qui isole l'effet du curriculum de celui de la taille de
    groupe » par : « c'est le contrôle qui isole l'effet du curriculum. Notons qu'à 64 trajectoires
    par mise à jour, passer de $G=8$ à $G=16$ divise aussi par deux le nombre de tâches vues par
    mise à jour, de huit à quatre ; les deux effets ne sont pas séparés dans nos runs. »
  * 4.4, paragraphe « Effet de la taille de groupe seule », après « à recette identique » ajouter :
    « à une réserve près : à 64 trajectoires par mise à jour, le second ne voit que quatre tâches
    par mise à jour au lieu de huit, et cette différence de diversité par pas n'est pas séparable
    de l'effet de $G$ dans nos données. Un run à $G=16$ et huit tâches par mise à jour, soit 128
    trajectoires par pas, trancherait ; il n'a pas été couru. »

---
# Relecture v2 par Vadim (06/09) — remarques organisées par section

## Introduction
- [x] **1.1 — STaR et ReST** : vérifier les deux références et la phrase qui les cite (titres exacts,
  ce que chacun fait : STaR = bootstrapping de raisonnements filtrés ; ReST-EM = self-training
  itératif filtré par la récompense).
- [x] **1.2 — Équation 2** (gradient REINFORCE) : à réviser (notation, explication, cohérence avec
  l'annexe A.3).
- [x] **1.2 — Équation 4 (PPO)** : « où est la KL ? » L'objectif clippé est écrit sans le terme
  β·KL alors que l'équation 3 l'a. Soit l'ajouter, soit une phrase disant que la KL d'ancrage
  s'ajoute à l'objectif clippé (comme dans l'équation GRPO de l'annexe A.10).
- [x] **Référence [79]** : à identifier dans le PDF compilé et vérifier (entrée douteuse ?).

## État de l'art
- [x] **Bibliographie** (fait le 06/09 : 81 → 66 entrées, format condensé ; 8 citations à retirer dans le corps, liste donnée) : filtrer les références non lues / non utiles, il y en a trop (81 entrées ;
  9 jamais citées déjà repérées ; passer en revue celles citées « au passage »).
- [ ] **2.2 — « environnement comme artefact à générer »** : Self-Challenging Agents (zhou2025sca)
  génère des TÂCHES, pas des environnements ; DreamGym (dreamgym2025) inconnu de Vadim. Vérifier
  la pertinence des deux, reformuler (« générer des tâches et leurs vérificateurs ») ou retirer.
- [ ] **2.4 — agents autotéliques** : vaut-il le coup d'en parler si on ne l'applique pas ? Garder
  seulement comme objectif du dernier mois du stage / ouverture, réduire.
- [x] **2.4 dernier § — MAGELLAN** (décision 06/09 : on laisse tel quel, annexe E conservée) : « son résultat, négatif et instructif, est présenté en 4.5 et
  annexe E » → plutôt passer sous silence le run et présenter l'autocurriculum comme la suite,
  le programme du dernier mois. Décider : garder l'annexe E ou la retirer.
- [ ] **2.5 — méta-apprentissage** : dire honnêtement que c'est hors sujet par rapport à l'offre
  de Criteo (comment améliorer ses agents) et que c'est un intérêt personnel ; ne pas le
  présenter comme « prolongement le plus naturel ».

## Cadre technique
- [ ] **3.3 — « décodage à l'évaluation »** : expliquer le terme (greedy vs échantillonnage,
  température, top-p, nombre de passes) ou le remplacer par une formulation claire.

## Expériences
- [ ] **4.1 Infrastructure — suffixes .1/.2** (déjà noté) : après « seuls les meilleurs modèles,
  avec l'état de leur optimiseur, sont sauvegardés durablement », expliquer que l'infra Coder est
  souvent remise à jour, ce qui coupe l'entraînement ; on sauve le dernier état et l'optimiseur
  pour repartir après chaque coupure ; les .1, .2 sont les reprises des expériences coupées.
- [ ] **4.1 Évaluation — la formule √(p(1−p)/100)** : réexpliquer d'où elle vient (proportion de
  100 tirages de Bernoulli, variance np(1−p), écart-type de la proportion = √(p(1−p)/n), n = 100
  tâches) et la détailler dans le texte. Remplacer « mériterait une étude propre » par une
  estimation concrète : pour chaque expérience, taux de réussite moyen par profondeur sur les
  10 dernières évaluations, variance binomiale par profondeur × nombre d'items de la profondeur,
  somme → écart-type plus précis que la borne 5 points.
- [ ] **4.4 Compute — les trois axes annoncés en 4.1** (tokens, heures GPU, tours d'environnement)
  sont-ils tous comparés ? Un seul graphe (tokens cumulés). Produire les deux autres courbes
  (pass@1 vs heures GPU, pass@1 vs tours estimés) ou justifier pourquoi une seule.
- [x] **Tableau 5 (récapitulatif 4.5)** (décision 06/09 : ligne MAGELLAN conservée) : la ligne MAGELLAN est-elle utile ? Cohérent avec la
  décision prise pour 2.4/annexe E.

## Annexes
- [ ] **Annexe A, intro** : ajouter une phrase : « le but est un résumé vulgarisé rapide du
  fonctionnement de ces algorithmes, pour avoir les bons réflexes de manipulation lorsque Criteo
  souhaitera entraîner ses agents ; nous ne prétendons ni à l'exhaustivité ni à la précision, l'utilité
  est dans la sélection des informations essentielles à la compréhension de GRPO et PPO, les
  algorithmes majoritairement utilisés aujourd'hui dans ces cas d'usage. » Corriger « Berkley ».
- [ ] **Annexe B, intro** : ajouter une phrase de cadrage : brouillon ouvert d'une piste intéressante
  mais trop hors sujet pour aboutir ; à la croisée de la data valuation et du calcul à l'inférence
  (Snell et al., exploration de meilleures trajectoires) ; ici on ne construit pas les trajectoires
  tour par tour mais on recombine des trajectoires générées d'une traite ; apport visé : efficacité
  et facilité d'adaptation au cadre GRPO (GRPO classique sur les 64 trajectoires, puis recomposition
  de trajectoires intéressantes, et on mesure si le score s'améliore).
- [ ] **Annexe C.2** : retirer l'argument « harnais » (on le connaissait parfaitement). La difficulté
  venait du manque de transparence des auteurs sur les hyperparamètres, notamment pour le 3B, et de
  la difficulté notoire de stabilisation du RL. Réécrire le paragraphe suivant : pas des questions
  scientifiques au sens publication, mais un rapport technique de stabilisation et d'efficience, plus
  utile pour Criteo, d'autant que les papiers parlent rarement de leurs méthodes de stabilisation.
  Renommer C.2 (« Faire la lumière sur une technique trop absente des papiers », ou mieux).
- [ ] **Annexe C.3** : supprimer (ou ne garder qu'une ligne sur vLLM/TRL/glibc comme exemple de
  problème d'infra). Pas de liste de difficultés qui donne l'impression de se plaindre du stage ; le
  message est dans C.1 (recadrage) et C.2.
