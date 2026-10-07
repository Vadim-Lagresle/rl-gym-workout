# Discours, partie 4 et conclusion (slides 17 à 28 du deck SOUTENANCE_V2)

Idées à dire, dans l'ordre, une slide à la fois. Pas de texte à lire, des points à développer.

## 17. Réplication en full finetuning
- Point de départ : la recette du papier, hyperparamètres pris dans son code, sur un seul GPU.
- Ce que montre la figure : pass@1 monte régulièrement, KL dérive lentement et linéairement, entropie
  monte à 1,2 puis s'affûte sans à-coup. Aucun collapse en 247 époques.
- Les chiffres : 40 % à 30 époques (le papier annonce 75), 69 % au meilleur, plateau 61.
- Le problème n'est pas la stabilité, c'est le coût : 238 h GPU, 12 Go par sauvegarde, une grille
  d'hyperparamètres impossible. La référence existe mais ne peut pas servir de base de travail.

## 18. Efficience : LoRA
- Pourquoi LoRA : on n'entraîne qu'un adaptateur de rang faible, l'optimiseur tient en mémoire,
  180 Mo par sauvegarde au lieu de 12 Go, mises à jour trois fois plus rapides, grilles possibles.
- La caution : LoRA Without Regret (Schulman 2025) montre qu'en RL, LoRA égale le full finetuning,
  et propose un taux d'apprentissage dix fois plus élevé. Hypothèse avancée : le RL n'apporte que
  quelques bits par épisode, un adaptateur suffit.
- Effet de bord utile : l'ancre KL est naturelle, c'est le modèle de base avec adaptateur à zéro.

## 19. LoRA à référence fixe : dérive KL
- Figure : KL vers le modèle de base au fil des époques, full finetuning et grille LoRA.
- Full finetuning : dérive linéaire, +10^-3 par époque, 0,08 après 247 époques. Stable.
- LoRA, même référence : dérive géométrique, ×5 à 10 toutes les deux époques, pour tous les rangs,
  taux et β. Tous franchissent 0,1, puis 1, en moins de quatre époques, puis s'effondrent.
- Pourquoi β ne sauve rien : β·KL vaut 10^-5 en régime sain contre 3·10^-3 pour le terme de
  politique. Le frein ne pèse qu'à KL ≈ 0,3, trop tard. β retarde, le rang accélère.
- Conclusion : avec une référence fixe la pénalité KL n'est pas une région de confiance. Ne pas
  prétendre expliquer pourquoi LoRA dérive plus vite que le full-FT : constat empirique, question ouverte.

## 20. Correctif : référence KL mobile
- L'idée : si la politique s'éloigne trop de la référence, déplacer la référence. Toutes les quatre
  époques : fusion de l'adaptateur dans le modèle, adaptateur vierge, référence = politique courante,
  optimiseur remis à zéro.
- Figure : croisement référence fixe/mobile × β. La KL reste à 10^-3 sur toute la durée. 65 puis 73 %.
- Lecture du croisement : c'est le régime de référence qui gouverne l'issue, pas β. β = 0,01 et 0,001
  sont indiscernables à référence mobile.
- Parenté : ProRL (NVIDIA 2025) réinitialise référence et optimiseur sur stagnation. Même idée,
  retrouvée indépendamment ; la nôtre est calendaire.

## 21. Bilan : deux modes de collapse
- On prend les 15 runs de la campagne, on regarde leurs 16 premières époques, on les classe par issue.
- Trois familles : 6 stables (full-FT, ancre mobile G=8 à deux β, les trois curriculums) ;
  6 collapses à référence fixe (la grille LoRA) ; 3 collapses à G=16 sans curriculum (le contrôle,
  sa variante à ancre toutes les 12 époques, l'autocurriculum MAGELLAN).
- Signature 1, référence fixe : la KL part en premier, géométriquement ; l'entropie monte puis chute.
- Signature 2, G=16 sans curriculum : la KL reste saine à 10^-3, c'est l'entropie qui descend
  d'abord, lentement, puis tout casse.
- L'ancre mobile règle la première. Elle ne règle pas la seconde. Transition : le second mode apparaît
  dès qu'on augmente la taille de groupe ; le curriculum est la réponse qu'on a testée.

## 22. Trois régimes de curriculum
- Recette commune stabilisée : LoRA r=8, taux 3·10^-6, β=0,01, ancre mobile /4 époques, G=16, 80 époques.
- Deux leviers du curriculum : doser les moyens de l'agent (Horizon : tours ; Budget : tokens par
  tour) ou ordonner les tâches (Profondeur : recettes de profondeur croissante).
- Deux références sans curriculum : Ancre-Mobile à G=8 (73 %), et Contrôle-G16 à recette identique,
  qui isole l'effet du curriculum.

## 23. Calendriers
- Horizon et Budget : paliers fixés, longs, pour laisser chaque niveau saturer. Le papier change
  toutes les 100 mises à jour, deux époques environ ; nous toutes les 15 à 20.
- Profondeur : passage adaptatif, palier suivant quand la récompense d'entraînement dépasse 0,8 sur
  une époque, ou après 10 époques. Paliers observés : 2, 12, 22. La première version à calendrier
  fixe saturait : six époques sans gradient sur le palier 1.
- Ce qu'un palier change : Horizon tronque l'épisode, Budget tronque la réponse, Profondeur filtre
  les tâches sans toucher aux moyens.

## 24. Mécanisme proposé
- Rappel : seuls les groupes mixtes produisent du gradient.
- Sans curriculum, à 30 tours, les tâches dures sont mixtes dès le départ : une réussite courte et
  routinière contre quinze échecs longs pleins d'actions rares. L'avantage récompense la routine
  (exploitation) et punit token par token l'exploration.
- Cui et al. 2025 : l'entropie baisse quand le probable est récompensé et l'improbable puni. Rien ne
  la recharge. Au plancher la politique se fige, puis la KL part.
- Le curriculum décide quand une tâche a le droit de produire du gradient : les tâches hors de portée
  ne punissent pas, les échecs sont courts, les réussites du palier suivant viennent d'actions rares.
- Dire explicitement : hypothèse, cohérente avec toutes les courbes, deux maillons non mesurés.

## 25. Résultats des curriculums
- Figure : pass@1 et entropie des trois curriculums face aux deux références et au contrôle.
- Le contrôle atteint 54 % puis meurt à l'époque 10. Les trois curriculums tiennent 80 époques.
- Horizon 82 au meilleur, 78 au plateau ; Budget 80/75 ; Profondeur 80/74 ; référence G=8 73/70 ;
  full-FT 69/61. Papier : 75. Incertitude d'une évaluation : environ 3 points.
- L'entropie des curriculums descend plus bas que la référence G=8 (0,2-0,5 contre 1,0), mais sur
  60 époques et avec un pass@1 qui monte : affûtage sur des tâches acquises, pas effondrement.
  Le contrôle, lui, passe de 1,2 à 0,4 en cinq époques et meurt.

## 26. Par profondeur et par classe d'erreur
- Même ordre d'acquisition partout : profondeur 1 en cinq époques, 2 vers 90 %, 3 entre 30 et 45 %,
  4 jamais (une seule recette d'entraînement).
- Les curriculums accélèrent sans changer l'ordre.
- Erreurs de forme éliminées en moins de mille mises à jour ; erreurs de planification lentes.

## 27. Contributions
- Quatre lignes : le cadrage ; la réplication à conditions égales et la recette LoRA stable ;
  les deux modes de collapse ; le curriculum comme protection à compute égal.
- Une phrase pour Criteo : infrastructure reproductible, prête pour un autre environnement.

## 28. Périmètre et perspectives
- Ce que ça autorise : comparaisons internes ; conditions de stabilité indépendantes du modèle.
- Ce que ça n'autorise pas : un run par configuration, un environnement, un modèle ; écart au papier
  inexpliqué ; facteurs G / tâches par pas / fréquence d'ancre non séparés.
- Suites directes : rééquilibrer le jeu par profondeur, répéter les runs, éprouver ailleurs
  (AgentGym-RL puis environnement d'achat, où la récompense n'est ni binaire ni exacte).
- Ouvertures : combiner les leviers ; faire choisir la difficulté par le modèle, sur base stabilisée.
