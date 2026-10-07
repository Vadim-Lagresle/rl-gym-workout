# Plan des slides — soutenance de stage (30 min)

Titre de travail : **Stabilisation et efficience de l'entraînement autonome des agents LLM**

Principe : la présentation raconte la même histoire que le point de mi-stage, sans la
compresser. Le cadrage du sujet est une contribution à part entière et il est présenté comme
telle : sujet initial et littérature initiale, pourquoi elle ne colle pas aux agents, ce qu'on a
choisi d'étudier, pourquoi et comment. Les concepts des agents sont posés avant TextCraft.
TextCraft va vite. La partie expériences ne parle que de réplication, LoRA et curriculums.
Tout le reste (SNIS, mono/multi-tour, MAGELLAN, contrôle final) part en backup pour les
questions. Environ une slide par minute, 30 slides principales.

Budget : intro 4 min · périmètre et concepts 7 min · TextCraft 4 min · expériences 11 min ·
conclusion 4 min.

---

## Partie 1 : sujet initial et question large (4 min, slides 1-4)

1. **Titre.**
2. **Une histoire de signaux d'apprentissage.** Le tableau de la mi-stage (pré-entraînement,
   SFT, RLHF/DPO, RLVR, agentique). À chaque phase la donnée annotée est le goulot et la réponse
   est de plus en plus synthétique. L'agentique pousse la logique à bout : l'auto-amélioration
   n'est plus une option. (§1.1)
3. **Le sujet initial et la question large.** Sujet exploratoire : auto-amélioration d'agents
   LLM par génération de données synthétiques et filtrage de préférences, avec pour horizon les
   agents de commerce de Criteo. Les quatre papiers fondateurs : West-of-N, ScPO, Coral, BOND.
   Quatre manières de se passer d'annotations.
4. **Pourquoi cette littérature ne colle pas aux agents.** Les quatre méthodes supposent une
   réponse unique et mono-tour, vérifiable par vote ou par reward model. Un agent produit une
   trajectoire multi-tour, dans un environnement, avec une récompense terminale. La consigne
   d'Alberto : rester sur des papiers académiques, pas sur des POC internes. D'où la nécessité
   de préciser le périmètre avant de toucher aux agents de commerce.

## Partie 2 : précision du périmètre et concepts (7 min, slides 5-11)

Objectif : poser les concepts qui cadrent la littérature, dans l'ordre de l'état de l'art du
rapport (§2.1 à §2.4), et arriver à la question précise.

5. **Qu'est-ce qu'un système agentique.** LLM + harnais : outils, mémoire, format d'action,
   boucle observation-action. L'agent n'est jamais évalué nu. Ce qui vient des poids et ce qui
   vient du harnais. (§2.1)
6. **Qu'est-ce qu'un environnement agentique.** Interactif, évolutif, multi-tour, récompense
   vérifiable. Benchmarks comme définition des objectifs (AgentGym, Toolathlon, DreamGym). (§2.2)
7. **Le problème des trajectoires.** Mono-tour vs multi-tour ; boucle ouverte (plan puis exécution)
   vs boucle fermée (agir, observer, corriger) ; espace des trajectoires exponentiel en l'horizon ;
   récompense terminale et attribution du mérite ; peu d'information rétropropagée par unité de
   compute. (§2.2, §2.5 en une ligne)
8. **RL en environnement interactif : les objets qu'on va mesurer.** GRPO et ses groupes ;
   avantage nul quand le groupe est uniforme ; KL vers une référence ; entropie de la politique ;
   collapse. Les trois courbes qu'on lira dans toute la suite. (§2.3)
9. **Mesurer : pass@k et le plafond.** pass@1 vs pass@k ; l'hypothèse du plafond (Yue et al.) :
   le RL exploite une marge que le modèle de base possède déjà ; pass@k comme borne haute de ce
   qu'on peut espérer. (§2.3)
10. **Curriculum, autocurriculum, agents autotéliques.** Définition (Bengio, Narvekar) ; deux
    familles : ordonner les tâches ou doser les moyens ; autocurriculum (Portelas, MAGELLAN) ;
    pourquoi c'est intéressant théoriquement, ici, une fois pour toutes. (§2.4)
11. **Le périmètre choisi et la question précise.** Trois leviers d'amélioration : harnais,
    calcul à l'inférence, optimisation des poids. On choisit les poids, par renforcement, en
    multi-tour, sur un environnement donné. Question : **stabilisation et efficience des
    algorithmes d'apprentissage des agents LLM**. Le cadrage est présenté comme un résultat du
    stage.

## Partie 3 : TextCraft, le banc d'essai (4 min, slides 12-15)

12. **AgentGym-RL coche les cases.** Tableau : critères (environnements multi-tour, récompense
    vérifiable, code disponible, baseline chiffrée, difficulté mesurable, papier en conférence)
    × plateformes candidates, dont celles de la littérature initiale. Pourquoi TextCraft parmi
    les environnements : arbre de craft, profondeur = difficulté. (§3.1, §3.2)
13. **Anatomie de TextCraft.** Le system prompt, une tâche à profondeur 2 et son déroulé, la
    distribution train/test par profondeur (test : 31/41/25/3). (§3.3)
14. **La table de performance.** Papier 3B 75 %, Qwen2.5-3B nu 10 %, oracle pass@20 47 %
    (profondeur 3-4 à zéro), Qwen2.5-7B nu 33 %, Qwen3.5-4B nu 78 %. Il y a une marge, mais pas
    partout. (§3.3, §4.5)
15. **La boîte noire.** Le papier ne donne ni courbes, ni époques, ni incertitude ; texte et code
    divergent. Deux motivations : répliquer à conditions égales pour avoir une référence, et se
    confronter aux difficultés réelles de l'auto-apprentissage en environnement agentique pour
    en faire un retour à Criteo. (§3.2, §3.3)

## Partie 4 : expériences (11 min, slides 16-26)

Classification annoncée dès la première slide : références (réplication full-FT, LoRA à
ancre mobile), contrôle (G=16 sans curriculum), curriculums (Horizon, Budget, Profondeur).

16. **Protocole commun et nomenclature.** GRPO, 64 trajectoires par pas, on-policy strict,
    comparaison en mises à jour, évaluation périodique sur les 100 tâches test, plateau comme
    métrique de tête. Le tableau de nomenclature. (§4.1)
17. **Deux préalables.** Masquage des observations (figure) ; récompense binaire plutôt que
    façonnée. Conditions de possibilité, pas résultats. (§4.2.1)
18. **Réplication full-FT.** 69 % au plateau, dérive KL linéaire, stable. Mais 238 GPU-heures et
    une mémoire qui interdit toute grille. Constat : trop long, trop de compute. (§4.2.2,
    figure full-FT)

### 4a. Efficience : LoRA

19. **LoRA comme porte vers l'efficience.** Motivation : LoRA Without Regret (parité avec le
    full-FT en RL, quelques bits par épisode). Ce que LoRA coûte et économise chez nous. (§4.2.3)
20. **La recette publiée s'effondre.** Grille rangs × β : dérive géométrique de la KL vers la
    référence fixe, collapse entre 1,5 et 4 époques, quel que soit le réglage. Figure grille KL.
    (§4.2.3)
21. **Diagnostic et correctif : l'ancre mobile.** Ce que dit la KL fixe vs mobile ; merge et
    repart toutes les 4 époques, remise à zéro d'Adam ; parenté avec ProRL. Résultat : 73 % au
    même coût que LoRA. Figure carré ancre. (§4.2.4)
22. **Bilan stabilisation : deux modes de collapse.** Signaux précurseurs : KL d'abord (ancre
    fixe) vs entropie d'abord (G=16 sans curriculum). L'ancre mobile règle le premier ; le second
    reste. Figure signaux de collapse. Transition vers le curriculum. (§4.2.5)

### 4b. Stabilité : les curriculums

23. **Pourquoi le curriculum stabilise.** Le mécanisme « qui produit du gradient » : seuls les
    groupes mixtes ; à 30 tours, les tâches dures sont mixtes ; succès courts récompensés, longs
    échecs punis token par token, entropie consommée. Le curriculum décide quand une tâche peut
    produire du gradient. Présenté comme hypothèse, maillons non mesurés nommés. (§4.3.3)
24. **Trois régimes à recette commune.** Horizon (10→20→30 tours), Budget (256→512→1024 tokens),
    Profondeur (1→2→3→4, calendrier puis automatique). Définitions en trois puces. (§4.3.1)
25. **Résultats.** Figure pass@1 et entropie des trois curriculums face aux deux références et au
    contrôle. Tableau plateau/best : Horizon 78/82, Budget 75/80, Profondeur 74/80, référence
    G=8 70/73, full-FT 61/69, contrôle collapse. Par profondeur : le gain se fait en profondeur
    2 et un peu 3, la 4 reste à zéro. (§4.3.2, figures curricula + depths)
26. **À compute contrôlé, et ce qu'il reste à faire.** Trois axes (mises à jour, GPU-heures,
    tours d'environnement) : le curriculum gagne sur les trois. Réserve : le contrôle diffère par
    G et par le nombre de tâches par pas ; le run qui devait trancher est confondu par la
    fréquence d'ancre (détail en backup). (§4.4, F.6)

## Partie 5 : conclusion (4 min, slides 27-30)

27. **Contributions.** Quatre lignes : le cadrage du sujet ; une réplication à conditions égales
    et une recette LoRA stable (ancre mobile) ; la caractérisation de deux modes de collapse ;
    le curriculum comme protection à compute égal.
28. **Périmètre.** Un environnement, un modèle de 3B, un seul GPU, 100 tâches test ; ce que ça
    autorise et ce que ça n'autorise pas. (§5)
29. **Directions et pistes.** Contrôle propre du confondeur ; ancre mobile et remise à zéro
    d'Adam sur d'autres environnements AgentGym ; autocurriculum ; l'environnement shopping de
    Criteo comme terrain suivant. Le terrain des questions est préparé par les backups. (§5)
30. **Merci, questions.**

## Backup (pour les questions)

- B1. SNIS : idée, dérivation en une slide, pourquoi arrêté (annexe B, §4.5).
- B2. Transfert planification mono-tour / contrôle multi-tour : +7 pts, un seul sens testé
  (annexe D).
- B3. MAGELLAN : la méthode, notre port, collapse à l'époque 7 (annexe E).
- B4. Contrôle final G=16 × 8 tâches (exp39) : figures, le double confondeur, le run propre à
  faire (F.6).
- B5. Estimateur pass@k de Chen et al. ; oracle détaillé par profondeur.
- B6. Bruit d'évaluation : σ binomiale stratifiée 3,1-3,4 pts, borne haute.
- B7. β=0.01 vs 0.001 sous ancre mobile : indiscernable.
- B8. Divergence texte/code du papier : 20 vs 30 tours, époques absentes, 7B seul.
- B9. Compute par lignée : le tableau complet.
- B10. Qwen3.5-4B à 78 % sans RL : « pourquoi pas juste un meilleur modèle ».
- B11. Écart au papier inexpliqué : la réponse en deux phrases.
- B12. Few-shot : gain immédiat et amorçage.
- B13. Rappel GRPO : objectif, avantage de groupe, masquage.
- B14. Cadrage RecSys → agents (annexe C.1) si la question vient sur le premier mois et demi.

---

## Points à confirmer

- **« La partie contrôle »** : je l'ai lu comme le contrôle final exp39, envoyé en backup B4 et
  réduit à deux lignes de réserve sur la slide 26. Si tu voulais dire autre chose, dis-le.
- **Transfert mono/multi-tour** : une ligne sur la slide 7 seulement, le reste en B2.
- **30 minutes** : confirmé ? Si c'est 20, on coupe les slides 9, 17 et 28.

## Figures disponibles (docs/rapport/figures/)

fig_fullft_stabilite, fig_kl_lora_vs_fullft, fig_anchor_square, fig_collapse_signals,
fig_curricula_pass1_entropy, fig_curricula_depths, fig_curricula_errors, fig_pass1_tokens,
fig_pass1_gpuhours, fig_pass1_envturns, fig_errors_by_type, fig_exp39_control,
fig_exp39_vs_curricula. À produire : tableau des cases cochées (slide 12), figure masquage si
absente du dossier.
