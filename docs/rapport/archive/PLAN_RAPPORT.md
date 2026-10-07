# Plan détaillé du rapport — enchaînement des idées et papiers à citer

> Rédigé le 19/08/2026, révisé le 26/08 (remarques Vadim intégrées : fil rouge
> renversé, contributions restructurées, curriculum recentré, SNIS/test-time
> réduits à « autres pistes »). Calé sur la structure LaTeX existante du brouillon.
> Soutenance : 8 septembre. Gel des expériences : 2/09. Rédaction : 2-7/09.
>
> **Fil rouge du rapport** (à poser en intro, instruire au fil des chapitres,
> trancher — ou laisser honnêtement ouvert — en conclusion) : la littérature
> récente défend que *le RL on-policy aiguise une compétence préexistante mais
> n'en crée pas* (« sharpening », Huang et al. 2024 ; pass@k du modèle nu comme
> plafond, Yue et al. 2025). **Ce travail prend cette thèse comme hypothèse
> adverse à mettre à l'épreuve, pas comme acquis** : un curriculum
> d'apprentissage bien construit peut-il non seulement améliorer les
> performances, mais *percer* des murs de difficulté (le mur depth 4 de
> TextCraft) que le pass@k du modèle nu déclare infranchissables ? Notre étude
> est une *première* étude des curriculums dans ce cadre : ne pas percer le mur
> ici ne validerait pas la thèse du sharpening (facteurs confondants identifiés,
> au premier rang un train set quasi vide en depth 3-4 — cf. §3.3), et rien ne
> dit qu'une étude plus aboutie n'y arriverait pas. C'est cette tension
> instruite honnêtement qui fait l'intérêt du rapport. Qui plus est on peut dire que le rapport de l'étude se lit à la fois comme percer le sharpening et ces considérations très haut niveau, mais d'autre part également comme une étude empirique de la stabilisation de l'algorithme GRPO (Deep RL) dont les curriculums sont un des outils de stabilisation, que ça aide ou non à briser l'hypothèse de sharpening. 

---

## 0. Front matter (décision 27/08)

Après la table des matières : une **table des notations et abréviations**
(demi-page). Pas de glossaire de concepts (les notions se définissent à leur
première occurrence ; le formalisme vit en annexe A). La table fixe les
conventions : récompenses $R_i$, ratio d'importance $r_t(\theta)$, taille de
groupe GRPO $G$, $N$ réservé au nombre de tirages (best-of-$N$, pass@$k$),
définition précise de pass@$k$ (budget et température toujours rapportés).
V1 rédigée le 27/08, à coller après le sommaire.

## 1. Introduction

**Enchaînement d'idées :**
0. Présentation rapide du sujet des *self-improving AI agents* et de son statut
   de hot topic (cours Stanford récent dédié, vagues de papiers agents à
   ICLR/NeurIPS — ça intéresse du monde). Transition : pour comprendre
   *pourquoi* ça intéresse du monde, retracer l'évolution des signaux
   d'apprentissage de la dernière décennie.
1. Des LLMs statiques aux agents : le signal évolue — texte (pré-entraînement),
   préférences humaines (RLHF), puis **récompenses vérifiables d'environnement**
   (RLVR) — la 3e source de signal, la seule scalable sans annotateur. Clin
   d'œil historique : avant les LLMs, le RL se posait déjà ces questions en
   termes d'objectifs et de tâches — la vague agentique est moins une rupture
   qu'un *retour à la normale* du RL, outillé par les LLMs.
2. Promesse du self-improvement : un agent qui apprend de ses propres
   trajectoires. Question centrale (le fil rouge, posé comme QUESTION) : cette
   boucle peut-elle **créer** de la compétence, ou ne fait-elle que
   **concentrer** ce que le modèle sait déjà ? Et surtout : l'*ordre* (et pas que) dans
   lequel on présente l'expérience (curriculum) change-t-il la réponse ?
3. L'algorithme : présenter **GRPO et GRPO seul**, rapidement (avantage relatif
   de groupe, pas de critic — adapté à la récompense sparse terminale du
   multi-tour). Pour les fondements deep-RL (POMDP → policy gradient → PPO),
   renvoi à l'**annexe A** ; pour un traitement complet, renvoi aux cours de
   Chelsea Finn (Stanford CS224R) et Sergey Levine (Berkeley CS285).
4. Annonce du terrain (TextCraft/AgentGym-RL, Qwen2.5-3B, mono-GPU) et des
   **trois contributions** :
   1. **Réplication du papier et stabilisation de l'apprentissage** : analyse
      des recettes, runs et hyperparamètres testés (full-FT, LoRA, β KL, LR,
      ancre KL mobile), présentée comme un **guide de bonnes pratiques pour
      entraîner GRPO sur des environnements RL multi-tour** — en particulier :
      la recette LoRA de la littérature single-turn (*LoRA Without Regret*)
      collapse en multi-tour sparse, et notre correctif (ancre KL mobile,
      merge-and-restart) qui donne le record du projet.
   2. **Étude des curriculums** : présentation et implémentation de chaque
      régime — horizon (ScalingInter), difficulté par depth, autocurriculum
      APPRIS (port de MAGELLAN), budget de sortie par tour — à recette
      identique et compute contrôlé.
   3. **Analyse et comparaison inter-curriculums** : courbes d'entraînement
      (wandb), transfert entre tâches — succès par depth au fil du temps,
      taxonomie d'erreurs par depth × régime, coût en compute.
   Le mur depth 4 est traité comme **question ouverte instruite** (avec ses
   facteurs confondants, dont le train set), pas comme un résultat revendiqué
   dans un sens ou dans l'autre.

**Papiers :** Ouyang et al. 2022 (InstructGPT, RLHF) ; Shao et al. 2024
(DeepSeekMath, GRPO) ; DeepSeek-R1 2025 (RLVR à grande échelle) ; Huang et
al. 2024 (*Self-Improvement = Sharpening*) — cité dès l'intro comme hypothèse
adverse ; Schulman et al. 2017 (PPO) relégué à l'annexe A.

## 2. État de l'art

> Structure RÉVISÉE le 28/08 (décision Vadim) — six sous-sections :
> 2.1 agents · 2.2 environnements · **2.3 RL en environnement interactif :
> stabilité et mesure** (NOUVELLE : les 3 pathologies définies — reward
> hacking/Skalse, dérive/ancre KL, collapse d'entropie/Cui 2025 — + annonce du
> guide 4.2 ; pass@k, Yue 2025 et sharpening-hypothèse déplacés ICI, renvoi
> 4.5) · 2.4 curriculums (LE cœur) · 2.5 transfert mono/multi-tour (étoffée :
> + lien méta-apprentissage via Dupoux-LeCun-Malik 2026, arXiv:2603.15381) ·
> 2.6 autres pistes (scaling inférence sans pass@k + SNIS, renvois annexes).

### 2.1 Systèmes agentiques dans la continuité des LLMs
Définition opératoire (LLM + boucle observation/action + mémoire de contexte),
ReAct comme format canonique. **Papiers :** Yao et al. 2022 (ReAct).
(Décision 27/08 : Wei 2022 (CoT) déplacé vers l'intro §1.1 ; Xi 2023 (survey
agents) réservé pour §2.3 — bibitems fournis, à ajouter à la biblio si cités.)

### 2.2 Environnements et benchmarks agentiques
Panorama : mondes textuels à vérificateur exact (TextCraft, ALFWorld,
ScienceWorld, BabyAI) vs environnements ouverts (WebShop, WebArena). Ce qui
fait un bon banc : vérifiabilité, difficulté graduée, contamination faible.
Poser ici la notion de **difficulté paramétrique** (depth de l'arbre de craft)
qui rendra le curriculum mesurable. Tendance à souligner : les benchmarks
statiques et le texte disponible s'épuisent — la nouvelle manière d'entraîner
les LLMs propriétaires semble être de **créer des environnements et
d'entraîner dedans** (structures agentiques, usage d'outils) ; les plateformes
type AgentGym sont la version ouverte de ce mouvement. **Papiers :** Prasad et
al. 2024 (ADaPT, origine de TextCraft) ; Shridhar et al. 2020 (ALFWorld) ;
Wang et al. 2022 (ScienceWorld) ; Chevalier-Boisvert et al. 2018 (BabyAI) ;
Yao et al. 2022 (WebShop).

### 2.3 Curriculum learning, autocurriculum et agents autotéliques
**Section pivot de l'état de l'art** (cœur revendiqué du stage). Enchaînement :
1. Curriculum « humain » : ordonner les tâches du facile au difficile (Bengio).
2. Curriculum automatique : la machine choisit quoi apprendre — par progrès
   d'apprentissage (Graves ; Matiisen teacher-student), par regret/replay
   priorisé (PLR), par co-évolution agent-environnement (POET). **Y ancrer
   MAGELLAN** (Gaven et al. 2025) : le progrès d'apprentissage *prédit* par
   une tête apprise sur les représentations internes du LLM — c'est le régime
   que nous portons et testons (exp34).
3. Agents autotéliques : l'agent **génère ses propres buts** (IMGEP, motivation
   intrinsèque) — le curriculum devient une propriété émergente. Lien avec le
   méta-apprentissage : apprendre *quoi* apprendre.
4. Où se situe notre étude : entre 1 et 2 (curriculum par depth designé,
   curriculum d'horizon, autocurriculum ALP appris), et l'ouverture
   (self-challenging) pointe vers 3.
**Papiers :** (révisé 28/08 — anti name-dropping : chaque référence citée est
expliquée) Bengio et al. 2009 ; Gao et al. 2025 (survey self-evolving) ;
Graves et al. 2017 (LP) ; Rajaraman et al. 2026 (garanties autocurriculum) ;
**Gaven et al. 2025 (MAGELLAN)** ; Carta et al. 2025 (HERAKLES, une phrase) ;
retirés de 2.3 : GLAM et BOSS (hors sujet curriculum), SCA (déplacé vers
§5/annexe cadrage), Matiisen/PLR/POET/Portelas (non cités) ;
Colas et al. 2022 (survey agents autotéliques) ; Oudeyer & Kaplan 2007
(motivation intrinsèque).

### 2.4 Transfert planification mono-tour ↔ contrôle multi-tour
La compétence « plan » et la compétence « exécution interactive » sont-elles la
même chose ? Nos expériences single-turn (exp16 : collecter le plan en un tour,
le rejouer ; acquis partiel : +7 pts) instrumentent cette question. On n'aura
vraisemblablement pas le temps de nouvelles expériences ici, mais la section
reste dans le corps du rapport : elle parle de transfert (le fil de 4.3) et
fait le lien avec les papiers mono-tour de l'offre de stage. **Papiers :**
Prasad et al. 2024 (ADaPT décompose précisément ainsi) ; Huang et al. 2022
(Inner Monologue) ou Ahn et al. 2022 (SayCan) pour plan vs act.

### 2.5 Autres pistes considérées (un paragraphe, renvois annexes)
Deux directions ont été étudiées puis dépriorisées au profit du curriculum ;
les nommer en UN paragraphe chacune, sans développement dans le corps :
- **Scaling du calcul à l'inférence** (largeur best-of-N vs profondeur
  d'interaction ; le pass@k comme mesure du support — la notion est réutilisée
  en 4.5) → développement et papiers en annexe (Snell 2024, self-consistency,
  ToT, BOND ; Yue et al. 2025 reste cité au corps en 4.5).
- **SNIS** (recomposition off-policy de trajectoires ; v1 instructive mais
  variance/ESS pathologiques) → dérivation et bilan en **annexe B**.

## 3. Cadre technique : AgentGym-RL et TextCraft

Les considérations qui précèdent ont été dégagées au fil de lectures (papiers,
tech reports, surveys, cours) ; il fallait ensuite un cadre de travail qui
réponde à cette taxonomie — vérificateur exact, difficulté paramétrique,
chiffres de référence à répliquer. D'où le choix d'AgentGym-RL/TextCraft.

### 3.1-3.2 L'écosystème et le positionnement critique
AgentGym-RL comme plateforme de RL agentique (serveurs d'environnements HTTP
unifiés) : le papier couvre **5 environnements** (TextCraft, BabyAI, SciWorld,
SearchQA, WebArena) — c'est la plateforme AgentGym d'origine qui en expose 14
(vérifié dans `external/agentgym_rl_paper/`). Justification du choix
(vérificateur exact, difficulté paramétrique, chiffres à répliquer).
**Positionnement critique** : ce que le papier ne dit pas — variance
inter-runs, sensibilité aux hyperparamètres non documentés, et l'asymétrie de
moyens à assumer explicitement : **une cinquantaine d'auteurs et un cluster
multi-GPU côté papier, une réplication mono-personne mono-GPU ici** (ce qui
donne sa valeur au guide de bonnes pratiques de 4.2). *Rappel critique à
développer : contribution théorique du papier mince (ScalingInter = un
schedule d'horizon, le reste est de l'ingénierie), et dataset TextCraft aux
défauts non discutés — cf. 3.3.*
**Papiers :** Xi et al. 2024 (AgentGym) ; Xi et al. 2025 (AgentGym-RL).

### 3.3 TextCraft : anatomie, dataset et questions ouvertes
Arbre de craft, définition formelle de la depth, format d'action, récompense
terminale binaire. **Développer les prompts de tâches** (figure : exemple
d'épisode annoté, prompt système + observation + action). Expliciter les
résultats du papier sur TextCraft (leur Table : AgentGym-RL-3B = 75/100).

**Critique du dataset (chiffres vérifiés le 26/08 sur nos fichiers)** :
distribution des depths **train : d1=109, d2=221, d3=43, d4=1** ;
**test : d1=31, d2=41, d3=25, d4=3**. Un SEUL item depth 4 à l'entraînement,
alors que le test en compte 3 et que l'arbre de craft complet en génère
davantage — la distribution du train n'est ni représentative ni justifiée dans
le papier (items sélectionnés pour servir les résultats ? la question mérite
d'être posée). Conséquence méthodologique : l'échec éventuel d'un curriculum
sur d4 est confondu avec la quasi-absence de support d'entraînement. Nous
n'avons pas testé d'autres distributions (étude parallèle, hors budget temps),
mais **stratifier le nombre d'items par difficulté** est la suite naturelle :
les dynamiques de curriculum s'exprimeraient sans ce confound (→ repris en §5).

**C'est ici qu'on installe les questions que le chapitre 4 instruit** :
est-ce que le curriculum marche (vitesse ? asymptote ?) ; peut-on taper la
depth 4 avec un curriculum ; les pass@k du modèle nu sont-ils des oracles de
performance insurmontables, ou une bonne gradation permet-elle de les
dépasser ? Si les curriculums sont prometteurs et le temps le permet :
extension à d'autres environnements d'AgentGym-RL pour validation (à la
manière de ScalingInter chez eux) — cf. phase 4 du plan d'expériences.

## 4. Expériences et plan d'expérimentation

> Ordre narratif = ordre logique, pas chronologique : (a) protocole, (b) rendre
> le RL stable (prérequis + contribution 1), (c) curriculums et transfert
> (cœur, contributions 2-3), (d) inférence et ablations transverses (contexte).

### 4.1 Protocole commun
Modèle (Qwen2.5-3B-Instruct), GRPO à 64 trajectoires/step on-policy ; N=8
rollouts/prompt pour la lignée historique, **N=16 à trajectoires constantes
pour la phase curriculum** (harmonisation du 24/08, verdict du bench GPU
exp25.1 : 18 % plus rapide ET meilleure estimation d'avantage ; contrôle
dédié exp36 pour isoler l'effet de N). Éval périodique pass@1 sur test fixe
de 100 items, oracle pass@k stratifié par depth, budget mono-B200, stack
TRL+vLLM colocate. Honnêteté méthodologique : bruit d'éval (±4-5 pts),
politique de sélection du best. **Papiers :** von Werra et al. (TRL) ; Kwon
et al. 2023 (vLLM).

### 4.2 Expériences techniques : stabiliser le RL multi-tour (contribution 1)
À présenter comme un **guide de bonnes pratiques** GRPO multi-tour, adossé aux
runs. Enchaînement narratif :
1. **Réplication full-FT** : la lignée exp23 valide que la recette du papier
   tient en mono-GPU (montée régulière, 69/100). Leçons : warm-start, seuils de
   sauvegarde, bruit d'éval.
2. **LoRA : pourquoi, et le mythe de la simplicité.** Raisons du choix
   (mémoire optimiseur ~0.1 %, adapters légers, hypothèse de régularisation)
   + curiosité scientifique : tester *LoRA Without Regret* hors de son régime
   (single-turn dense → multi-tour sparse). Résultat : la recette du blog
   (10× LR) **ne transfère pas** — collapses. Grille LR × β × rang : β faible
   = pics puis effondrement ; β fort = stable mais plafonné.
   **Diagnostic géométrique** (4 mécanismes, à présenter comme hypothèses
   cohérentes mais partiellement confondues entre elles — LR, α/r et β
   diffèrent entre lignées) : (a) le full-FT répartit un gain de reward en
   déplacements infinitésimaux de ~3 Md de poids (petite KL), la LoRA confine
   le même gain à un sous-espace de rang faible → elle voyage PLUS LOIN dans
   les directions disponibles, plus de KL par point de reward gagné ;
   (b) chaleur effective : LR 3e-6 × facteur α/r (4 pour r8/α32) = pas ~6-12×
   plus agressif que le step full-FT ; (c) paramétrisation produit BA :
   couplage multiplicatif des gradients (argument du blog sur la pénalité
   gros batch) ; (d) reward sparse = gradients bruités que le full-FT moyenne
   et que la LoRA concentre. Preuve interne : la lignée full-FT (exp23,
   100+ epochs à β0.001) n'a jamais eu besoin d'ancre mobile.
3. **La solution : ancre KL mobile** (merge-and-restart périodique de
   l'adapter). Mécanisme à expliquer en ces termes : quand la politique est
   loin d'une ancre FIXE, le gradient de β·KL tire « vers l'arrière » (vers
   Qwen nu) — il dégrade le progrès sans amortir la dynamique locale ; une
   ancre MOBILE (dernier point de confiance) agit comme une **trust region
   locale**, qui pénalise la dérive rapide par rapport à un point récent —
   exactement le type de rappel qui prévient un runaway. exp25 : stabilité
   longue durée à petit rang, 65 puis 73/100 (exp25.2). Carré d'ablation
   ancre × β complété (exp30/31/31.1, 08/2026) : **aucune valeur de β ne
   sauve l'ancre fixe** (0.01 plafonne puis décroche, 0.001/0.0001 divergent) ;
   l'ancre mobile est nécessaire. **Le confound β/ancre de la grille exp24
   (correction méthodologique de Vadim, 19/08) est résolu par exp31** :
   β=0.001 + ancre mobile est STABLE (53) là où β=0.001 + ancre fixe
   collapsait — c'est bien l'ancre, pas le β, qui lève le collapse.
   **Correction 03/09 (analyse KL, docs/RESULTS.md) :** à epoch égale (ep 49-58),
   β=0.001 et β=0.01 sont INDISCERNABLES au test (46-53 vs 45-52) ; le 65-73
   d'exp25 vient de 110+60 epochs supplémentaires, exp31 ayant été arrêtée à
   l'ep 60. β=0.01 est un choix de continuité de lignée, pas une supériorité
   mesurée ; seule différence mesurée : β=0.001 garde plus d'entropie (1.20 vs
   1.05 à l'ep 60). Ne PAS écrire « β=0.01 reste meilleur ». Paragraphe prêt à
   coller (mettre à jour les chiffres : 65/73 au lieu de 12→50) : « Le full-FT
   amortit naturellement le RL à reward sparse : le progrès se répartit en
   déplacements infinitésimaux de tous les poids, ce qui maintient la
   politique proche de la référence en KL. La LoRA concentre le même progrès
   dans un sous-espace de rang faible avec un pas effectif amplifié (α/r) :
   à gain égal, elle s'éloigne plus vite en KL et sans rappel adapté elle
   devient instable. Une pénalité KL forte vers une ancre fixe stabilise mais
   plafonne ; déplacer périodiquement l'ancre transforme le rappel en trust
   region locale et lève le plafond sans sacrifier la stabilité. » Figure clé :
   courbes superposées.
4. **Analyse transverse — prédicteurs de collapse** (décision 27/08, analyse D) :
   méta-analyse des ~8 collapses documentés (exp19.2, exp22.2, 3 bras exp24,
   exp30, exp31.1, exp36) vs 5 survivants longs (exp20, 25, 25.2, 31, 32) —
   entropie plancher, écrasement de la longueur des complétions, spike KL comme
   signaux avant-coureurs. Figure : trajectoires d'entropie colorées par issue.
   Caveat assumé : hyperparamètres hétérogènes → axe DIAGNOSTIQUE, pas causal —
   c'est le cœur du « guide de bonnes pratiques ».
5. **Justifications quantitatives des choix (ajout 03/09, chiffres dans
   docs/RESULTS.md « Analyse transverse KL »)** — chaque décision de recette doit
   être adossée à un chiffre :
   - *full-FT → LoRA* : un best full-FT = 5.8 Go de poids + 8.1 Go d'Adam 8-bit
     (~14 Go) sur un home de 35 Go, contre 60 Mo + 120 Mo en r8 (~80× moins) ;
     grille de 9 runs, chaînes d'ancres (26 cycles) et reprises après purge ne sont
     possibles qu'en LoRA. Plus la question scientifique (LoRA Without Regret hors
     de son régime). Résultat final : full-FT 69 en ~250 epochs cumulées (lignée
     exp23) ; LoRA + ancre 73 en ~170 ; + curriculum 82 en 80.
   - *r=8* : grille exp24 à LR 3e-6/α32, le rang est anti-corrélé au score et
     corrélé à la dérive KL (r16 35-39 / r32 31 / r64 25-29 ; KL méd. ep 4-6 :
     r16 0.040, r64 0.28). Le plus petit rang est le plus stable, cohérent avec le
     « rang 1 suffit » du blog ; r8 = un cran sous le vainqueur de la grille.
     Caveat honnête : r8 n'était pas dans la grille, pas de bras r4/r16 sous ancre
     mobile ; validé a posteriori (exp25 : 35 à ep 15 = record de la grille, sans
     collapse).
   - *β=0.01* : sous ancre fixe, β retarde d'une fenêtre mais ne sauve pas
     (β·KL = 10⁻⁵ à KL 10⁻³, 0,3 % du terme de politique 2.9e-3 ; le frein
     n'engage qu'à KL ≈ 0.3, après le point de non-retour 0.05-0.1). Sous ancre
     mobile, β=0.01 et 0.001 sont indiscernables à epoch égale (point 3 ci-dessus).
   - *ancre mobile à β « élevé » plutôt que π_ref fixe à β très faible* : exp31.1
     (fixe, β=0.0001) est la divergence la plus rapide de toutes (KL 478 à ep 6-8) ;
     baisser β retire le seul frein sans retirer la dérive géométrique. L'ancre
     mobile annule la dérive par construction (KL 0.0010 ± 0.0002 sur 111 ep) et
     laisse β=0.01 jouer un amortissement local sans plafond (82 à 80 ep).
   - *Deux mécanismes de collapse à distinguer dans le texte* : à ancre fixe, KL
     d'abord (entropie encore 1.3-1.4) ; à G16 sans curriculum, entropie d'abord
     (KL stable 0.003). L'ancre traite le premier, le curriculum le second.
**Papiers :** Hu et al. 2022 (LoRA) ; Thinking Machines 2025 (*LoRA Without
Regret*) ; Ziegler et al. 2019 / Ouyang et al. 2022 (pénalité KL comme rayon
de confiance) ; Cui et al. 2025 (collapse d'entropie en RLVR).

### 4.3 Expériences RL : curriculums et transfert (CŒUR, contributions 2-3)
1. Question : ordonner l'expérience accélère-t-il ou améliore-t-il le RL
   multi-tour, et **le gain se transfère-t-il entre difficultés** ?
2. **Quatre régimes de curriculum** comparés à recette identique (celle
   stabilisée en 4.2), chacun expliqué en détail (contribution 2) :
   - **horizon** (exp32, ScalingInter : max_rounds 10→20→30 par paliers
     d'epochs) ;
   - **difficulté** (exp33 : depth ≤1→≤2→≤3→≤4 par paliers, sampler pondéré) ;
   - **autocurriculum APPRIS** (exp34, port fidèle de MAGELLAN : échantillonnage
     ∝ progrès d'apprentissage prédit par tête SR sur les embeddings du LLM —
     section dédiée à son fonctionnement, prédiction pré-enregistrée : l'ALP
     doit désinvestir d4 faute de support train). Argument de fond à porter
     ici (note 19/08) : **la depth est une difficulté *apparente pour
     l'humain*** — rien ne garantit que ce soit la vraie structure de
     compétence du modèle (largeur des recettes, chevauchement d'ingrédients,
     confusabilité des noms pourraient compter davantage). MAGELLAN fournit
     l'outil pour trancher : la partition induite par l'estimateur appris
     (« les groupes que la machine voit ») se compare à la partition par depth
     (information mutuelle, pureté de clusters) — **résultat intéressant dans
     les deux cas** (divergence = le curriculum humain optimise le mauvais
     axe). Si le run en ligne ne suffit pas, la sonde OFFLINE exp34c (tête
     entraînée sur les embeddings des checkpoints d'exp25 + succès loggés,
     zéro GPU de course) produit seule la figure machine-vs-humain ;
   - **budget de sortie** (exp35 : max tokens/tour 256→512→1024 par paliers).
   Contre **deux baselines** : GRPO uniforme N=8 (exp25, historique) et GRPO
   uniforme N=16 (exp36 — contrôle qui isole l'effet de N du curriculum).
3. **Métriques de transfert** (contribution 3) : pass@1 par depth au fil du
   temps ; matrice « entraîné sur depth ≤ d → perf sur depth d' » ; taxonomie
   d'erreurs par depth × régime — outillage `analyze_eval.py` ; courbes
   d'entraînement wandb (reward, KL, entropie). Deux angles retenus le 27/08
   (analyses B et C, extraites des logs existants, zéro GPU) :
   - **ordre implicite d'acquisition par depth** : le GRPO uniforme apprend-il
     naturellement d1→d2→d3 (curriculum émergent) et les curriculums explicites
     changent-ils cet ordre ? (pass1_d* loggé /47 steps sur tous les runs) ;
   - **ordre d'élimination des erreurs** : format → recette → planification ?
     le curriculum change-t-il la séquence ? (err_* par éval, tous runs).
4. **Lecture — les deux issues sont recevables** (fil rouge) : (a) le
   curriculum n'aide que là où le modèle nu a du support (lecture sharpening) ;
   (b) le curriculum débloque au-delà — signal au 27/08 : exp32 (horizon, N=16)
   atteint **82/100**, au-dessus de l'objectif papier (75), et le contrôle
   exp36 (N=16 SANS curriculum, recette identique) **collapse à 54** (entropie
   1.0→0.10 aux ep 8-12, KL runaway malgré l'ancre mobile) → premier verdict :
   N=16 seul n'explique pas le 82 ; **le curriculum agit aussi en RÉGULARISEUR
   d'exploration** (mécanisme détaillé au point 5 ci-dessous — angle à part entière pour le
   rapport, au-delà du débat vitesse/asymptote). Claim formulé à recette commune, conditionnel (pas de
   tuning par bras — ablations d'ancre notées en travail futur) ; reste à
   analyser par depth. Dans les deux cas, le confound du train set (d4=1)
   borne ce que TextCraft peut prouver sur le « mur ».
5. **Mécanisme du collapse sans curriculum et de la protection par le
   curriculum (validé avec Vadim le 03/09 — à rédiger en 4.3, c'est le cœur
   scientifique du verdict exp32 vs exp36).** Écrire au niveau du modèle, pas
   au niveau « commencer par le facile ». Enchaînement :
   - *Base théorique* : Cui et al. 2025 — la variation d'entropie à un pas de
     policy gradient est ∝ −Cov(log π(a), A(a)). L'entropie baisse quand les
     actions déjà probables reçoivent un avantage positif et les improbables un
     avantage négatif ; elle monte dans le cas inverse. Tout se joue sur QUELLES
     actions sont punies/récompensées, pas sur le reward moyen.
   - *Qui produit du gradient* : l'avantage est calculé par tâche sur le groupe
     des G essais du même prompt. Un groupe G/G ou 0/G a un avantage nul. Seuls
     les groupes MIXTES produisent du gradient. « Contrôler qui produit du
     gradient » = contrôler quels items sont mixtes.
   - *Sans curriculum, à 30 tours, la configuration a le mauvais signe* : le
     modèle nu a du support sur les tâches difficiles (pass@20 = 47), donc une
     d3 donne typiquement 1-3 réussites sur 16. Groupe mixte. La réussite,
     courte et routinière (actions probables), est amplifiée à A ≈ +3.9 ; les 15
     échecs durent 30 tours et se remplissent de dérive (actions improbables),
     tous punis token par token (avantage au niveau séquence, pas d'attribution
     par action : les bons get du début sont punis avec la dérive). Positif sur
     du probable + négatif sur de l'improbable → entropie en baisse à chaque pas,
     sur des dizaines d'items, pendant 10 epochs.
   - *Ce n'est pas la direction du signal qui est fausse* : le modèle apprend
     (exp36 monte à 54 PLUS VITE qu'exp32 au même step). C'est un effet
     secondaire : l'entropie est une ressource finie, chaque groupe mixte à longs
     échecs en brûle, rien ne la recharge. À ~0.1 la politique est déterministe
     et toute mise à jour fait sauter la KL (exp36 : 1.0 → 0.10 en 12 ep ; exp32 :
     0.83 → 0.22 en 80 ep). Phrase : « le signal est correct mais il consomme
     l'exploration plus vite qu'il ne la renouvelle ».
   - *Le curriculum horizon change le signe et le rythme, pas la direction* :
     (i) le cap à 10 tours transforme les groupes mixtes à longs échecs en groupes
     0/16 (la réussite d'une d3 demande > 10 tours) → gradient nul, entropie
     PRÉSERVÉE sur ces items parce qu'aucun gradient n'y touche ; les seuls
     groupes mixtes restants sont des items solubles en 10 tours, à échecs
     courts (peu de masse négative sur de l'improbable) ; (ii) au palier suivant
     ces items redeviennent mixtes dans de meilleures conditions (briques
     fiables → plus de réussites par groupe, échecs coupés à 20) et les
     nouvelles réussites viennent d'actions que le modèle produisait PEU (craft
     d'un intermédiaire au tour 2, get après un craft, > 2 crafts enchaînés) →
     positif sur de l'improbable → l'entropie REMONTE. Le curriculum ne choisit
     pas les items à la main : c'est le cap qui décide quels items sont mixtes
     et quand.
   - *Corollaire G = 16* : avec 16 essais, P(au moins une réussite) sur une tâche
     difficile est plus grande qu'avec 8 → plus de groupes mixtes à longs échecs
     dès le départ → plus de gradient du mauvais signe → collapse plus tôt
     (exp36 ep 10 vs exp25 stable 110 ep). Avec curriculum ces items sont à 0/16
     quoi qu'il arrive : G = 16 ne nuit pas (exp32). Même lecture pour exp34
     (MAGELLAN, G16, collapse ep 7-8).
   - *Curriculum depth vs horizon* (hypothèse) : l'horizon débloque à chaque
     palier des items de toutes profondeurs (dont 221 d2) → forte recharge
     d'entropie ; la depth ne débloque que 43 items au palier 3 et 1 au palier 4
     → recharge faible. Candidat pour expliquer 82 (horizon) vs 72-80 (depth).
   - *Statut épistémique à écrire* : mécanisme cohérent avec les courbes
     d'entropie/KL de tous les runs, mais la covariance n'a pas été mesurée, ni
     la probabilité des actions dans les réussites post-palier. Vérification pas
     chère si temps : longueur moyenne des trajectoires échouées exp36 vs exp32
     au palier 1 ; part de « craft non-cible » et « get après craft » dans les
     réussites avant/après palier. Sinon : hypothèse illustrée, pas loi.
**Papiers :** reprendre 2.3 (Bengio, PLR, teacher-student, MAGELLAN) + Xi et
al. 2025.

### 4.4 Comparaison des régimes de curriculum
Tableau synthétique + figure de transfert. Discussion : coût (epochs jusqu'au
seuil) vs plafond ; le curriculum change-t-il la *vitesse* ou l'*asymptote* ?
(rattacher à la littérature : le curriculum aide surtout la vitesse — Portelas.)

**Méthodologie de comparaison à compute contrôlé (décision 26/08).** L'axe
principal des courbes est l'EPOCH pour les runs N=16 entre eux : 1 epoch =
374 items × 16 rollouts = 5 984 trajectoires = **93 steps** de 64 traj.
⚠ L'epoch dépend de N : un epoch N=8 (baseline exp25) = 46 steps = 2 992 traj,
soit MOITIÉ moins — toute comparaison exp25 vs runs N=16 se fait en STEPS ou en
trajectoires cumulées, jamais en epochs (erreur d'échelle corrigée le 26/08).
Mais même à N constant l'epoch n'égalise PAS le compute : c'est précisément ce que les curriculums
modifient (exp32 : épisodes ~3× plus courts en début de run ; exp35 : tours
jusqu'à 2× plus longs en fin ; exp33 : items d1 courts d'abord). Le tableau
final reporte donc, par run, trois indicateurs déjà loggés (aucun re-run) :
1. **tokens générés cumulés** (`num_tokens`, log TRL/wandb) — le vrai compute
   de génération (~80 % du temps de step), l'égaliseur principal ;
2. **heures GPU** (somme des `step_time` / durée du run) — le coût réel ;
3. **tours d'environnement consommés** (mean_rounds train/éval) — le budget
   d'interaction, la ressource « expérience » au sens RL.
Lecture double : Pass@1 à epochs égales (efficacité en données) ET Pass@1 à
tokens générés égaux (efficacité en compute) — un curriculum peut gagner sur
le second sans gagner sur le premier (démarrer court = epochs moins chères).
La question N=8 vs N=16 (efficacité du groupe à trajectoires égales) se
traite ICI, dans la comparaison ScalingInter vs GRPO N=8 (exp25) vs GRPO
N=16 (exp36/36.1) — et, si le temps le permet, vs ancres plus espacées ;
un seul couple pré-collapse (exp25 vs exp36) : observation, pas loi.

**Régime de mise à jour vs le papier (décision 02/09 — justification validée
par Vadim, à placer en 4.1 ou 4.4 avec le tableau de décompte).** Texte
prêt à coller :

> Notre règle de mise à jour est celle du papier, prise à collecte réduite.
> Le papier effectue une seule passe sur chaque collecte, en quatre
> mini-batches de 64 trajectoires sur une collecte de 256. Notre budget
> mono-GPU nous a conduits à une collecte de 64 trajectoires, choisie comme
> compromis entre débit et taille de groupe ; la collecte tenant alors dans
> une seule mini-batch, les quatre pas du papier se réduisent à un pas unique
> strictement on-policy, où le ratio d'importance vaut 1 et le clipping est
> inactif. Ce régime a en outre été retenu comme garantie de correction après
> deux incidents de générateur désynchronisé rencontrés en cours de
> réplication. La mécanique complète du papier, collecte de 256 et quatre pas
> clippés, a été reproduite en full finetuning (\textsc{Réplication-FT},
> 69 %) ; son transfert au régime LoRA à ancre mobile, comme la réutilisation
> multi-passes des trajectoires que le papier n'effectue pas non plus,
> restent des leviers d'efficacité en génération à explorer.

Tableau de décompte associé (vérifié 02/09 dans le code du papier,
`dp_actor.update_policy` = une passe, et dans exp23 config) :

| | Papier (30 ep) | Nous G=16 (80 ep) |
|---|---|---|
| Collecte | 32 prompts × 8 = 256 traj | 4 prompts × 16 = 64 traj |
| Pas de gradient / collecte | 4 (mini-batches 64, clipping actif sur 3) | 1 (ratio ≡ 1) |
| Utilisations / trajectoire | 1 | 1 |
| Pas de gradient / epoch | ~47 | 93 |
| Total pas de gradient | ~1 400 | 7 440 |
| Total trajectoires | ~90 000 | ~476 000 |

Sources historiques : WORKLOG audit du 11/06 (points 2, 3, 5), incident
générateur figé du 16/06, migration v2 du 22/07 (paire
`num_iterations=1` + IS-correction off). Ne PAS écrire « parité
sémantique » : dire « même règle de mise à jour pour toutes nos
comparaisons ».

### 4.5 Expériences d'inférence et ablations transverses
1. **Oracle pass@k stratifié par depth — instruire l'hypothèse du « mur »** :
   depth 4 = 0 % même à k=20, pour le 3B ET le 7B nus, alors que le pass@1
   global est décent. Sous la lecture sharpening, c'est un plafond ; notre
   cadre le traite comme une *mesure de départ* que les curriculums attaquent
   — en gardant le confound d4-train en tête. Recouper avec la Table 3 du
   papier à la rédaction : leur score post-RL coïncide-t-il avec le support
   pass@k du modèle nu ?
   **Test direct du sharpening (analyse A, décision 27/08 — l'analyse phare)** :
   le 3B nu fait pass@20 = 47 % et exp32 fait pass@1 = **82 %** — le pass@1
   post-RL DÉPASSE le pass@20 du modèle nu. À instruire par depth (croiser les
   eval_logs d'exp32 avec l'oracle : quels items exp32 réussit-il là où le nu
   échoue 20 fois ?) avec les caveats k fini et température. Complément
   post-gel si possible : pass@20 du best exp32 (le support s'est-il ÉTENDU ?).
2. **Few-shot** : gain immédiat, sensibilité au format, rôle d'amorçage
   (injecter du support exogène — l'autre voie, complémentaire du curriculum,
   pour attaquer d4 en on-policy). Côté RL : la seule paire propre est
   exp22.5/22.6 (+8 pts, unique delta --fewshot 10, interrompues par purge) ;
   `exp37_fewshot_n16` (job 49, si temps) teste few-shot × N=16 sans
   curriculum. Repli assumé si non courue : « le few-shot amorce et accélère
   les premières epochs, sans garantie de stabilité ni de plus-value vs GRPO
   classique — risque de sur-imitation des recettes des exemples au détriment
   de la tâche courante ».
3. **Ablation inter-générations** : Qwen3.5-4B sans training dépasse l'objectif
   du papier → une partie du « mur » est une question de modèle de base, pas
   de RL. Discussion taille/génération vs spécialisation RL.
**Papiers :** Yue et al. 2025 ; Huang et al. 2024 ; Brown et al. 2020
(few-shot) ; Qwen2.5 technical report (Yang et al. 2024).

## 5. Délimitation du périmètre et ouverture

1. Ce qu'on ne couvre pas : multi-env systématique (un seul env de transfert
   en stretch), 7B full-FT (une phrase : tentative avortée sur bug de sync
   vLLM/embeddings non liés, arbitrage calendrier — renvoi annexe C), mix
   SFT+RL. **Pistes temporairement étudiées puis abandonnées** (un paragraphe
   au total, détail en annexes) : scaling d'inférence expérimental (BOND,
   best-of-N distillé) et **SNIS** (v1 instructive — variance des poids,
   effondrement de l'ESS, crash — version guidée par juge esquissée mais non
   courue ; annexe B).
2. **Travail futur directement actionnable — re-stratifier le train set** :
   la distribution d1=109/d2=221/d3=43/d4=1 est le principal confound de
   l'étude ; regénérer un train stratifié par difficulté (l'arbre de craft le
   permet) et re-courir les curriculums dessus est l'étude parallèle qu'on n'a
   pas eu le temps de faire, et le test le plus direct de l'hypothèse du mur.
3. Ouverture principale : **agents self-challenging / autotéliques** — si la
   frontière est la *génération du support*, l'étape suivante est l'agent qui
   se pose ses propres tâches (curriculum émergent), et la distillation
   (transférer le support d'un grand modèle vers un petit). Prolongement
   méta-learning à esquisser (note 19/08, angle soutenance) : un agent SANS
   prior de difficulté, qui évalue séquentiellement (taux de réussite) et
   adapte sa représentation des tâches à ces observations — la tête de
   compétence de MAGELLAN en est un premier pas, la prédiction bayésienne de
   LP (Kumar et al. 2024, cité par MAGELLAN §2.2) la direction ; connexion
   directe : le Challenger de Self-Challenging Agents devrait être piloté par
   le learning progress. **Papiers :** Zhou et al. 2025 (Self-Challenging
   Agents) ; Colas et al. 2022 ; DeepSeek-R1 (distillation).

## 6. Conclusion
Répondre au fil rouge avec ce que les données montrent, sans forcer le trait :
ce que la stabilisation, les quatre curriculums et le pass@k par depth disent
de la question « le RL seul peut-il rendre un petit modèle compétitif, et
l'ordre d'apprentissage peut-il repousser ses limites ? ». Trancher ce que nos
résultats tranchent (records, vitesse, transfert d2-d3, effet N), délimiter ce
qu'ils ne peuvent pas trancher (d4, confondu par le train set), et poser la
suite (stratification, self-challenging). Frugalité (LoRA + petit modèle
spécialisé + mono-GPU) comme résultat pratique transverse.

## Annexes
- **A. Fondements DRL** : POMDP → policy gradient → PPO → GRPO — **à
  développer** (l'intro ne présente plus que GRPO ; l'annexe porte le reste,
  avec renvois aux cours Finn/Levine).
- **B. SNIS** : dérivation + bilan de la piste (v1, pathologies, version
  guidée non courue) — **à développer depuis le début** (plus de section
  dédiée dans le corps).
- **C. Analyse critique : parcours, reproduction et dette technique**
  (existant) : purge /tmp, crashes, bug 7B (tie_word_embeddings), variance
  d'éval — le « vrai coût » de la réplication mono-personne. **Y ajouter
  (décision 27/08, analyse E — en annexe, pas en avant)** : la mécanique de
  persistance best+optimizer (sélection sous bruit, swap atomique, reprise à
  moments Adam cohérents) et une courte analyse warm-start (lignée exp23 :
  le « 64 » pic bruité, régression au vrai niveau ; exp25→25.2). Y discuter
  (ajout 30/08) la sémantique de la reprise LoRA par merge chaîne+best :
  l'ancre KL y est repositionnée sur le best au lieu du dernier ré-ancrage
  (principale source d'écart vs un run ininterrompu) et les moments Adam
  repartent de zéro sur l'adapter frais ; la « reprise exacte » (base = cycles
  seuls, adapter injecté, optimizer chargé, initial-cycle calé) est décrite en
  backlog dans la session hebdo du 30/08.

---

## Checklist figures (à produire pendant le gel, 2-3/09)
1. Courbe maîtresse stabilisation : β0.001 fixe / β0.01 fixe / ancre mobile.
2. Courbes des 4 régimes de curriculum + 2 baselines (exp25 N=8, exp36 N=16).
3. Heatmap transfert : régime × depth (pass@1 fin de run).
4. Barres pass@1 vs pass@20 par depth, 3B et 7B nus (l'hypothèse du mur d4).
5. Taxonomie d'erreurs par depth et par régime (aires empilées).
6. Distribution des depths train vs test (le confound, une figure honnête).
7. Timeline des runs (annexe C).
