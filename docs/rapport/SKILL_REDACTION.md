# Skill de rédaction du rapport — règles à appliquer à CHAQUE passage

> Créé le 27/08 sur consignes de Vadim. À relire avant chaque réponse de
> rédaction, avec `PLAN_RAPPORT.md` (le plan est LE contexte de chaque section).

## Workflow

1. Vadim colle un passage et indique de quelle section il s'agit. Il peut y
   glisser des remarques (inline ou générales).
2. Claude réécrit le passage en appliquant les remarques ET les règles ci-dessous.
3. Un paragraphe SANS remarque est considéré bon : rester au plus près de
   l'original (style corrigé si besoin, fond intact).
4. **À CHAQUE paragraphe, sans exception (rappel Vadim 03/09 : c'est LA raison
   d'être de Claude dans cette passe)** : relire la section correspondante de
   PLAN_RAPPORT.md ET les remarques de REMARQUES_V2_2026-09-03.md, puis dire
   explicitement dans la réponse ce qui a été vérifié (« vérifié dans le plan
   §x : … ; remarques traitées : … »). COMPARER le passage reçu avec ce que le plan prévoit pour cette section
   (fil rouge, décisions du 26-27/08, analyses A-G). Signaler tout écart entre
   le passage et le plan. Vérifier les chiffres dans runs/INDEX.md ou
   docs/RESULTS.md. JAMAIS de chiffre de mémoire.
5. Abstract et introduction se rédigent EN DERNIER.
6. **Ordre de la passe v2 (décision Vadim 03/09)** : cœur d'abord (§4
   expériences), puis §3 cadre technique, puis §2 état de l'art, puis intro et
   conclusion, puis annexes (succinctes). Les remarques rouges de Vadim sur la
   v1 complète sont consignées dans `REMARQUES_V2_2026-09-03.md` : les relire
   avant chaque section.

## Style (impératif)

- **Phrases courtes, vraiment (rappel 03/09).** Défaut principal de la v1 :
  des phrases à trois virgules ou plus, qui empilent incise, énumération et
  subordonnée. Règle mécanique : plus de deux virgules dans une phrase =
  la couper. Une énumération de plus de trois éléments = une liste à puces
  ou plusieurs phrases, jamais une phrase-fleuve.
- **Listes à puces autorisées** (ajout 03/09) quand elles sont pertinentes,
  de temps en temps : énumérations parallèles (contributions, régimes,
  étapes d'un protocole, conditions expérimentales). Pas de puces pour un
  raisonnement ou une argumentation, qui restent en prose. Une puce = une à
  deux phrases.
- **Vocabulaire d'humilité (03/09)** : pas de « contribution » (dire « nos
  travaux », « ce travail », « nous proposons ») ; pas de « guide de bonnes
  pratiques » (dire « retour d'expérience », « rapport d'exploration organisé
  par constats ») ; pas d'« ablation » pour des comparaisons à un run (dire
  « comparaison complémentaire », « variante ») ; pas de « démontrent »
  (« indiquent », « suggèrent », « montrent sur un run »).
- **Cadrage positif** : jamais « ne pouvant prétendre », « faute de moyens ».
  Dire ce qui nous intéresse (stabilité et efficience de l'apprentissage dans
  ces environnements) et l'horizon applicatif (adapter ces méthodes à des
  environnements d'achat, agents shoppers Criteo).
- **Ordre des deux travaux principaux** : LoRA pour l'efficience D'ABORD, puis
  curriculum pour la stabilisation. Le troisième volet est une analyse
  (transfert, utilité du curriculum au-delà de la stabilisation, ouverture
  méta-apprentissage et apprentissage continu). Les agents autotéliques sont
  l'OUVERTURE (§5), pas un résultat mis en avant dans l'abstract.

- **Aucun tiret d'incise** (— ou –). Dès qu'un tiret apparaît, reformuler la
  phrase. Les traits d'union de mots composés restent autorisés.
- **Phrases courtes.** Une phrase = une proposition principale, au plus une
  subordonnée. Dès qu'une phrase s'allonge ou empile des propositions,
  raccourcir ou couper en deux.
- **Un paragraphe = 1 à 4 idées qui partagent un même sujet ou thème**
  (règle corrigée le 27/08 : la version « une idée = un paragraphe » hachait
  le texte). Ne séparer deux idées que si elles ne partagent pas de thème.
  Ordre de grandeur : une subsection = 3 à 6 paragraphes, reliés par des
  transitions.
- Ton scientifique sobre. Aucune phrase à effet ou destinée à créer une
  émotion. Décrire, ne pas vendre.
- **Registre toujours soutenu et scientifique, jamais familier ni
  approximatif** (exemple refusé : « l'envie de brancher des LLMs sur des
  APIs »). Chaque affirmation est précise ou n'est pas.
- **Prendre le temps de définir avant de mobiliser** (ajout 28/08) : chaque
  notion est posée en une ou deux phrases avant d'être utilisée dans
  l'argument. Ne jamais entrer « dans le vif » sans avoir dit de quoi on parle.
- **Transitions explicites** : chaque paragraphe s'ouvre en disant pourquoi il
  vient (ce qu'il ajoute au précédent). Le lecteur ne doit jamais se demander
  ce qu'on veut dire.
- **Pas de name-dropping** (ajout 28/08) : une référence citée = au moins une
  phrase qui dit ce que ce travail fait et ce qu'il apporte à l'argument.
  Jamais une liste de citations en une ligne. Un travail qui ne sert pas
  l'argument de la section est déplacé (autre section, annexe) ou coupé,
  pas cité « au passage ».
- Claims conditionnels et honnêtes : « à recette commune », « sur un seul
  run ». Le bruit d'évaluation n'est expliqué QU'UNE fois (4.1, court, limites
  assumées) et n'est plus invoqué ensuite sauf s'il sert vraiment le propos
  (décision Vadim 03/09). Aucune promesse d'expérience
  non courue. Le sharpening est une hypothèse à tester, pas un acquis.
- Terminologie et notations unifiées : récompenses $R_i$ (jamais $r_i$, qui
  est le ratio d'importance $r_t(\theta)$) ; taille de groupe GRPO : **$G$**
  partout (décision Vadim 27/08 — $N$ réservé au nombre de tirages
  best-of-$N$/pass@$k$) ; annexes nommées A, B, C (jamais « annexe 1 »).

## Citations et bibliographie

- Source unique : `docs/rapport/biblio_refs.tex` (copie de la thebibliography
  du rapport). Vérifier CHAQUE `\cite{...}` dedans par grep avant de l'écrire.
- Citation absente de la biblio : fournir SYSTÉMATIQUEMENT, à la fin de la
  réponse, l'entrée `\bibitem` complète au même format (auteurs, titre,
  venue, arXiv), prête à copier-coller. L'ajouter aussi dans biblio_refs.tex.
- Ne jamais citer une référence dont l'existence n'est pas sûre. En cas de
  doute, vérifier (web) ou proposer une alternative de la biblio.
- Entrées connues à problème : `foster2026autocurriculum` (ID placeholder),
  `qwen35_2026` (URL à vérifier), `carta2025herakles` (à recouper). Ne pas
  s'appuyer dessus sans signalement.

## LaTeX

- Termes techniques anglais en `\emph{...}` à la première occurrence.
- Tableaux en `booktabs` (\toprule/\midrule/\bottomrule). Pas de capture
  d'écran : reporter les chiffres.
- Figures non encore produites : placeholder `\framebox` avec LÉGENDE
  DÉFINITIVE (le texte peut y renvoyer sans attendre la figure).
- Chiffres du projet à jour (03/09, toujours revérifier dans runs/INDEX.md) :
  baseline 18 ; few-shot k=5 31 ; full-FT exp23.4 69 ; exp25.2 73 ; exp32
  (horizon) 82 (> objectif papier 75) ; exp33.1/33.2 (depth) 72/80 ; exp35/35.1
  (budget) 78/80 ; exp34 (MAGELLAN) 45 puis collapse ep 7-8 ; exp36 (G16 sans
  curriculum) 54 puis collapse ep 10-12 ; exp36.1 (ancre /12) et exp38 (buffer
  256) en cours/en file ; Qwen3.5-4B nu 77 ; Gemini 99 ; oracle 3B nu pass@20
  47 ; splits train 109/221/43/1, test 31/41/25/3, few-shot 70.
- Corrections de fond acquises le 03/09 (PLAN §4.2 point 5 et §4.3 point 5) :
  β=0.01 et β=0.001 indiscernables à epoch égale sous ancre mobile (ne pas
  écrire « β=0.01 meilleur ») ; deux modes de collapse (KL d'abord à ancre
  fixe, entropie d'abord à G16 sans curriculum) ; mécanisme du curriculum =
  contrôle de quels items produisent du gradient (groupes mixtes), covariance
  de Cui et al. ; corollaire G=16.

## Format de réponse pour chaque passage réécrit

1. Le passage réécrit, en LaTeX, prêt à coller.
2. Une courte liste « ce que j'ai changé et pourquoi » (remarques traitées,
   règles appliquées, chiffres vérifiés).
3. Les éventuels `\bibitem` nouveaux.
