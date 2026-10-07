# Notes pour le rapport — idées et arguments à développer

> ⚠ **FUSIONNÉ dans PLAN_RAPPORT.md le 27/08** (relire PLAN_RAPPORT seul suffit) :
> §LoRA/géométrie + trust region + formulation courte → PLAN §4.2 (avec mise à
> jour : l'« ablation manquante » β0.001+ancre mobile a été courue = exp31, 53
> stable → confound résolu) ; §curriculum humain vs machine → PLAN §4.3
> (partition machine vs depth, sonde 34c) et §5 (ouverture méta-learning).
> Ce fichier est conservé comme archive de la formulation d'origine.

Fichier de capture : arguments conceptuels issus des discussions de session, à
reprendre dans le rapport. Une section par idée, avec les expériences qui l'appuient
et les ablations manquantes.

## LoRA vs full-FT : stabilité, KL et géométrie des updates (19/08)

### 1. Le confound β / régime d'ancre (correction méthodologique — remarque de Vadim)

On ne peut PAS conclure des collapses des bras β=0.001 d'exp24 que « la LoRA a
besoin de β=0.01 » : ces bras tournaient avec une ancre FIXE (Qwen nu). Le collapse
peut venir du β trop faible OU de l'ancre lointaine — les deux facteurs sont
confondus dans la grille.

Nuance mécanique importante : à β égal, une ancre fixe lointaine et une ancre mobile
ne fournissent pas le même type de rappel. Quand la politique est loin de l'ancre
fixe, le gradient de β·KL tire « vers l'arrière » (vers Qwen nu) — il dégrade le
progrès sans vraiment amortir la dynamique locale. Une ancre mobile (dernier point
de confiance) agit comme une **trust region locale** : elle pénalise la dérive
rapide par rapport à un point récent, ce qui est précisément le type de rappel qui
prévient un runaway. Il est donc plausible que β=0.001 + ancre mobile soit stable
là où β=0.001 + ancre fixe collapse.

**Ablation manquante** : exp25 avec β=0.001 (ancre mobile /4 ep, r8, LR 3e-6).
Si stable et plus rapide que β=0.01, l'histoire « c'est l'ancre, pas le β » est
confirmée ; si collapse, le β compte bien en soi.

### 2. Pourquoi la LoRA serait intrinsèquement moins stable que le full-FT ici

Hypothèses (cohérentes avec nos observations, mais partiellement confondues entre
elles — LR, α/r et β diffèrent entre les lignées) :

- **Géométrie du sous-espace** : le full-FT peut obtenir un gain de reward en
  bougeant imperceptiblement chacun des ~3 Md de paramètres — solution proche en
  espace de fonctions, petite KL. La LoRA est confinée à un sous-espace de rang
  faible (r=8 : ~0.1 % des paramètres) : pour le même gain fonctionnel elle doit
  voyager PLUS LOIN dans les quelques directions disponibles → plus de KL par
  point de reward gagné → la pénalité KL mord plus tôt, et chaque step déplace
  davantage la politique (moins d'amortissement).
- **Chaleur effective** : LR 3e-6 (vs 1e-6 full-FT) × facteur α/r (4 pour r8/α32,
  2 pour r16/α32) → le déplacement par step dans le sous-espace est ~6-12× plus
  agressif que le step full-FT.
- **Paramétrisation produit BA** : le delta LoRA est un produit de deux matrices
  apprises ; le gradient de A est proportionnel à B (et réciproquement), un
  couplage multiplicatif qui peut amplifier les updates de manière non linéaire
  (argument du blog Thinking Machines sur la pénalité gros batch de LoRA).
- **Reward sparse multi-tour** : gradients bruités par nature ; le full-FT les
  moyenne sur tous les poids, la LoRA les concentre dans le sous-espace.

Preuves internes : exp24 β0.001 (r16/r32/r64) collapse tous ; r16/β0.01 fixe monte
à 39 mais zigzague (chute à 10 ep ~12) ; exp25 (r8, ancre mobile, β0.01) monte de
12 → 50 sur 50 epochs sans un seul accident, KL constante ~0.001. La stabilité de
la lignée full-FT (exp23 : 100+ epochs à β0.001 sans collapse) n'a jamais requis
d'ancre mobile.

### 3. Formulation courte pour le rapport

« Le full-FT amortit naturellement le RL à reward sparse : le progrès se répartit
en déplacements infinitésimaux de tous les poids, ce qui maintient la politique
proche de la référence en KL. La LoRA concentre le même progrès dans un sous-espace
de rang faible avec un pas effectif amplifié (α/r) : à gain égal, elle s'éloigne
plus vite en KL et sans rappel adapté elle devient instable. Une pénalité KL forte
vers une ancre fixe stabilise mais plafonne ; déplacer périodiquement l'ancre
(merge-and-restart) transforme le rappel en trust region locale et lève le plafond
sans sacrifier la stabilité (exp25 : 12 → 50 en 50 epochs, KL ~0.001 constante). »

## Curriculum humain vs structure découverte par la machine (19/08 — remarque de Vadim)

### L'idée

La depth est une difficulté *apparente pour l'humain*. Rien ne garantit que ce
soit la vraie structure de compétence du modèle : la largeur des recettes, le
chevauchement d'ingrédients entre tâches, ou la confusabilité des noms d'objets
pourraient structurer la compétence davantage que la profondeur de l'arbre.
MAGELLAN (Gaven et al. 2025) fournit l'outil pour trancher : un estimateur de
compétence appris sur les embeddings du LLM induit une partition des tâches —
« les groupes que la machine voit » — comparable à la partition par depth
(information mutuelle, pureté de clusters). Si les deux divergent, le
curriculum humain optimise le mauvais axe : résultat publiable dans les deux cas.

### Le lien méta-learning (ouverture soutenance)

Cadre où l'agent n'a AUCUN prior sur la difficulté : il évalue séquentiellement
(pass@k, taux de réussite), et adapte la représentation des tâches à ces
observations — la tête de compétence de MAGELLAN est un premier pas, la version
bayésienne (prédiction de LP, Kumar et al. 2024, cité par MAGELLAN §2.2) la
direction. Connexion directe : autotélique (Colas et al.), Self-Challenging
Agents (le Challenger devrait être piloté par le LP), et notre section 2.3.

### Le plan en trois volets (coût décroissant du risque)

- **exp34a** (sûr) : EK-Online-ALP par depth — baseline « groupes humains ».
- **exp34c** (sans GPU de course) : sonde OFFLINE — entraîner la tête MLP de
  MAGELLAN sur les embeddings de checkpoints successifs d'exp25 + les couples
  (tâche, succès) déjà loggés ; comparer partition induite vs depth au fil de
  l'entraînement. De-riske 34b et produit seule la figure machine vs humain.
- **exp34b** (si 34c concluant + calendrier) : sampler à LP appris EN LIGNE,
  version allégée de MAGELLAN — tête seule entraînée (BCE, buffer récence),
  embeddings recalculés depuis la politique COURANTE toutes les ½ epochs
  (forward no-grad, ~2 min/3000 items), compétence retardée par snapshots des
  poids de la tête. Pas de 2e adapter (l'ablation « LLM gelé = mauvais » de
  MAGELLAN concernait un porteur statique 248M ; notre porteur est la politique
  en cours d'entraînement, ses embeddings évoluent).
