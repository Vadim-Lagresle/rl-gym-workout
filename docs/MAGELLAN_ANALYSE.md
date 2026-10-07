# MAGELLAN (Gaven et al. 2025, Flowers/Inria) — analyse du papier et du code

> Analysé le 19/08/2026 pour préparer exp34 (autocurriculum).
> Papier : https://arxiv.org/abs/2502.07709 — Code : https://github.com/flowersteam/MAGELLAN
> (cloné dans `/tmp/MAGELLAN`, ~1800 lignes de Python, très lisible).

## 1. L'idée en une phrase

Un agent autotélique doit choisir *quels buts pratiquer* dans un grand espace de
buts. MAGELLAN échantillonne les buts proportionnellement au **progrès
d'apprentissage absolu** (ALP = |compétence maintenant − compétence il y a N
updates|), et sa contribution est de rendre cette estimation **gratuite et
généralisante** : au lieu de ré-évaluer chaque but (coûteux) ou de grouper les
buts à la main (fragile), il apprend un petit estimateur de compétence
`C_θ(g) ≈ P(succès | but g)` branché sur les embeddings du LLM de la politique
— la proximité sémantique entre buts fait généraliser la compétence d'un but
pratiqué vers les buts voisins jamais pratiqués.

## 2. Leur stack (très différente de la nôtre — à savoir avant de comparer)

| Composant | MAGELLAN | Nous (TextCraft) |
|---|---|---|
| Modèle | Flan-T5 **248M** (seq2seq) | Qwen2.5-3B-Instruct (causal) |
| RL | **SAC** discret + critic + target critic | GRPO (pas de critic) |
| Action | **scoring de candidats** : le LLM score les 8 actions possibles, softmax, échantillonnage | génération libre multi-tour (vLLM) |
| Infra | lamorel (leur lib de serving distribué) | TRL + vLLM colocate |
| Env | Little-Zoo (maison) : 8 actions, déterministe, épisodes courts | TextCraft : espace d'action ouvert, 30 tours |
| Espace de buts | 25k-100k buts, **80 % impossibles** | ~3000 items train, 4 depths, tous possibles |
| LoRA | r16/α32 (comme notre défaut historique) | r8/α32 (exp25) |

Le point structurel : chez eux la politique ne génère pas, elle **score** des
actions candidates fournies par l'env (`LogScoringModuleFn` : somme des
log-probs de chaque candidat, distribution catégorielle dessus). Tout le RL
(SAC, replay buffer off-policy de 500k transitions, n-step returns) est
inapplicable tel quel chez nous. **Ce qui transfère, c'est exclusivement la
couche "goal sampler"** — et elle est remarquablement propre et découplée
(`goal_sampler.py`, 302 lignes, interface `sample()` / `update()`).

## 3. Les quatre samplers de `goal_sampler.py` (leur taxonomie = notre grille de choix)

### 3.1 `RandomGoalSampler` — la baseline
Uniforme sur les buts. C'est notre GRPO actuel (passe uniforme sur le dataset).

### 3.2 `OnlineGoalSampler` — ALP par but, sans réseau
Le mécanisme minimal, 40 lignes utiles :
- une deque de succès **par but** (maxlen = `buffer_size`) ;
- LP par but = |moyenne de la 2e moitié − moyenne de la 1re moitié| de la deque :

```python
midpoint = len(buffer_array) // 2
self.sr[i] = np.mean(buffer_array[midpoint:])
self.sr_delayed[i] = np.mean(buffer_array[:midpoint])
self.lp = np.abs(self.sr - self.sr_delayed)
```

- échantillonnage ∝ LP, avec ε-greedy **recuit** (ε : 1.0 → 0.2, décroissance
  exponentielle) — un plancher de 20 % d'uniforme est TOUJOURS conservé.

Limite (leur section 3.3) : pas de transfert de compétence — pratiquer le but
A ne met pas à jour l'estimation du but B pourtant similaire. Avec beaucoup de
buts et peu d'épisodes par but, le signal par but est famélique.

### 3.3 `EKOnlineGoalSampler` — ALP par GROUPE expert (le candidat pour nous)
Même mécanique, mais les deques sont **par groupe défini par expert** (chez
eux : grasp / grow plant / grow herbivore / grow carnivore / impossibles).
On échantillonne d'abord un groupe ∝ LP du groupe, puis un but uniforme dans
le groupe. Faiblesse chez eux : les groupes à la main supposent une compétence
homogène dans le groupe et aucun transfert entre groupes. **Chez nous cette
faiblesse disparaît presque : la depth EST un regroupement naturel, ordonné et
quasi homogène en difficulté** — c'est le cas favorable où leur baseline EK
est presque optimale (leurs propres résultats : EK-Online-ALP est la meilleure
baseline, MAGELLAN ne la dépasse que parce que leurs groupes cachent de
l'hétérogénéité et des buts impossibles).

### 3.4 `MAGELLANGoalSampler` — l'estimateur appris
Deux ajouts par rapport à Online :
1. **Tête de compétence** (`SRHeadModuleFn` dans `models.py`) : MLP minuscule
   (hidden → 128, tanh → 1, sigmoïde) branché sur le hidden state du dernier
   token du prompt du but, entraîné par BCE sur les couples (but, succès) des
   épisodes récents. Le LLM porteur est entraîné aussi, via des **adapters LoRA
   séparés de ceux de la politique** (leur ablation D.1 : LLM gelé = mauvais,
   adapters partagés avec la politique = instable).
2. **Compétence retardée par snapshot de poids** (`updater.py`) : une deque des
   N derniers snapshots des poids de l'estimateur ; l'ALP d'un but =
   |C_θ(t)(g) − C_θ(t−N)(g)| où le terme retardé est calculé en copiant le plus
   vieux snapshot dans un adapter "delayed" gelé. Astuce élégante : la
   compétence passée est ré-estimée *sur tous les buts* d'un coup (un forward
   par but), pas stockée par but.

Détails d'implémentation à retenir :
- `lp[lp < 0.01] = 0.0` — coupe le bruit de fond (sinon la masse résiduelle
  d'ALP bruitée dilue l'échantillonnage).
- Échantillonnage du buffer d'entraînement de l'estimateur **pondéré par
  récence** (`p ∝ arange(1, len+1)`) — l'estimateur suit la politique de près.
- Recalcul de l'ALP de TOUS les buts toutes les `recompute_freq=32` updates
  seulement (un forward par but, batché) — c'est ça qui rend le coût "high
  efficiency" dans leur Table 1.
- L'ALP est **absolu** (valeur absolue) : il capture aussi l'**oubli** — un but
  dont la compétence chute redevient prioritaire. Important pour nous : c'est
  exactement l'anti-catastrophic-forgetting qu'on veut dans un curriculum depth
  (ne pas sacrifier d1-d2 en poussant d3).

## 4. Ce que ça dit pour notre exp34 (autocurriculum TextCraft)

### Le choix de granularité est dicté par notre budget d'épisodes
- MAGELLAN complet : nécessite une 2e paire d'adapters + tête MLP entraînée en
  ligne DANS le trainer TRL (forwards supplémentaires côté HF, plomberie
  lamorel à réécrire pour TRL). Faisable mais ~3-4 jours d'ingénierie risquée
  — hors budget avant la soutenance. Sa valeur ajoutée (généralisation
  sémantique entre buts) est en outre **faible chez nous** : 3000 items, 4
  familles bien définies, pas de buts impossibles noyant l'espace.
- Online-ALP par item : ~3000 deques ; à ~3000 épisodes/epoch chaque item est
  vu ~1×/epoch → il faut ~8 epochs pour remplir une deque de 8 → signal trop
  lent.
- **EK-Online-ALP par depth : le bon choix.** 4 deques, remplies de centaines
  d'épisodes par epoch → LP par depth réactif en ~½ epoch. C'est leur
  meilleure baseline, dans le cas favorable (groupes = depth) où elle est
  presque équivalente à MAGELLAN. Et on garde MAGELLAN complet comme
  discussion/ouverture dans le rapport.

### Transposition concrète (design proposé pour exp34)
1. **Stats** : dans `rollout.py`, à chaque fin d'épisode, pousser (depth,
   succès) dans une deque par depth (maxlen ~200).
2. **LP par depth** : formule du demi-buffer ci-dessus, recalculée toutes les
   ~23 steps (½ epoch), avec le seuil `lp < 0.01 → 0`.
3. **Échantillonnage** : le dataset GRPO tire les items ∝ LP(depth(item)) avec
   plancher ε = 0.2 d'uniforme (fixe, pas recuit — nos runs sont courts et le
   recuit ajouterait un hyperparamètre).
4. **Amorçage** : LP initial uniforme tant qu'une deque n'a pas ≥ 2×K épisodes
   (équivalent de leur warmup).
5. **Logs** : sr / sr_delayed / lp par depth à chaque recalcul (wandb +
   fichier), pour la figure "trajectoire du curriculum émergent".

### La prédiction scientifique (à écrire AVANT de lancer — lien fil rouge)
Chez eux, 80 % des buts sont impossibles et le LP les évacue naturellement
(compétence figée à 0 → LP = 0 → jamais échantillonnés). Notre analogue est
**depth 4 : support nul (0 % @ pass@20)** → prédiction : l'autocurriculum va
détecter LP(d4) ≈ 0 et **désinvestir depth 4 de lui-même**, concentrant le
budget sur la frontière d2-d3. Si c'est observé, c'est la version
"comportementale" de l'argument sharpening : le signal de progrès dit à
l'agent où l'aiguisage est possible, et il dit d4 = mur. Résultat intéressant
que le run réussisse ou non à battre la baseline uniforme.

## 5. Papiers à ajouter au rapport via MAGELLAN
- Gaven et al. 2025 (MAGELLAN) — §2.3 état de l'art autocurriculum, et §4.3
  pour situer notre exp34 dans LEUR taxonomie (nous = EK-Online-ALP, groupes =
  depth ; justification : cas favorable des groupes experts).
- Lopes & Oudeyer 2012 (bandit multi-bras sur l'ALP) — le schéma de sélection.
- Baranes & Oudeyer 2013 (ALP absolu, progrès + oubli).
- Kanitscheider et al. 2021 (LP par ré-évaluation, Minecraft) — le coûteux que
  MAGELLAN évite.
- Gaven et al. 2024 (SAC-GLAM) et Carta et al. 2023 (GLAM/BabyAI-text) — si on
  parle de la lignée "LLM agents RL en ligne" de Flowers.
