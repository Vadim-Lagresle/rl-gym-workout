# Session 2026-08-27 — collapse d'exp36 (verdict : le curriculum est un régularisateur), lancement exp33, exp36.1 en file

## Le résultat du jour : exp36 collapse, et c'est le 1er verdict du contrôle N=16

exp36_n16 (recette exp25 stricte, seul delta N=16, '30:0' constant) tuée à 09:00
après 15 h 40 (SIGKILL nécessaire — boucle de génération dégénérée). Analyse
comparée des trois runs à recette identique (métriques extraites des logs,
médianes par tranche de 2 epochs) :

| | exp25 (N=8, 30 tours) | exp32 (N=16, horizon 10/20/30) | exp36 (N=16, 30 tours) |
|---|---|---|---|
| Éval | 12→35 (15 ep) → 65 | 12→**82** | 12→**54** (ep ~13) → 13 |
| Entropie | stable 0.98-1.13 | 0.65-0.83, stable, remonte | 1.0 → 0.59 (ep 8-10) → **0.10 (ep 10-12)** |
| KL | ~0.001 | ~0.001-0.002 | 0.003 → **6×10⁸ (ep 10-12)** |
| Reward @ ep 6-8 | 0.30 | 0.375 | **0.45** |
| Groupes zero-std | ~0.50 | ~0.50 | **0.25** (ep 2-8) |

**Chaîne causale** : N=16 densifie le signal GRPO (zero-std 0.25) → apprentissage
~2× plus rapide (reward 0.45 @ ep 7) → sharpening accéléré sur distribution
STATIQUE (30 tours dès le départ) → effondrement d'entropie ep 8-12 (politique
quasi déterministe, sorties ~180 tokens, reward train qui TIENT à 0.5 pendant que
l'éval chute 54→30 : politique cassante) → KL numériquement explosive malgré les
ré-ancrages ep 4/8/12 — **l'ancre mobile /4 ep SUIT la politique en cours de
déterminisation et fige la dégénérescence au lieu de la freiner** → runaway
(même signature terminale qu'exp31.1), générations détruites (8.5k tokens).

**Pourquoi les deux autres tiennent** : exp25 = même recette mais sharpening 2×
plus lent (gradients moyennés sur 8 items/step) — dans la zone où β/ancre/entropy
suffisent. exp32 = même vitesse N=16, mais le curriculum d'horizon ré-injecte de
la difficulté (10→20 ep 15, →30 ep 30) pile quand le sharpening s'installerait :
groupes re-mixés, entropie stabilisée.

**Lecture pour le rapport (figure quasi écrite)** : le curriculum d'exp32 n'est
pas qu'un accélérateur, c'est un **régularisateur d'exploration** ; N=16 amplifie
le meilleur (82) comme le pire (collapse), et c'est le curriculum qui décide du
côté. Nuance honnête : exp36 n'est pas un contrôle « propre » de N (confondu par
le collapse) — son 54 se cite comme « N=16 sans curriculum avant instabilité »,
plus rapide qu'exp25 à trajectoires égales.

## Décision méthodologique (échange avec Vadim)

La comparaison reste **à hyperparamètres égaux** (recette partagée de la phase 1) :
chercher la période d'ancre optimale par bras transformerait 2 runs en une grille
par bras, et toute config commune avantagera toujours un bras — le claim du
rapport est formulé conditionnellement (« à recette commune, le curriculum
survit et atteint 82 là où le sans-curriculum collapse ») ; les ablations d'ancre
(4 vs 8 vs 12 ep) sont notées comme travail ultérieur. Rappel utile : « 4 epochs »
n'est de toute façon pas le même hyperparamètre en temps-updates selon N
(184 steps à N=8, 372 à N=16). Corollaire (Vadim) sur le design d'exp36.1 : plus
honnête de comparer à **β fixe** en réglant la période d'ancre (curseur d'un
mécanisme existant) que de toucher β/entropy-coef (force nouvelle, delta de plus).
Note méthodologique complète copiée dans docs/RESULTS.md (après le bloc exp36).

## Actions

1. Dernier ckpt copié (`saves/trl_grpo/exp36_n16_checkpoint_1410`, post-collapse,
   analyse seulement) ; best 54 + optimizer déjà sur le home (step 1222).
2. Kill exp36 (SIGTERM sans effet ~1 min → SIGKILL) → job 43z clos (exit 0,
   RESULTS.md rempli et interprété) → **exp33_depth lancée à 09:00** (job 44).
   Démarrage vérifié : selftests OK, `[depth-schedule] palier depth<=1 :
   109/374 items`, sampler pondéré actif, N=16, ancre mobile armée.
3. **Job 48_exp36.1_anchor12 créé en FIN de file** (option 3 de Vadim) : exp36 à
   l'identique, SEUL delta `--moving-anchor-every-epochs 12` (proposition Vadim
   8/10/12 ; 12 retenu : le frein reste Qwen nu — entropie maximale — sur toute
   la fenêtre critique ep 8-12, et il reste ~6 ré-ancrages sur 80 ep). β et
   entropy-coef inchangés : un seul delta. Passera après exp33/34/35 si le
   calendrier le permet (gel 2/09).
4. Docs : INDEX (exp36 clos + lignes exp36.1 et exp33), RESULTS.md, plan de
   rapport §4.3 (verdict préliminaire mis à jour).

## File au 27/08 09:15

exp33_depth (EN COURS) → exp34_magellan → exp35_budget256-1024 → exp36.1_anchor12.
Gel GPU 2/09 : ~5,5 jours pour 3-4 runs → arrêts au plateau, exp36.1 sacrifiable.
