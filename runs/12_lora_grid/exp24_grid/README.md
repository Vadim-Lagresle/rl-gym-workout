# exp24 — grille d'hyperparamètres LoRA sur la recette exp23 (« LoRA Without Regret »)

Décidée le 2026-08-11, après le 69/100 d'exp23.4 (full-FT). Question : **LoRA peut-il
égaler le full-FT sur notre tâche RL, et à quels (rang, LR, beta) ?**

## Ancrage : findings du blog Thinking Machines

Source : [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman et al.,
sept. 2025). Ce qu'on en retient pour cette grille :

1. **En RL, LoRA égale le full-FT même à rang 1** — le policy gradient n'absorbe que
   ~1 bit/épisode, la capacité n'est jamais le facteur limitant. La grille de rangs
   teste donc la *dynamique*, pas la capacité.
2. **LR optimal LoRA ≈ 10× le LR optimal full-FT** (multiplicateur fitté : 9.8, constant
   sur 14 modèles Llama/Qwen). Notre full-FT optimal : 1e-6 → prédiction LoRA : **1e-5**.
3. **LR optimal ~indépendant du rang** grâce au préfacteur α/r (< 2× d'écart entre r=4
   et r=512) — MAIS ceci suppose **α fixe pour tous les rangs** (eux : α=32).
   Notre convention historique α=2r est donc remplacée ici par `--lora-alpha 32` partout.
4. **LoRA sur TOUTES les couches** (MLP surtout ; attention seule sous-performe même à
   paramètres égaux). Notre code cible déjà q,k,v,o + gate,up,down — conforme.
5. Mise en garde : **LoRA tolère moins bien les gros batches** (pénalité indépendante du
   rang). Notre buffer de 256 trajectoires est concerné — c'est un des points que la
   grille mesurera de fait (on garde 256 pour la comparabilité avec exp23).

## Design

**Base commune = recette exp23 à l'identique**, seuls changent LoRA/LR/beta :
zero-shot, vrai PPO (buffer 256 → 4 updates clippées eps 0.2), bonus entropie 0.001,
max_rounds 30, LR constant, adamw fp32 (défaut LoRA), ancre KL = Qwen nu,
**100 epochs** (~4400 steps, ~40 h/run) — décision Vadim 11/08 : budget long d'emblée,
IL surveille et stoppe manuellement un run qui plafonne ou s'effondre (`kill <pid>` du
train : la file passe alors automatiquement au job suivant).

3 rangs × 3 combos (LR, beta) = **9 runs** :

| Combo | LR | beta KL | Rationale |
|---|---|---|---|
| **A** | 1e-5 | 0.001 | la prédiction du blog (10× full-FT), beta de la recette exp23 |
| **B** | 3e-6 | 0.001 | conservateur (3×) — borne basse de la U-curve |
| **C** | 1e-5 | 0.01 | 10× mais ancre KL forte (notre beta LoRA historique) — teste si la KL doit suivre le LR |

Ordre d'exécution (file `runs/queue/`) : combo A sur les 3 rangs d'abord (teste le claim
principal + l'indépendance au rang dès les 3 premiers runs), puis B ×3, puis C ×3.

## Grille de lecture

- **Barre full-FT à budget égal : exp23 était à ~30-32/100 à epoch 15** (steps 658-705).
  Un combo LoRA qui fait ≥ 30 à 15 ep « égale le full-FT » au sens du blog.
- Si A >> B : le 10× du blog transfère chez nous. Si B > A : notre tâche multi-tour
  (bruit du reward sparse) demande plus de prudence que leurs benchs maths.
- Si les 3 rangs d'un même combo sont ~confondus : l'indépendance au rang tient (α=32
  fixe) → prendre r=16 pour les runs longs (adapter 4× plus petit).
- C vs A : la KL saine est ~0.001-0.01 chez nous ; si C stabilise sans ralentir,
  l'ancre forte est gratuite ; si C plafonne sous A, beta 0.01 bride à LR 1e-5.
- Signes de LR trop haut (leçon exp22.2) : KL > 0.05 soutenue avant epoch 3, entropie
  qui s'effondre < 0.15, évals test qui décrochent pendant que le reward train monte.

## Restructuration du 2026-08-12 (verdict du 1er run)

`exp24_r64_lr1e-5_b0.001` : **COLLAPSE**. Pic 24/100 (step 94-141) puis dégringolade
monotone jusqu'à 0/100 (step 658+) — politique produisant ~1800 tokens/trajectoire de
bruit, 0 action valide sur 30 rounds. **La prédiction 10× du blog ne transfère PAS telle
quelle** sur notre setup (RL multi-tour sparse + gros batch 256, là où le blog
teste du single-turn math — leur propre caveat batch s'applique).

Note vitesse (question du 12/08) : le run faisait ~74 steps/h au départ (≥ exp23 scratch,
~64 st/h « 56 s/step ») puis a ralenti à ~34 st/h de moyenne À CAUSE du collapse
(génération 3× plus longue, 30 rounds brûlés par épisode). Ni vLLM (0.26 = version
d'exp23.3/23.4), ni la file, ni un changement de code. Les ~137 st/h en mémoire
venaient d'exp23.4, warm-startée sur une politique efficace (épisodes courts).
**Run LoRA sain de 15 ep ≈ 9-11 h** ; un run effondré ralentit mécaniquement.

Grille révisée : **15 epochs par run** (l'écran suffit), arm (1e-5, 0.001) arrêtée —
r16/r32 parqués dans `runs/queue/skipped/` (l'indépendance au rang prédit le même
collapse) — et **nouvelle arm LR 1e-6** (le LR full-FT, ratio 1×).

Second ajustement (même jour, analyse des LoRA historiques) : l'arm (1e-5, 0.01) est
remplacée par **(3e-6, 0.01)**. Lecture en « chaleur effective » α·LR (la mise à jour
LoRA ∝ α·LR) : tous les collapses du projet sont à α·LR ≥ 3e-4 (exp10, exp22.2, exp24
job 1), la zone d'apprentissage historique est ~1.6e-4, TOUJOURS avec β=0.01 (garde-fou
adopté après le collapse d'exp10 à β=0.001 — que le job 1 a rejoué). (1e-5, 0.01)
= 3.2e-4 : l'historique prédit le collapse malgré l'ancre. (3e-6, 0.01) teste
« β 0.01 vs 0.001 à chaleur égale (9.6e-5) » contre l'arm 04-06 — l'ablation la plus
informative.

## Incident du 12-14/08 : purge pod silencieuse

Le job 04 est **mort le 12/08 à ~17h16** (purge pod : /tmp vidé, tous les process tués —
runner, textcraft, train). Il tournait à vitesse normale (~73 st/h) jusqu'à l'epoch ~10.6 ;
le GPU est resté idle ~2 jours avant détection le 14/08 (le check de vie `pgrep` s'auto-
matchait sur le wrapper shell — leçon : vérifier `nvidia-smi` + mtime du log, pas un pgrep).
Sa lecture est considérée acquise (10.6/15 ep) : **plateau 15→25→19, pic 25/100** — à peine
au-dessus de la baseline 18, loin de la barre full-FT (~30 à ep 15). Non rejoué.

Relance du 14/08 : file réordonnée — **le job β0.01 r64 passe en tête** (ablation β vs
job 04 à chaleur égale = la lecture la plus informative), l'arm 1e-6 passe en dernier.

## Runs

| Ordre file | Run | r | LR | beta | statut |
|---|---|---|---|---|---|
| — | `exp24_r64_lr1e-5_b0.001` | 64 | 1e-5 | 0.001 | **collapse** (pic 24 → 0/100, tué step ~760/ep 17) |
| — | `exp24_r16/r32_lr1e-5_b0.001` | 16/32 | 1e-5 | 0.001 | parqués (`queue/skipped/`) |
| — | `exp24_r64_lr3e-6_b0.001` | 64 | 3e-6 | 0.001 | **mort purge ep 10.6** — plateau 15→25→19, pic 25/100 (lecture acquise, non rejoué) |
| 21 | `exp24_r64_lr3e-6_b0.01` | 64 | 3e-6 | 0.01 | **terminé 15/08** (15 ep, fin propre) — stable 21-29, fin 27, ZÉRO collapse ; pic 29 |
| 22 | `exp24_r16_lr3e-6_b0.001` | 16 | 3e-6 | 0.001 | **terminé 15/08** — monte à **35** @ ep ~12 (> barre full-FT !) puis **collapse** (2-15/100 sur les 3 dernières évals) |
| 23 | `exp24_r32_lr3e-6_b0.001` | 32 | 3e-6 | 0.001 | **terminé 15/08** — pic 31 @ ep ~6 puis **collapse** (2-16/100 sur la 2e moitié) |
| 24 | `exp24_r16_lr3e-6_b0.01` | 16 | 3e-6 | 0.01 | mort purge pod 16/08 00h01 après 1 éval (15/100) — **relancé 17/08** (rejeu complet) |
| 25 | `exp24_r32_lr3e-6_b0.01` | 32 | 3e-6 | 0.01 | en file |
| 26-28 | `exp24_r{64,16,32}_lr1e-6_b0.001` | 64/16/32 | 1e-6 | 0.001 | en file |

## Lecture au 17/08 (ablation β répondue)

À chaleur α·LR égale (9.6e-5), **β tranche tout** :
- **β 0.001** : monte plus vite et plus haut (31-35, r16 dépasse la barre full-FT ~30-32
  à ep 15) mais **collapse systématiquement en 2e moitié de run** — les 3 rangs.
- **β 0.01** : aucun collapse en 15 ep, mais plafonne ~27-29 (r64).

Le garde-fou historique β=0.01 pour LoRA est re-confirmé en zero-shot. Et le pic 35 de
r16/β0.001 montre que LoRA PEUT égaler le full-FT à budget égal — si on le protège du
collapse (β fort, ou LR decay / arrêt au pic : le best adapter @ 35 est sauvé).
Rangs : r16 ≥ r32 > r64 à β0.001 — le petit rang n'est pas le facteur limitant,
conforme au blog.

### Caveat : le 10× du blog ne transfère pas du single-turn dense au multi-turn sparse

Le multiplicateur 10× (LR LoRA optimal = 10× le LR full-FT) est mesuré par le blog sur
du single-turn à reward dense (maths Tulu3/OpenThoughts) : gradient informatif à chaque
exemple, épisode = 1 génération. Chez nous — RL multi-tour (jusqu'à 30 rounds), reward
sparse 0/1 en fin d'épisode, batch 256 (leur propre caveat gros batch) — le 10× (1e-5)
collapse : pic 24 → 0/100. Notre zone de travail mesurée est **3× (3e-6), avec β=0.01
pour tenir 15 ep sans collapse** (β=0.001 monte plus haut mais explose en 2e moitié).
S'ajoute leur « warmup implicite » (B part de zéro → LR effectif croissant en cours de
run) : le danger d'un LR trop haut grandit avec les steps — cohérent avec nos collapses
TARDIFS à β0.001. Enfin, **le blog ne dit rien du beta KL** : leurs runs RL sont du
policy gradient sans ancre KL discutée ; notre ablation β est orthogonale à leurs
findings et spécifique au régime long multi-tour sparse.

Budget restant : 9 × ~9-11 h ≈ **4 jours GPU** (file séquentielle, `scripts/run_queue.sh`),
arrêts manuels de Vadim en cas de collapse (un run effondré gaspille en plus du temps).
Bests : adapters seuls (120-480 Mo/run, pas d'optimizer.pt — runs d'écran), seuil -1
(première éval = premier point de reprise).

Résultats : à consigner dans ce tableau (Pass@1 max et @ ep 15) + INDEX.md.
