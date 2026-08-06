# Session 2026-07-29 — autopsie exp22.1 (coupes sur bruit), ménage disque, exp22.2 (rang 64 + critère durci)

## 1. Autopsie exp22.1

Le run est mort tout seul à 06:45 (step 2158, epoch 46.9) : **purge de `/tmp`**
(l'interpréteur Python du run y vivait, les checkpoints périodiques aussi).
Bilan scientifique en deux temps :

**Ce qui marche** — le restore du best train : après chacune des 7 coupes, le
reward remonte au niveau du best (0.47-0.50) au lieu de s'effondrer (exp22 :
0.47 → 0.24). Test stable 42-53 sur 46 epochs, best 53/100 @ step 850.

**Ce qui casse** — les coupes étaient pilotées par le bruit :
- la référence « best » est un **max de fenêtres bruitées** → biaisée +1-2σ
  (tirage chanceux d'items/générations) ;
- eps = 0.005 avait été calibré sous hypothèse iid (σ ≈ 0.009), mais le bruit
  inter-fenêtres **mesuré sur politique gelée** (epochs 28-46, LR 2.3e-9 : la
  moyenne roulante fluctue 0.4253-0.4752 sans qu'aucun poids ne bouge) est de
  ±0.015-0.02, soit 3-4× eps ;
- résultat : coupes #2-#4 espacées de 2.5 epochs = la **cadence minimale
  possible** du mécanisme (1 ep cooldown + 0.5 réf + 3×0.5 patience). Un
  détecteur qui sature sa fréquence max ne détecte rien : il échantillonne du
  bruit. LR au plancher 2.3e-9 dès l'epoch 28 → 18 epochs brûlées à LR ≈ 0.

Autre signal : **collapse d'entropie** 0.95 → 0.12 dès le step ~400 (politique
quasi déterministe ; frac_reward_zero_std jusqu'à 0.875 = 7 groupes GRPO sur 8
sans variance de reward donc sans gradient). Starvation d'exploration
indépendante du LR — argument de plus pour la piste best-of-N/distillation.

Constat transversal : exp19.2 / 20 / 22 / 22.1 / lignée exp10 plafonnent TOUTES
à train ~0.47-0.50, test ~45-55. Quatre stratégies de schedule, même mur → le
problème n'est plus l'optimisation mais la capacité/exploration.

## 2. Ménage disque (home 35 Go, était à 100 %)

Politique actée : **le home ne stocke QUE les best (+ optimizers)**.
- Supprimés : `exp17_snis_v1_best` (5.8 Go, modèle complet d'un run raté),
  `wandb/` local (1.3 Go, tout est sur le serveur), env legacy
  `~/envs/agentgym-rl` (12 Go, stack v1 — `agentenv-textcraft` CONSERVÉ),
  `models/Qwen2.5-3B-Instruct` du home (5.8 Go).
- Le modèle vit sur `/tmp/models/` (copié), retéléchargeable en ~2 min via le
  nouveau `setup/ensure_qwen_tmp.sh` (HF joignable, vérifié).
  `DEFAULT_MODEL_PATH` (src/train/data.py) pointe sur /tmp.
- Les 11 adapters best regroupés dans `saves/keep_best/` (1.4 Go, README).
- `setup_agentgym_rl_v2.sh` corrigé : base Python = pyenv 3.11.7 (l'ancienne
  base était l'env legacy supprimé).
- Résultat : 9.4 Go utilisés / 35 (28 %).

## 3. exp22.2_rank64_median — capacité + critère durci

Deux leviers (fiche : `runs/10_fewshot_rl/exp22.2_rank64_median/config.yaml`) :

1. **Rang LoRA 16 → 64** (α 32 → 128, ratio 2 conservé ; `--lora-r/--lora-alpha`
   désormais en CLI) — teste le volet capacité du mur commun aux runs r=16.
2. **Critère adaptatif durci** (`RewardAdaptiveLrCallback`, nouveaux params) :
   - `--lr-adaptive-ref-median-k 5` : la stagnation se mesure contre la
     MÉDIANE des 5 derniers checks du palier (niveau typique) et plus contre
     le max historique (tirage chanceux) — le max ne sert plus qu'à choisir
     quand SAUVER le best ;
   - `--lr-adaptive-eps 0.02` : ~2σ du bruit mesuré ;
   - `--lr-adaptive-min-stage-epochs 3` (demande Vadim) : chaque palier de LR
     dure au moins 3 epochs — une coupe déclenchée avant est « retenue »
     (n_bad conservé) et n'est exécutée que si la stagnation persiste ;
   - `--lr-adaptive-min-lr 1e-7 --lr-adaptive-stop-at-floor` : plancher
     réaliste qui ARRÊTE le run (5e-6 → 3 coupes max → 1.85e-7 → stop) ;
   - `--lr-adaptive-save-optimizer` : l'état Adam (fp32, ~2× l'adapter) est
     sauvé avec le best train et RESTAURÉ à la coupe (poids + moments
     cohérents, du moment du best) — et reprise exacte après coupure infra.

Validation : selftest CPU étendu (test 3 : zéro coupe sur 40 epochs de bruit
pur avec la médiane, là où le max coupait ; coupe retenue jusqu'à l'epoch 3
sur vraie régression ; arrêt au plancher) + smoke GPU avant lancement.

## 4. Verdict exp22.2 (soir) et lancement exp22.3

**exp22.2 arrêté à step 671** (epoch ~14.5, ~7 h). Le diagnostic tient en une
phrase : on a donné au « gros camion » r=64 le régime moteur du r=16.

- **Dérive violente dès l'epoch 3** : spike KL ~12 000 et grad_norm ~214 000
  (steps 115-119). Avec 4x plus de paramètres entraînables à même LR par
  paramètre, l'adapter s'éloigne ~4x plus vite de l'ancre KL (Qwen nu).
- **Le test ne s'en remet jamais** : 47/100 @ step 50 (avant le spike, jamais
  rebattu) → 25 @ 150 → plateau bruité 26-45. Depths : d1 0.74, d2 0.29,
  d3 = d4 = 0 — mur identique aux runs r=16.
- **Le critère durci a bien fonctionné mécaniquement** : 3 coupes ÷3 (epochs
  3.5, 8.5, 12.0), palier min 3 ep respecté, restore poids + moments Adam OK
  à chaque coupe. Mais il stabilisait un état déjà dégradé : le rolling mean
  train plafonne à 0.44, SOUS le plafond r=16 (~0.47-0.50).
- Bests archivés : `saves/keep_best/exp22.2_rank64_median_{best,besttrain}`.

**exp22.3_rank64_lr_div4** (wandb `r4yjswfs`) : ablation à UNE variable — le
LR initial passe de 5e-6 à **1.25e-6** (÷4, demande Vadim), beta reste 0.01
(le levier beta×3 est gardé en réserve si le LR ÷4 ne suffit pas). Tout le
reste est identique à exp22.2. Avec le plancher 1e-7 : 1.25e-6 → 4.2e-7 →
1.4e-7 → stop (2 coupes max).

Question posée : **à départ stable, r=64 dépasse-t-il le plafond train r=16 ?**
- Critère 1 : pas de spike KL dans les epochs 1-5 (KL saine ~0.001-0.01) ;
- Critère 2 : rolling mean train > 0.44 (pic 22.2) puis > 0.50 (plafond r=16)
  — c'est LE test de l'hypothèse capacité ;
- Critère 3 : test > 47 (best 22.2), viser > 53 (exp22.1).

Si le train replafonne à ~0.47-0.50 malgré un départ propre, l'hypothèse
capacité est réfutée → le mur est un problème de SIGNAL (récompense sparse,
exploration) et on passe aux pistes best-of-N/BOND, curriculum, mix SFT.

Premier step sain : KL 0, grad_norm 0.12, reward 0.30, entropy 0.95, ~50 s/it
(montée du rolling mean attendue plus lente qu'exp22.2 — LR ÷4 ; premières
coupes pas avant l'epoch ~6-8 si tout va bien).

## 5. Verdict exp22.3 (31/07) et exp22.4 — calendaire ÷2 / 4 epochs

exp22.3 arrêtée à l'epoch ~28.7 : la stabilité est acquise (zéro spike KL,
test 36 → 49 sur les epochs 1-11) mais le test décroche ensuite alors que le
train se maintient — les coupes adaptatives arrivent APRÈS le décrochage
(epochs 9.5 et 13). Et le pic train 0.478 = plafond r=16 : l'hypothèse
capacité prend un 2e coup. exp22.4 (wandb `o9wva7cn`) repasse au calendaire,
version douce : LR et beta ÷2 toutes les 4 epochs, départ 1.25e-6, r=64,
24 epochs / 6 paliers — couper pendant la montée plutôt qu'après le
décrochage. Critères : pic tenu au-delà de l'epoch 11, battre 49.

## 6. exp22.4 (calendaire court) → exp22.5 : restore du best de palier (31/07)

exp22.4 (÷2 / 4 epochs) reproduit le symptôme d'exp22.3 : pic 50/100 vers
l'epoch 5, érosion ensuite — couper tôt ne suffit pas. Leçon de la lignée :
la coupe consolide l'état COURANT, qui après le pic est déjà une dérive.
Le facteur manquant n'est pas le timing de la coupe mais le POINT DE DÉPART
du palier suivant.

exp22.5 (idée Vadim) = `StagedBestRestoreCallback` : paliers de 10 epochs
minimum (mesures stables), best du palier sauvé en continu (adapter +
moments Adam, moyenne roulante train) + best global promu sur disque, et à
chaque frontière de palier : restore du best du palier écoulé puis LR/beta
÷2. La dérive de fin de palier est jetée au lieu d'être consolidée.
Wandb `096kat76`, r=64, départ 1.25e-6, 50 epochs = 5 paliers (week-end).
Lecture lundi : le palier 2 (LR 6.25e-7 depuis le best du palier 1)
dépasse-t-il 50 ? Ensuite battre 58 (exp20).
