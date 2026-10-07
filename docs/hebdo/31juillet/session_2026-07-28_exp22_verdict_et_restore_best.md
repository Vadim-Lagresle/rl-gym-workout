# Session 2026-07-28 — verdict exp22 (coupes sans restore = dérive figée) → exp22.1 restore du best train

## 1. Verdict exp22 (LR adaptatif au reward train, SANS restore)

Arrêt manuel à step 759/4600 (epoch 16.5) après ~13 h. Le critère adaptatif a
bien fonctionné *mécaniquement* (4 coupes ÷3 aux epochs 6.5 / 8.0 / 9.5 / 12.0,
pas de collapse KL), mais le run est un échec *scientifiquement instructif* :

| Signal | Trajectoire |
|---|---|
| Test Pass@1 | 33 → 40 → 46 → **49 @ step 200** → dégradation continue → 22-26 après step 400 |
| Rolling mean train (fenêtre 1 ep) | montée jusqu'à **0.4721** (epoch ~4.5-5) → chute → stagnation ~0.24 |
| LR | 5e-6 → 4 coupes → 6.2e-8 (gelé) |

**Le défaut structurel** : la coupe arrive par construction APRÈS la dérive
(il faut `patience` = 1.5 epoch de checks sous le best pour couper, et le best
lui-même date d'une fenêtre entière). La coupe #1 (epoch 6.5) est tombée ~1.5-2
epochs après le pic train — entre-temps le test était déjà passé de 49 à ~30.
Et comme on gardait les **poids courants**, chaque LR réduit *consolidait
l'état dégradé* au lieu du meilleur état : les coupes suivantes n'ont fait que
geler le modèle de plus en plus bas. C'est le miroir d'exp19.2 (dérive post-pic
à 5e-6), simplement sans l'explosion KL finale.

Leçon : **détecter la stagnation ne suffit pas, il faut pouvoir *revenir* au
meilleur état connu** quand on réduit le LR — sinon on polit la dérive.

## 2. exp22.1 — save/restore du best TRAIN à chaque coupe

Demande de Vadim (le « pourquoi pas repartir du best ckpt » du 27/07, précisé :
best **train**, jamais test — pas de peeking, le Pass@1 périodique reste un
estimateur honnête). Implémenté dans `RewardAdaptiveLrCallback` (`schedules.py`),
activé par `--lr-adaptive-restore-best` (`train_grpo.py`, LoRA uniquement) :

1. **Save à chaque nouveau best** de moyenne roulante train : l'adapter LoRA
   (~120 Mo) est écrit sur le disque persistant
   (`saves/trl_grpo/<run>_besttrain`), en swap atomique `.tmp` → dossier, ancien
   best remplacé. Un fichier `.besttrain_info` (step/epoch/mean/lr/beta/n_cuts)
   rend la reprise traçable → **point de redémarrage si coupure infra**.
2. **Restore à chaque coupe ÷3** : poids rechargés via
   `set_peft_model_state_dict` (mécanisme de continuation d'adapter validé le
   21/07 — injecte les poids SANS recréer d'adapter ni bouger l'ancre KL) +
   **moments Adam remis à zéro** (l'optimizer gardait l'élan de la trajectoire
   divergente). TRL resynchronise vLLM au step suivant (sync colocate par step).
3. **Cooldown propre** (nouvelle sémantique post-coupe) : historique vidé, best
   remis à None, prochain check seulement après une fenêtre **complète** de
   rewards post-coupe — la nouvelle référence ne mélange plus les régimes
   avant/après coupe (exp22 faisait `best = valeur courante`, ce qui pouvait
   ancrer la référence sur un creux).

**Validation avant lancement** :
- selftest CPU `python -m src.train.schedules` : 2 scénarios (sans/avec
  best_dir, hooks disque mockés) — 1 restore par coupe, cooldown qui saute le
  check suivant, re-save du best sous le nouveau régime ;
- smoke GPU 6 steps avec coupe forcée (`eps=-1`, patience 1) : save réel de
  l'adapter, restore + reset Adam au step 4, rollouts post-restore sains
  (preuve que la sync vLLM suit), re-save au step 6, `.besttrain_info` correct.

**Lancement** : `exp22.1_restore_best`, config strictement exp22 par ailleurs
(LoRA depuis zéro, k=10, LR 5e-6, beta 0.01, fenêtre 1 ep / check ½ ep /
patience 3 / eps 0.005, 100 epochs, éval test 100 items / 50 steps).
Fiche : `runs/10_fewshot_rl/exp22.1_restore_best/config.yaml`.

## 3. À surveiller demain matin

- Les lignes `[reward-adaptive-lr]` : `best train sauvé`, `>>> COUPE` suivi de
  `poids RESTAURÉS` — et surtout si le reward train **remonte vers le best**
  après une coupe (contrairement à exp22 où il restait en dessous).
- Risque identifié : oscillation best → dérive → restore → dérive sans progrès
  net = plateau atteint, stopper le run.
- Références : exp20 = 58/100 @ step 800 (à battre) ; exp22 = 49 @ step 200.
