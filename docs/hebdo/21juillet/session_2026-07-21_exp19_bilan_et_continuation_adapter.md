# Session 2026-07-21 — Bilan exp19 (RL + few-shot) et mécanisme de continuation « même adaptateur, même ancre »

Référence : `runs/10_fewshot_rl/{exp19_fewshot_rl_k10,exp19.1_warmstart43,exp19.2_scratch_5e-6}/config.yaml`
pour le détail chiffré de chaque run. Ce document résume la chronologie et prépare
**exp19.3**, prête à coder/lancer à la prochaine session (rien n'a été relancé aujourd'hui).

## 1. Résultats des runs précédents (rappel, weekend du 17 au 21/07)

| Run | Point de départ | LR | Résultat | Verdict |
|---|---|---|---|---|
| **exp19** | Qwen base + k=10 | 7,333e-7 (régime polissage exp10.8, mal assorti) | 43 pic / **~35 réel** @ step 400 | interrompu step 847 par l'infra (purge /tmp), pas un bug ; adaptateur mince (‖B‖≈0,12, signature « polissage ») |
| **exp19.1** | merge 43 % + adaptateur neuf | 7,333e-7 (même) | ~33-37 (quasi immobile) | ablation LR involontaire : LR de polissage depuis une ancre basse ≈ rien n'apprend |
| **exp19.2** | Qwen base + k=10 (échelle rejouée) | **5e-6** (régime apprentissage exp10.3) | **26→45 (step 300, sain)** puis **collapse KL** steps 350-450 (KL ×3500, complétions ×3, pass@1→2-5) | PAS du reward hacking (reward sparse sans shaping, il s'est effondré) — collapse classique. **Best 45/100 sauvé avant la casse** (`saves/trl_grpo/exp19.2_scratch_5e-6_best`) |

**État au 21/07** : tous les runs sont arrêtés (aucun training actif, GPU libre). Le meilleur
résultat honnête de toute la lignée few-shot est **45/100 au step 300 d'exp19.2** — au-dessus
du pic bruité d'exp19 (43, winner's curse) et proche du best historique du projet (54, exp10.8,
sans few-shot). C'est ce checkpoint qui sert de point de départ pour la suite.

**Leçon transversale confirmée deux fois ce weekend** : la norme de l'adaptateur (`‖B‖`) est
un diagnostic fiable du régime de LR — apprentissage réel (exp10.5 : 0,61 ; exp19.2 : 0,75)
vs polissage quasi-inerte (exp10.7/10.8/exp19 : 0,10-0,13). Le LR doit toujours être choisi
en fonction du niveau de l'ancre de départ, pas recopié tel quel d'une étape précédente.

## 2. La question du 21/07 : comment continuer sans déplacer l'ancre KL ?

Objectif demandé : reprendre depuis le best45, **LR ÷3** (≈1,667e-6, entre le régime
d'apprentissage 5e-6 qui a collapsé et le régime de polissage 7,33e-7 qui n'apprenait
rien), **sans merger** l'adaptateur dans une nouvelle base (ce qui déplacerait l'ancre KL
vers 45 %, comme l'avait fait le warm-start exp19.1), et **sans repartir d'un adaptateur
neuf** (B=0) qui perdrait tout l'acquis du best45.

### Le piège identifié (lecture du code TRL 1.4.0 installé)

Intuition naturelle mais **fausse** : charger le best45 sur un `PeftModel` posé sur Qwen
(sans merge dense), puis passer ce `PeftModel` déjà instancié au nouveau `GRPOTrainer`.

Lecture de `trl/trainer/grpo_trainer.py:359-371` : si `model` est déjà un `PeftModel` (pas
une chaîne) et qu'aucun `peft_config` n'est passé, TRL prend la branche
`elif is_peft_model(model) and args.beta != 0.0:` — elle **clone l'adaptateur chargé en un
second adaptateur `"ref"` figé** (copie exacte des poids au moment de la construction), et
calcule la KL contre cette copie. Résultat : l'ancre se retrouve quand même sur le best45,
par un mécanisme différent (clonage d'adaptateur) mais avec le même effet qu'un merge dense.
**Passer un `PeftModel` pré-construit à un nouveau trainer reproduit donc silencieusement
le problème qu'on veut éviter.**

### Le mécanisme qui marche (vérifié bout en bout, voir `[rollout.py]`/session)

1. Laisser TRL construire l'adaptateur **neuf** comme d'habitude : `model` passé comme
   **chemin** vers Qwen (pas un objet), `peft_config=LoraConfig(...)` comme d'habitude.
   TRL prend alors la PREMIÈRE branche (`get_peft_model(model, peft_config)`), crée un
   adaptateur `"default"` initialisé standard (`B=0`), et **aucun** adaptateur `"ref"`
   n'est ajouté.
2. **Juste après la construction du `GRPOTrainer`** (le modèle n'est pas encore wrappé par
   `accelerator.prepare`, qui n'a lieu qu'au début de `.train()`), injecter les poids
   appris du best45 dans cet adaptateur `"default"` via l'API PEFT native :
   `peft.set_peft_model_state_dict(trainer.model, load_file(adapter_path), adapter_name="default")`.
3. Résultat vérifié sur le vrai modèle (Qwen2.5-3B + best45) :
   - avant injection : `‖B‖ = 0,000000` (init standard) ;
   - après injection : `504` clés chargées, `0` incohérente, `‖B‖ = 0,749` (cohérent avec
     un adaptateur de régime d'apprentissage, comme attendu du best45 à 45/100) ;
   - `model.peft_config.keys() == ['default']` — **aucun** `'ref'` créé ;
   - `model.disable_adapter()` révèle **Qwen pur**.

Donc : un seul adaptateur existe pour TRL ; le désactiver pour le calcul de la KL révèle
Qwen brut, sans second mécanisme d'ancrage. Les deux contraintes sont satisfaites
simultanément : **ancre KL = Qwen inchangée**, **même adaptateur qui continue** (poids
injectés, pas de redémarrage à B=0). Seule concession (déjà actée précédemment dans le
projet, jugée mineure) : l'état de l'optimiseur Adam (momentum/variance) n'est pas repris,
il repart de zéro — pas de checkpoint TRL complet disponible ici, seul l'adaptateur seul
a été sauvé par le callback d'éval.

## 3. Plan pour la prochaine session — exp19.3 (rien lancé, à valider avant exécution)

### Code à écrire
- Nouvel argument CLI `--adapter-init <chemin>` dans `train_grpo.py` : chemin vers un
  dossier d'adaptateur sauvegardé (ex. `saves/trl_grpo/exp19.2_scratch_5e-6_best`).
- Juste après `trainer = GRPOTrainer(...)` (avant `trainer.train(...)`) : si
  `--adapter-init` est fourni, charger `adapter_model.safetensors` et appeler
  `set_peft_model_state_dict(trainer.model, state_dict, adapter_name="default")`.
- Garde-fou : `--adapter-init` incompatible avec `--full-ft` (pas d'adaptateur en full-FT).

### Validation avant tout lancement GPU long
- Smoke à sec : vérifier que l'injection charge bien les 504 clés attendues, `‖B‖` non
  nul après injection, `disable_adapter()` révèle Qwen (comme vérifié aujourd'hui en
  standalone — à refaire une fois le code intégré dans `train_grpo.py`).
- Smoke GPU 1 step : rollout + éval périodique en condition k=10, comme pour tous les
  runs précédents.

### Paramètres proposés pour le run (à confirmer avant lancement)
- Base : `models/Qwen2.5-3B-Instruct` (chemin, pas de merge).
- `--adapter-init saves/trl_grpo/exp19.2_scratch_5e-6_best`.
- `--learning-rate 1.667e-6` (5e-6 ÷ 3 — entre le régime d'apprentissage qui a collapsé
  et le régime de polissage qui n'apprenait rien).
- `--beta 0.01` (inchangé).
- `--fewshot 10` (mêmes exemples, format dialogue — inchangé).
- `--best-init-score 0.45` (ne jamais sauver pire que l'acquis actuel).
- `--eval-every` : **25** proposé (au lieu de 50) pour repérer un début de dérive KL plus
  tôt, vu que le collapse d'exp19.2 s'est joué en ~100 steps (350→450) entre deux points
  d'éval à 50 steps — à confirmer avec Vadim, alternative : garder 50 si jugé suffisant.
- Nom de run proposé : `exp19.3_continue_best45_lrdiv3` (famille `runs/10_fewshot_rl/`).

### Surveillance recommandée pendant le run
- `kl` sur wandb : le collapse d'exp19.2 était détectable dès `kl` franchissant l'ordre de
  ~0,01-0,05 de façon soutenue (contre ~0,0008 stable en régime sain) — un arrêt manuel
  anticipé à ce seuil éviterait d'attendre le prochain point d'éval.
- `completions/mean_length` : doublait/triplait pendant le collapse (590-741 → 1700-2037) —
  un signal parallèle à la KL, plus facile à lire d'un coup d'œil sur la courbe wandb.
- `grad_norm` : montée de 0,05-0,08 (régime sain) vers 0,4-0,8 (début de collapse).

**Aucun run n'a été lancé aujourd'hui.** Prochaine étape : coder `--adapter-init`, valider
à sec puis en smoke GPU, obtenir la confirmation des paramètres ci-dessus, puis lancer.
