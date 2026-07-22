# Session 2026-07-22 — vLLM récent installé et testé : Qwen3.5-4B fonctionne (isolé, pas encore migré)

## 1. Rappel : la découverte de la mise à jour glibc n'avait jamais été écrite

Le glibc 2.28 (bloquant vLLM récent, Qwen3.5, flash-attention) est mentionné partout dans le
projet (`CLAUDE.md`, `docs/hebdo/5juin/session_2026-06-01.md`). Une conversation antérieure
(même session que ce document, non commitée sur le moment) avait établi que **la VM
d'entraînement a été migrée vers CentOS Stream 10, glibc 2.39** — bien au-delà du plafond
2.28 documenté. Ce constat n'avait jamais été écrit ni dans `WORKLOG.md` ni dans `docs/`,
d'où la question de Vadim aujourd'hui pour le retrouver. **C'est réparé par cette note et
l'entrée WORKLOG correspondante.**

Rappel du contexte de la découverte : un ticket Jira (CODEX-3453, résolu 24/06 par Nassim
Ennour) avait ajouté une image EL10 aux templates **Coder** — plateforme distincte de la
JupyterHub B200 utilisée ici. La VM d'entraînement (`ws-vlagresle-agentgyrl-*`) a été relancée
sur cette base EL10 le 2026-07-20 (âge du `/` constaté ce jour-là). Le venv de training
(`~/envs/agentgym-rl`) restait inchangé : vLLM 0.9.1 toujours installée, glibc 2.39 seulement
un PLAFOND plus haut, pas une mise à jour automatique des paquets déjà installés.

## 2. Test réalisé aujourd'hui : installation isolée + Qwen3.5-4B validé de bout en bout

**Protocole** : rien touché dans `~/envs/agentgym-rl` (le venv de production). Nouveau venv
entièrement sur l'overlay `/tmp` (le home était déjà à 80-93 % plein pendant les tests — voir
incident disque ci-dessous), `python -m venv /tmp/envs/vllm-recent-test`, puis
`pip install vllm --cache-dir /tmp/pip-cache`.

**Résultat de l'installation** : **vLLM 0.25.1** (torch 2.11.0, transformers 5.14.1, CUDA 13) —
très au-delà de la fenêtre 0.12.0-0.18.0 que TRL réclame dans ses warnings actuels.

**Incident disque pendant l'installation** : la première tentative (venv sur le home) a rempli
`/home/criteo` à 100 % (cache pip 4,6 Go + venv partiel 2,6 Go — l'écosystème vLLM récent
tire CUDA 13 complet, flashinfer, tilelang : nettement plus lourd que l'installation actuelle).
Récupéré par `pip cache purge` + reconstruction entière sur `/tmp` (91-194 Go libres selon
le moment). **Le home (35 Go, quasi plein en permanence sur cette VM) ne peut plus accueillir
d'installation Python de cette taille — toujours utiliser `/tmp` pour ce genre de test.**

**Registre d'architectures** (`ModelRegistry.get_supported_archs()`) : confirmation que
`Qwen3_5ForConditionalGeneration` est bien présent (absent de la 0.9.1), ainsi que
`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MTP` (spéculatif), et toute la famille Qwen3
(`Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3VLForConditionalGeneration`,
`Qwen3ASR*`, etc.) — bien plus large que ce que 0.9.1 connaissait.

**Test fonctionnel réel** (pas seulement l'import du registre) :
- Qwen3.5-4B téléchargé sur `/tmp/models/Qwen3.5-4B` (8,8 Go, HF Hub, ~30 s).
- Chargement + génération réels : `llm.generate(["The capital of France is"], ...)` →
  `" Paris.\nA. True\nB. False\nAnswer:\nA..."`. Moteur chargé en 89 s (compilation JIT +
  capture CUDA graphs comprises), ~156 tok/s en sortie.
- **Architecture GPU détectée : `sm100` (Blackwell/B200) nativement reconnue** par les kernels
  FlashInfer — la vLLM 0.9.1 actuelle date d'avant le support B200 natif.
- **Qwen2.5-3B-Instruct (modèle de PRODUCTION actuel) testé aussi** avec cette même vLLM
  0.25.1 : charge et génère sans problème (~376 tok/s) — pas de régression de compatibilité
  sur le modèle utilisé aujourd'hui.

**Piège rencontré et résolu** : le premier essai de chargement échouait
(`FileNotFoundError: ninja`) — le process EngineCore (sous-process séparé de vLLM V1) ne
trouvait pas `ninja` (requis par FlashInfer pour la compilation JIT de kernels) car le venv
n'était pas sur le `PATH` (binaire python appelé directement, sans activation). Corrigé par
`export PATH="/tmp/envs/vllm-recent-test/bin:$PATH"` avant le lancement — **si ce venv est
réutilisé, toujours l'activer (ou exporter son PATH), pas juste appeler son binaire python
directement**, sinon les sous-process de vLLM V1 échouent silencieusement sur les outils
manquants.

## 3. Ce qui n'a PAS été fait (portée du test, à ne pas sur-interpréter)

- **Aucune migration de `~/envs/agentgym-rl`** : le venv de production garde vLLM 0.9.1,
  TRL 1.4.0, inchangés. Ce test est un venv jetable, entièrement sur `/tmp` (donc volatil —
  ne pas compter dessus après une purge, comme pour les checkpoints d'exp19).
- **Pas de test avec TRL** : TRL 1.4.0 avertit vouloir vLLM 0.12.0-0.18.0, pas 0.25.1. Aucune
  vérification faite aujourd'hui que `GRPOTrainer` / le mécanisme de sync in-process
  (`vllm_engine.py`) fonctionnerait avec cette version récente — c'est une étape distincte,
  plus risquée, à valider séparément avant toute migration du pipeline d'entraînement.
- **Pas de test du serveur HTTP OpenAI-compatible** (`start_vllm_server.sh` équivalent) —
  seul le mode `LLM()` offline/in-process a été testé. Le serveur n'est qu'une couche HTTP
  autour du même moteur, donc a de bonnes chances de fonctionner aussi, mais non vérifié.

## 4. Implication actionnable immédiate (sans toucher à l'entraînement)

Les scripts d'éval de ce projet (`eval_textcraft.py --backend vllm`, `eval_oracle.py`) parlent
au serveur vLLM par **HTTP** (`ChatGenerator` dans `llm_chat.py`) — ils ne savent pas quel
environnement Python fait tourner le serveur en face. Conséquence : **on peut dès maintenant
lancer un serveur vLLM 0.25.1 depuis ce venv `/tmp/envs/vllm-recent-test` pour SERVIR
Qwen3.5-4B, et pointer les scripts d'éval existants dessus (`--vllm-url`)** — sans toucher au
venv de training, et sans attendre une validation TRL. Ça remplacerait le fallback HF lent
(`--backend hf`, utilisé jusqu'ici pour Qwen3.5, cf. `exp15_qwen35_4b`) par le chemin rapide
KV-cache pour toute nouvelle éval de ce modèle.

**Migrer le pipeline d'ENTRAÎNEMENT** (donc bénéficier aussi de flash-attention et d'un moteur
vLLM in-process plus récent pour exp19.x) est une décision plus lourde — nécessite de valider
TRL 1.4.0 (ou une TRL plus récente) contre vLLM 0.25.1, dans un venv de test séparé, sans
perturber les runs en cours. Non fait aujourd'hui, à planifier si souhaité.

## 5. État des artefacts créés aujourd'hui (tous sur `/tmp`, volatils)

- `/tmp/envs/vllm-recent-test/` — venv vLLM 0.25.1 fonctionnel (9,4 Go). Réutilisable
  directement (`export PATH=...` puis lancer un serveur) sans réinstaller.
- `/tmp/models/Qwen3.5-4B/` — poids téléchargés (8,8 Go), prêts à être servis.
- `/tmp` : 194 Go libres après ces tests (sur 426 Go). `/home` : 2,6 Go libres (93 % plein) —
  **à surveiller**, plusieurs installations/téléchargements récents (checkpoints exp19,
  ce test) l'ont rempli progressivement.

## 6. Complément (même journée) — flash-attention validé, et la vraie portée du « bricolage »

### flash-attention fonctionne réellement

Pas de wheel précompilé pour torch 2.11 / CUDA 13 (trop récent) → compilation depuis les
sources (`pip install flash-attn --no-build-isolation`), ~50 min (72 fichiers `.cu`, kernels
générés pour 4 architectures GPU sm_80/90/100/120 par fichier). **`flash-attn 2.8.3.post1`
installé et testé fonctionnellement** (pas juste importé) :
`AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")` sur
Qwen2.5-3B → `model.config._attn_implementation == "flash_attention_2"` (pas de repli
silencieux vers `sdpa`) + génération correcte. C'est exactement le point que `CLAUDE.md`
documentait comme bloqué par le glibc 2.28 (`attn_implementation=sdpa` en repli) — débloqué.

### La découverte qui change le cadre : TRL 1.9.0 gère nativement notre cas d'usage

En lisant le code de TRL 1.9.0 installé à côté (pas seulement sa doc) :

- **Compatibilité déclarée** : `TRL currently supports vLLM versions from 0.17.0 to 0.25.1`
  (contre `0.12.0 to 0.18.0` en 1.4.0, la version installée en production). Notre paire
  actuelle TRL 1.4.0 / vLLM 0.9.1 est hors de la fenêtre déclarée depuis le début — pas
  seulement bridée par le glibc, un vrai décalage de version.
- **Le contrat `rollout_func` est identique** : `required_keys = {"prompt_ids",
  "completion_ids", "logprobs"}`, et `env_mask` explicitement reconnu (littéralement le
  même nom que notre code) avec le commentaire « Support custom env_mask from rollout_func
  (e.g., for environment feedback masking) ». `rollout.py`/`snis.py`/`data.py` (toute la
  boucle multi-tour TextCraft) n'auraient vraisemblablement pas besoin d'être réécrits.
- **Nouveauté absente de 1.4.0** : quand `rollout_func` ET `use_vllm=True` sont actifs
  ensemble, TRL 1.9.0 appelle **automatiquement** `self.vllm_generation.sync_weights()`
  avant chaque appel à notre `rollout_func` (`grpo_trainer.py:2151-2153`, commentaire
  « Keep vLLM weights in sync for custom rollouts that rely on vLLM utilities »). C'est
  exactement ce que `src/train/vllm_engine.py` fait à la main aujourd'hui (merge adaptateur
  → écriture ~5,8 Go sur disque → destruction/recréation du moteur, ~20-25 % du temps de
  step mesuré lors du profiling exp19). Cette fonctionnalité n'existait pas (ou pas sous
  cette forme) en TRL 1.4.0 — c'est elle qui a motivé la construction de `vllm_engine.py`
  à la main à l'époque.

### Ce qui reste nécessaire quoi qu'il arrive vs ce qui pourrait disparaître

Le « bricolage » a deux parties distinctes, à ne pas confondre :
1. **La boucle multi-tour TextCraft** (`rollout.collect_episodes` : get/craft/observe round
   par round) — nécessaire quelle que soit la version de TRL/vLLM ; TRL ne fait pas
   nativement d'interaction environnement multi-tour. Cette partie resterait quasi inchangée.
2. **Le mécanisme de sync du moteur vLLM** (`vllm_engine.py` : `init_engine`, `generate_round`,
   `sync_trl_to_vllm`, `VllmSyncCallback`) — c'est la partie qui pourrait être remplacée par
   `use_vllm=True` + l'appel à `trainer.vllm_generation` (au lieu de notre propre moteur géré
   à la main) depuis l'intérieur de `rollout_func`.

### Portée exacte de ce qui a été vérifié aujourd'hui (à ne pas sur-interpréter)

Tout ce qui précède sur TRL 1.9.0 est de la **lecture de code**, pas un test live de bout en
bout. N'a PAS été fait : construire un vrai `GRPOTrainer` avec `use_vllm=True` +
`rollout_func` + un vrai step d'entraînement sur ce venv de test ; vérifier que
`trainer.vllm_generation` expose une API de génération utilisable depuis notre boucle
multi-tour ; vérifier la cohabitation mémoire GPU de deux moteurs (modèle d'entraînement +
vLLM colocate) dans ce nouveau contexte. **Rien n'a été modifié dans `src/` ni dans
`~/envs/agentgym-rl` — uniquement le venv de test jetable sur `/tmp`.**

### Disque après ces tests

`/tmp` : 226 Go libres (venv vLLM 11 Go incluant flash-attn compilé, modèles téléchargés
17 Go — Qwen3.5-4B + Qwen3-4B). `/home` : 2,5 Go libres (93 % plein, stable).

### Proposition de plan de migration (à valider avant tout code)

1. **Étape isolée suivante** : dans ce même venv de test, construire un `GRPOTrainer` minimal
   (`use_vllm=True, vllm_mode="colocate"`, notre `rollout_func`) sur 1-2 items, vérifier que
   `sync_weights()` s'invoque et que `rollout_func` peut appeler le moteur vLLM de TRL.
2. Si concluant : adapter `vllm_engine.py` pour utiliser `trainer.vllm_generation` au lieu de
   notre moteur maison (garder `rollout.py`/`snis.py` quasi inchangés).
3. Valider à sec puis en smoke GPU (comme pour tout run), avant tout run long.
4. Seulement alors : migrer `~/envs/agentgym-rl` lui-même (ou créer un nouvel env dédié),
   avec un rollback possible (l'env actuel n'est jamais supprimé tant que la bascule n'est
   pas validée sur plusieurs runs).

Rien de tout cela n'a été exécuté aujourd'hui au-delà du point 0 (le venv isolé lui-même).
