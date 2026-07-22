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
