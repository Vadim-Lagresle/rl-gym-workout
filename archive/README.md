# archive/ — code retiré du chemin actif (2026-07-16)

Rien n'est supprimé : tout script remplacé ou périmé est déplacé ici par `git mv`
(historique préservé). **Ne pas importer depuis `src/`.** Chaque sous-dossier a
son README avec la raison de l'archivage et le remplaçant.

| Sous-dossier | Contenu |
|---|---|
| `eval/` | anciens scripts d'éval (boucle dupliquée, remplacés par `src/eval/eval_textcraft.py` + tronc commun) ; `api/` bornes SOTA Gemini/OpenAI et `single_turn/` pipeline exp16 (2026-09-23) |
| `analysis/` | analyses/figures one-shot à données codées en dur ; dashboards d'entraînement remplacés par les scripts de figures du rapport (2026-09-23) |
| `utils/` | doublons d'utilitaires fusionnés |
| `scripts/` | orchestrations one-shot terminées + scripts de merge LoRA originaux ; lanceurs de curriculum par stage et de baselines ; guetteurs de file (2026-09) |
| `train/` | variantes d'entraînement abandonnées : MAGELLAN, SNIS, mode plan (2026-09-23) |
| `setup/` | recettes d'environnement remplacées par `setup/setup_agentgym_rl_v2.sh` (2026-09-23) |
