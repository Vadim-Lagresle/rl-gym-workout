# exp25.1 — Étude d'optimisation GPU (throughput de génération)

> Décision 19/08 : avant de lancer la campagne finale (phases 1-2), vérifier
> qu'on tire le maximum du B200. **La conclusion de cette expé fixe les flags
> (`--vllm-gpu-util`, et éventuellement N) de toutes les expés suivantes.**

## Motivation (mesures du 19/08, exp25 en cours)

- GPU utilisé : **68 Go / 183 Go** — ~115 Go inutilisés pendant un run LoRA.
- `--vllm-gpu-util 0.17` est un réglage **fossile de l'époque full-FT** (validé
  quand Adam fp32 + gradients mangeaient le reste du GPU). En LoRA r8, le côté
  entraînement est minuscule.
- Calcul KV : un épisode à 20k tokens ≈ 0,7 Go de KV (Qwen2.5-3B, GQA). Le
  cache actuel (~25 Go utiles) héberge ~35 épisodes pleine longueur, or on en
  lance **64 en parallèle** → préemptions/queue vLLM probables en fin
  d'épisodes longs = débit de génération sous-optimal, alors que la génération
  représente ~80 % du temps de step.
- Enjeu dérivé : si le cache le permet, un bras **N=16 à budget de trajectoires
  constant** (16 rollouts × 4 prompts = 64 traj/step) densifierait le signal
  GRPO sur les items difficiles (item à 3 % de succès : P(groupe mixte) ~22 % →
  ~39 %) pour un surcoût temps à mesurer ici.

## Protocole (job `40b_exp25.1_gpu_bench.sh`, ~1 h GPU)

5 configs, **8 steps d'entraînement réel chacune** (LoRA r8 frais depuis la
base, dataset complet, 64 traj/step constant), mesure du s/it moyen sur les 4
derniers steps (les premiers incluent le warmup moteur) :

| Config | gpu-util | N (rollouts/prompt) | Prompts/step | Question |
|---|---|---|---|---|
| A_ref | 0.17 | 8 | 8 | référence actuelle |
| B_util04 | 0.40 | 8 | 8 | gain gratuit du cache élargi ? |
| C_util06 | 0.60 | 8 | 8 | rendement décroissant ? |
| D_n16u05 | 0.50 | 16 | 4 | surcoût réel de N=16 à traj constant |
| E_n16u06 | 0.60 | 16 | 4 | N=16 avec cache max |

Mesures relevées par config : s/it moyen, taille du KV cache annoncée par
vLLM au démarrage, nombre de préemptions dans le log.

Caveat : politique fraîche = épisodes longs (30 tours) = pire cas réaliste ;
c'est voulu (le débit en début de run est le régime le plus contraint).

## Règles de décision

1. Si B/C ≥ ~15 % plus rapides que A → `--vllm-gpu-util` retenu appliqué à
   **tous** les jobs suivants (41, 42, phase 2).
2. Si D/E coûtent < ~30 % de temps en plus que la meilleure config N=8 →
   le bras `exp36_n16` (N=16, 20 epochs) devient candidat prioritaire de la
   phase 2 (meilleure estimation d'avantage + groupes mixtes plus fréquents,
   quasi gratuit).
3. Sinon : on garde N=8 et le meilleur gpu-util, et le plafonnement se traite
   par l'échantillonnage (exp34) plutôt que par la profondeur de groupe.

## Résultats

Voir `results.md` (généré par le job).
