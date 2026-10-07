# Session 2026-08-11 — bilan de la lignée exp23 (69/100) et grille LoRA exp24

## 1. exp23.4 : 69/100, nouveau record du projet

Morte dans la nuit du 10-11/08 (purge pod, device home rbd1→rbd3) au step ~3170/6600
(epoch ~72/150). **Best 69/100 @ step 3055 (epoch ~69.4)**, sauvé complet sur le home
(poids 6.2 Go + optimizer.pt 8.1 Go) — le seuil abaissé à 0.55 a rendu la purge
indolore. Dernières évals 56-69 : encore ascendant, sous un bruit de ±5 pts.

Ménage post-mortem : poids du pic 64 (6.2 Go, pré-validé le 10/08 mais jamais exécuté —
shells morts) + wandb local → disque de 99 % à 81 % (6.7 Go libres).

## 2. Récap de la lignée warm-start (recette verl-sur-TRL, full-FT zero-shot)

Chaque maillon recharge les poids du précédent (et l'optimiseur quand il existait),
l'ancre KL restant TOUJOURS Qwen nu :

| Maillon | Départ | Adam | Epochs courues | Best | Fin |
|---|---|---|---|---|---|
| exp23 + 23.1 (scratch) | Qwen nu | — | ~96 | 58 @ ep 87.6 | purge /tmp |
| exp23.2 | poids du 58 | perdu (incident) | ~29 | **64** @ ep 25.6 | purge /tmp |
| exp23.3 | poids+Adam du 64 | chargé | 50 (complet) | aucun (max 63) | fin propre |
| exp23.4 | final d'exp23.3 | neuf | ~72 | **69** @ ep ~69.4 | purge pod |

**Total : ~247 epochs cumulées depuis scratch → 69/100** (papier : 75).
Après le premier rechargement : ~151 epochs pour +11 pts (58 → 69).

Leçon consolidée sur le « creux post-restart » : chaque redémarrage repart ~5-10 pts
sous le best parce que le best est un MAX sur des évals bruitées (pic chanceux), pas le
vrai niveau — exp23.3 l'a prouvé en redémarrant bas AVEC les moments Adam chargés. Le
vrai niveau, lui, monte de façon continue : ~58 → ~60 → ~63-65 (pics 68-69).

## 3. exp24 — grille d'hyperparamètres LoRA (« LoRA Without Regret »)

Décision de Vadim : basculer sur une recherche LoRA calquée sur les findings du blog
[Thinking Machines](https://thinkingmachines.ai/blog/lora/) — détail dans
`runs/12_lora_grid/exp24_grid/README.md`. L'essentiel :

- **RL = rang 1 suffit en capacité** (1 bit/épisode) → la grille r=16/32/64 teste la
  dynamique, pas la capacité.
- **LR optimal LoRA ≈ 10× full-FT** → notre 1e-6 full-FT prédit 1e-5 en LoRA.
- **α=32 FIXE pour tous les rangs** (le préfacteur α/r rend alors le LR indépendant du
  rang) — remplace notre convention α=2r sur cette grille.
- Notre code cible déjà toutes les couches (q,k,v,o + MLP) — conforme au blog.
- Caveat blog : LoRA souffre des gros batches — notre buffer 256 est gardé pour la
  comparabilité avec exp23, la grille le mesurera de fait.

9 runs de **100 epochs** (~40 h chacun — décision Vadim : budget long d'emblée, arrêt
manuel des runs qui plafonnent ou s'effondrent), recette exp23 par ailleurs identique.
Combos : A = (1e-5, 0.001) la prédiction du blog ; B = (3e-6, 0.001) conservateur ;
C = (1e-5, 0.01) ancre KL forte. Ordre : A sur les 3 rangs d'abord.
Barres de lecture : full-FT ≈ 30-32/100 à epoch 15, 58 à epoch ~88.

## 4. Nouveau : file d'expériences séquentielle (`scripts/run_queue.sh`)

Réponse à la demande d'enchaînement automatique : un runner consomme les jobs
`runs/queue/*.sh` dans l'ordre lexical, un à la fois, en reconstruisant la stack
entre deux jobs si besoin (/tmp env v2, Qwen, serveur TextCraft). Jobs terminés →
`runs/queue/done/`. Un échec ne bloque pas la file. Validé à sec (3 jobs factices,
dont un en échec). Limite : une purge pod tue le runner — le relancer reprend à la
première ligne non consommée.

Lancement : `setsid nohup bash scripts/run_queue.sh > logs/queue.log 2>&1 &`
