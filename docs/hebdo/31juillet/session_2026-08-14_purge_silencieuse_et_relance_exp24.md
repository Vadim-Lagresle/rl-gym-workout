# Session 2026-08-14 — purge pod silencieuse (2 j de GPU idle) et relance exp24

## Le constat

En voulant tuer le job 04 (`exp24_r64_lr3e-6_b0.001`) jugé « lent », découverte que
**tout était mort depuis le 12/08 ~17h16** (purge pod) : `/tmp` vidé, runner de file,
serveur TextCraft et train tués. GPU à 0 MiB, log du job figé depuis 2 jours.

Deux erreurs de lecture corrigées :

1. **Le job 04 n'a jamais été lent** : ~73 steps/h jusqu'à sa mort (epoch 10.6/15),
   vitesse nominale d'une politique scratch (réf. exp23 ~64 st/h).
2. **Le check de vie `pgrep -f train_grpo` s'auto-matchait** sur le wrapper du shell
   sandbox → faux « train vivant » pendant 2 jours. Leçon : la vérité terrain c'est
   `nvidia-smi` + mtime du log, pas un pgrep.

## Lecture du job 04 (acquise, non rejoué)

Évals /100 tous les 47 steps : 15, 17, 17, 19, **25**, 23, 23, 21, 19 (dernière @ step 423,
ep 9.6 ; mort step ~468). Plateau à peine au-dessus de la baseline 18, loin de la barre
full-FT (~30 à ep 15). LoRA r64 / LR 3e-6 / β 0.001 n'apprend pas vraiment.

## Relance (validée « vazy »)

- Job 04 classé dans `runs/queue/done/` (pas de rejeu : 10.6 ep suffisent à conclure).
- **File réordonnée** (renumérotation 21-28) : le job **β 0.01 r64 passe en tête**
  (ablation β vs job 04 à chaleur α·LR égale 9.6e-5 = la lecture la plus informative),
  puis les rangs 16/32 des deux β, l'arm 1e-6 en dernier.
- Runner relancé : reconstruction complète de `/tmp` (env v2 + Qwen) puis démarrage
  du job 21 `exp24_r64_lr3e-6_b0.01`.

Nouvel ordre de file : 21 r64/3e-6/β0.01 → 22-23 r16,r32/3e-6/β0.001 →
24-25 r16,r32/3e-6/β0.01 → 26-28 r64,r16,r32/1e-6/β0.001.
