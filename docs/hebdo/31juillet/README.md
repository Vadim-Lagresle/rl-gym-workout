# Semaine du 31 juillet 2026 — notes de session

Dossier pour les comptes-rendus et décisions de la semaine à venir.

## Fichiers de cette semaine

- [`session_2026-07-27_exp21_pure_reasoning.md`](session_2026-07-27_exp21_pure_reasoning.md) — mode single-turn RL codé + lancement exp21.
- [`session_2026-07-27_paliers_lr_et_reward_train.md`](session_2026-07-27_paliers_lr_et_reward_train.md) — bilan paliers LR exp20 (stabilise + meilleures perfs) vs dynamiques train cassées ; piste LR adaptatif au reward train.

## Repères mis à jour (27/07)

- **exp20_staged_lr_v2** : terminé, **58/100 @ step 800**, pas de collapse — paliers LR/beta ÷3 / 3 epochs validés côté stabilité et pic test, mais le reward train suggère qu'on coupe des dynamiques d'apprentissage ; piste : LR piloté par la dérivée d'une moyenne roulante du reward train (~¼ epoch).
- **Run en cours** : `exp21_pure_reasoning` (single-turn, wandb `l4rwppam`) — voir note ci-dessus.

