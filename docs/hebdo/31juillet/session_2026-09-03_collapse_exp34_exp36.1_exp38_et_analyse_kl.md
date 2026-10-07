# Session 2026-09-03 — collapse exp34 (MAGELLAN), kill, exp36.1 en cours, exp38 (option A) en file, analyse KL transverse

## Décisions (Vadim)
- **exp34 tuée à 08:47** (0-2 % depuis step 940). Le runner a enchaîné automatiquement sur **job 48 = exp36.1** (ancre /12, G16 sans curriculum) = option B(i) à coût nul.
- **Option A validée** : mécanique du papier (collecte 256, 4 pas clippés) sur la recette LoRA + ancre mobile → **job 49 = exp38_buffer256_n16**, après exp36.1.
- **exp37 (few-shot N=16) parquée** dans `queue/skipped/` : plus le temps de l'exploiter avant la soutenance.
- Option B(ii) « π_ref fixe avec β/10 » : déjà couru (exp24 β0.001 fixe : collapse ; exp31.1 β0.0001 fixe : divergence) → pas de run.
- Gel scientifique : plus d'heure humaine sur du code ; le GPU tourne seul via la file.

## exp34 — verdict
Pic 45 (step 611, ep ~6.5), collapse ep 7-8 (entropie 0.66 → 0.19, KL 133 → 10¹²⁺), 0-2 % dès step 940. Sampler final p(d1)=0.79, p(d2)=0.18, p(d3)=0.03, p(d4)=0.0006. Prédiction « désinvestit d4 » confirmée ; mais le LP mesure du changement, un collapse est un changement énorme → concentration sur d1 → moins de diversité → collapse accéléré (rétroaction positive LP × instabilité). Lecture pour §4.3 : l'autocurriculum suppose un apprenant stable, il n'est pas une source d'entropie.

## exp38 — configuration exacte (job 49)
Recette exp36 (= lignée exp25/32, G16, sans curriculum) à l'identique : r8/α32, LR 3e-6 constant, β 0.01, ancre mobile /4 ep, entropy-coef 0.001, N=16, grad_accum 64 (64 traj/pas), 512 tok/tour, 30 tours fixes, gpu-util 0.5, éval /47 pas, 80 ep, save-best-optimizer.
**Seul delta : `--steps-per-generation 256`.** Conséquences :
- TRL génère 256 trajectoires d'un coup (16 prompts × 16), calcule les old_logprobs sous la politique de collecte, puis fait 4 pas d'optimisation de 64 traj : pas 1 on-policy (ratio ≡ 1), pas 2-4 hors-politique avec clipping ε=0.2 (les `clip_ratio/*` deviennent non nuls).
- Chaque trajectoire est utilisée UNE fois (`num_iterations=1`) : même nombre de trajectoires par epoch (5984) et de pas par epoch (~93) qu'exp36 → compute égal. Les évals tombent aux mêmes pas.
- Avantages calculés une fois par groupe à la collecte (inchangé). Ancre mobile inchangée : un ré-ancrage peut tomber au milieu d'une collecte ; la fusion est l'identité sur la politique (adapter fusionné, adapter neuf à zéro) donc les old_logprobs restent valides, seule l'ancre KL bouge.
- Pré-vol GPU (~5 min) : smoke LoRA + buffer 8/grad_accum 4 (2 pas par collecte) + ré-ancrage toutes les 0.4 ep (tombe au milieu d'une collecte) ; le job vérifie qu'un `RÉ-ANCRAGE #1` a eu lieu, sinon annule.
- Alternative écartée : ancre /12 comme exp36.1 → deux deltas vs exp36, comparaison confondue. Vadim peut changer les valeurs dans `runs/queue/49_exp38_buffer256_n16.sh` avant son démarrage (≥ 12 h).

## Analyse KL transverse
Table et 8 lectures dans `docs/RESULTS.md` (section « Analyse transverse KL »). Points saillants :
- KL saine = 10⁻³ (dérive d'un pas à LR 3e-6), tenue 80-110 ep par tous les runs à ancre mobile, quel que soit β.
- Full-FT à ancre fixe : dérive linéaire +0.001/ep (0.026 à ep 30), entropie constante. LoRA à ancre fixe : dérive géométrique ×5-10 par 2 ep dès l'ep 4 ; β retarde d'une fenêtre, le rang accélère ; entropie encore 1.3-1.4 quand la KL franchit 0.05 → emballement KL, pas collapse d'entropie.
- Point de non-retour : KL médiane 0.05-0.1. β·KL = 10⁻⁵ en régime sain (0,3 % du terme de politique 2.9e-3) ; le frein n'engage qu'à KL ≈ 0.3 (β0.01), 3 (β0.001), 30 (β0.0001) : après le point de non-retour pour tous les β testés.
- **Correction pour le rapport** : sous ancre mobile, β0.01 et β0.001 sont indiscernables à epoch égale (ep 49-58 : 45-52 vs 46-53) ; β0.001 garde plus d'entropie (1.20 vs 1.05). Le 65-73 d'exp25 est un effet de durée. β0.01 = continuité de lignée, pas supériorité mesurée. PLAN §4.2 corrigé.
- Collapses G16 (exp36/34) : entropie d'abord à KL stable 0.003 → mécanisme distinct, hors de portée de l'ancre.

## Fichiers touchés
`runs/queue/49_exp38_buffer256_n16.sh` (nouveau), `runs/queue/skipped/49_exp37_fewshot_n16.sh` (déplacé), `runs/INDEX.md` (exp34 verdict, exp36.1, exp38, exp37 parquée), `docs/RESULTS.md` (interprétation exp34 + analyse KL), `docs/rapport/PLAN_RAPPORT.md` (§4.2 : correction β + point 5 justifications quantitatives).

## Après-midi : purge /tmp, gel définitif des expériences (décision Vadim, 03/09 ~16:30)
- Purge du pod vers 15:00-16:00 : `/tmp/envs`, `/tmp/models`, `/tmp/trl_grpo_runs` effacés ; runner et exp36.1 morts (rien ne tourne sur le GPU).
- **exp36.1 (ancre /12) : verdict acquis avant la purge** — KL 10⁶-10⁹ dès l'époque 4,8, entropie 0,17-0,26, pass@1 44 → 34 à l'époque 6. Sans ré-ancrage avant l'époque 12 elle se comporte comme une référence fixe et diverge PLUS TÔT qu'exp36 (/4). Espacer l'ancre aggrave : option B(i) close.
- **Gel définitif** : jobs 48 (exp36.1) et 49 (exp38, option A) déplacés dans `runs/queue/skipped/`. exp38 (collecte 256 sur recette LoRA + ancre) n'a jamais démarré : reste un travail futur.
- Rédaction : figures 4.2 et 4.3 produites depuis les logs (`docs/rapport/figures/`), python système + matplotlib (`pip install --user`) car l'env v2 est purgé.
- À faire dans INDEX : lignes exp35 / exp35.1 manquantes (best 78 @ ep 55 ; reprise ancre 16 → 80 @ ep ~69, plateau 75) ; ligne exp36.1 (divergence ep 5) ; exp38 « jamais lancée, gel ».

## Soir : gel levé pour UN run, exp39 (décision Vadim 03/09 ~23:00)
- Facteur confondu relevé par Vadim : exp25 (8 tâches × 8 traj) vs exp36 (4 tâches × 16 traj) changent
  DEUX choses à 64 traj/pas. Le collapse G16 peut venir du nombre de tâches par pas, pas de G.
- **exp39_g16_8tasks** (job 50) : recette exp36 à l'identique, seul delta `--gradient-accumulation-steps
  128` → 8 tâches × 16 = 128 traj/pas, 47 pas/époque, éval /47 (une par époque), 80 ép., arrêt manuel.
  Verdict attendu en 10-15 ép. : casse → c'est G ; tient → c'était les tâches/pas ; casse plus tard → les deux.
- Runner relancé (`setsid nohup bash scripts/run_queue.sh`) : il reconstruit l'env v2 (purgé), retélécharge
  Qwen, relance le serveur TextCraft, puis démarre le job 50. Jobs 48/49 restent parqués.
- Rapport : réserve à insérer en 4.3.1 et 4.4 (le contrôle isole l'effet du curriculum, pas celui de G).

## 04/09 matin : 2e purge en 20 h, exp39 tuée au pas 26, relancée
- Diagnostic : aucune erreur dans le log d'exp39 (dernier pas 11:06, 26/3680, reward et KL normaux).
  À 11:08 le conteneur Coder a été recréé (`/tmp/code-server.log`, `coder-agent-init.log` datés 11:08,
  `/tmp/envs` et `/tmp/models` effacés, tous les processus tués : runner, train, serveur TextCraft).
  Ce n'est pas un crash du run mais une recréation du pod, la 2e en 20 h (précédente : 03/09 ~15-16 h).
- Relance 12:03 : runner relancé (`setsid nohup`), reconstruction de l'env v2 puis job 50 depuis zéro
  (44 min perdues). Le job était resté dans `runs/queue/` (le runner n'avait pas pu le déplacer).
- Vitesse mesurée avant la purge : ~113 s/pas à 128 traj/pas → ~87 min/époque → verdict (10-15 ép.)
  en ~15-22 h ; 80 époques prendraient ~5 jours (arrêt manuel prévu bien avant).
- Rien de sauvé sur le home avant le pas 47 (1re éval) : pas de reprise possible, redémarrage propre.

## 07/09 08:30 — verdict exp39 : ça tient (époque 22)
- Depuis la reprise (ckpt-94, 06/09 13:24) : KL médiane 0,001 constante sur 20 époques, entropie 1,05 → 0,93,
  pass@1 38 → 49 (pas 893 et 940), aucune divergence. exp36 (4 tâches × 16) était morte à l'ép. 10, exp34 à l'ép. 7-8.
- **Conclusion** : le facteur qui tuait exp36 est le nombre de tâches par pas (4 au lieu de 8), pas G=16.
  À 8 tâches × 16, la recette sans curriculum est stable comme à 8 × 8 (exp25).
- Nuance : exp39 est PLUS LENTE que les curriculums à époque égale (49 à l'ép. 22 vs ~60 pour Horizon/Budget) et
  qu'exp36 avant son collapse (54 à l'ép. 13). Le curriculum reste un accélérateur ; la protection contre le
  collapse peut venir soit du curriculum, soit de la diversité de tâches par pas.
- À répercuter : rapport 4.2.5 (corollaire G=16), 4.3.1 (réserve), 4.3.3 (corollaire), 4.4 (« en cours »), §5 ;
  slides « Limites » et annexe « collecte 64 vs 256 ». Figure : `docs/rapport/figures/fig_exp39_control.pdf`.

### Correction 07/09 09:30 (Vadim) : exp39 a un second facteur confondu
- L'ancre mobile est définie en époques : exp39 (46 pas/ép.) ré-ancre tous les 184 pas, exp36 (93,5 pas/ép.) tous les 372.
  Même volume de trajectoires par cycle, mais deux fois moins de pas de gradient entre deux ancres → région de
  confiance plus serrée par pas ; visible sur la figure (la KL d'exp39 retombe à 10⁻³ deux fois plus souvent).
- Donc exp39 vs exp36 = 2 deltas (tâches/pas ET fréquence d'ancre en pas). Verdict à reformuler : « à huit tâches
  par pas ET ancre tous les 184 pas, pas de collapse » ; l'attribution au seul nombre de tâches n'est pas établie.
- Run qui trancherait : 8 tâches × 16, `--moving-anchor-every-epochs 8` (372 pas entre ancres, comme exp36).
- Annexe F.6 et INDEX corrigés en conséquence.

### 08/09 19:50 — exp40 mis en file (job 52) : contrôle propre du confondeur
- Décision Vadim : après exp39, relancer la même config mais avec l'ancre au rythme EN PAS des
  curriculums et d'exp36 → `--moving-anchor-every-epochs 8` (8 × 46,75 = 374 pas ; exp36 : 4 × 93,5 = 374).
  Tout le reste identique à exp39 (G=16, 128 traj/pas = 8 tâches, 30 tours, 512 tokens, 80 ép., ckpt home).
- Lecture attendue : exp40 stable → la stabilité d'exp39 ne venait pas de l'ancre plus fréquente, le
  nombre de tâches par pas suffit ; exp40 collapse → c'était la fréquence d'ancre (ou les deux).
- exp39 à l'époque 68 (ré-ancrage #17, step 3128, 18:59) ; fin des 80 ép. estimée 09/09 matin.
- ⚠ Anomalie : DEUX runners `run_queue.sh` vivants (13778 = propriétaire d'exp39 ; 40813 bloqué en
  `do_wait` sur le serveur TextCraft qu'il a lancé le 06/09 13:21). Si le serveur meurt, 40813 se
  réveille et lancerait le premier job de la file en double → à tuer (proposé à Vadim).

### 10/09 08:30 — exp40 : COLLAPSE, arrêt (Vadim), verdict
- Parité vérifiée dans les jobs et les logs : LoRA r8/α32, LR 3e-6 (log), β 0.01, entropy-coef 0.001,
  G=16, 30 tours, 512 tokens, éval/47, ancre 374 pas (log : ré-ancrages aux pas 368 et 736 ; exp32 : 372, 744).
  Seuls écarts avec les curriculums : pas de curriculum, 128 traj/pas (8 tâches) au lieu de 64 (4 tâches), ckpt home.
- Déroulé : pass@1 49 au pas 705 puis 29/32/27/18 ; KL 0.04→1.14 entre ép. 4 et 7 AVANT la 1re ancre, entropie
  1.35→0.45 ; ancre ép. 8 → KL 1e-3 mais entropie 0.33 ; ép. 15 divergence (KL 2e3 → 1e19), longueurs 9 600 tokens.
- Conclusion : à rythme d'ancre égal en pas, sans curriculum ça casse (exp36 avec 4 tâches, exp40 avec 8),
  avec curriculum ça tient (exp32/33/35). Le facteur protecteur est le curriculum, pas le nombre de tâches par pas.
  La stabilité d'exp39 venait de son ancre 2× plus fréquente (187 pas).
- Réserve : exp40 consomme 48k traj par cycle d'ancre (128/pas) contre 24k pour exp36 et les curriculums.
- Runner terminé (file vide), GPU libre. Rien en file.

### 10/09 08:36 — exp41 lancé (job 53) : G=8, 8 tâches × 8, ancre 374 pas
- Répond à la réserve d'exp40 (128 traj/pas = 2× les curriculums par pas et par cycle d'ancre).
- exp41 = exp25 (8×8 = 64 traj/pas) avec ancre 8 ép. = 374 pas ; 80 ép., ckpt home, gpu-util 0.5. Aucun run
  antérieur ne couvrait cette case (vérifié dans INDEX et runs/queue).
- Lecture : tient → à 64 traj/pas et ancre 374, 8 tâches suffisent, le collapse d'exp40 vient des 128 traj/pas ou de G=16 ;
  casse → exp25 ne tenait que par son ancre à 187 pas, sans curriculum seule l'ancre fréquente protège.
- Runner relancé (`setsid nohup bash scripts/run_queue.sh`), un seul runner vivant.

### 11/09 — exp41 tient (ép. 48, pass@1 62, 6 ré-ancrages) ; exp42 MAGELLAN + Horizon mis en file (job 54)
- exp41 (G=8, 8×8 = 64 traj/pas, ancre 374 pas) stable → le collapse d'exp40 vient des 128 traj/pas ou de G=16,
  pas de la fréquence d'ancre. Sans curriculum, le collapse est propre à G=16.
- exp42 = exp32 + `--goal-sampler magellan` (compatible dans le code : MAGELLAN n'est exclusif qu'avec les
  calendriers de profondeur). Démarre automatiquement à la fin d'exp41.

### 11/09 08:38 — recréation du pod (~06:45) : exp41 tué à l'ép. 47, reprise exacte (job 53b)
- Symptômes : runner et python disparus, log coupé en pleine ligne sans erreur, /tmp/envs et /tmp/models absents,
  uptime 6 j (nœud inchangé, pod recréé). Aucune perte : ckpt-2162 (ép. 47) + optimizer sur le HOME, best 62 @ 2209,
  cycles 1-6 sur le home.
- Reprise = pattern du job 40 : base Qwen ⊕ cycles ≤ 2162 (1-5) via merge_anchor_chain.py `--adapter-step 2162`,
  `--model-path BASE --resume-from-checkpoint ckpt-2162 --moving-anchor-initial-cycle 5 --best-init-score 0.62`.
  Cycle 6 (step 2208 > 2162) déplacé dans `_anchors/post_ckpt2162_run1/` avec copie de chain.jsonl ; chain.jsonl tronqué à 5 lignes.
- Job 53 d'origine déplacé en done/ (sinon relance de zéro). Runner relancé (PID 5636) : reconstruit env v2 + Qwen puis 53b puis 54 (exp42).
