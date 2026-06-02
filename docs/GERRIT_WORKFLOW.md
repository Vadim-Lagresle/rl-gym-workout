# Workflow Gerrit — `research/vadim-lagresle`

Ce document explique pourquoi on a fait un clone Gerrit séparé en plus du
repo GitLab principal, et comment pousser proprement chaque semaine.

## Pourquoi un clone séparé (et pas deux remotes sur `rl-gym-workout`)

Techniquement, on ne peut pas avoir un seul `.git` qui push à la fois sur
GitLab et sur Gerrit. Trois raisons :

1. **Structure incompatible.** Sur GitLab, le code est à la racine de
   `rl-gym-workout/`. Sur Gerrit, il doit être dans le sous-dossier
   `research/vadim-lagresle/` d'un monorepo d'équipe
   (`agentic/ai-agentic-commerce-incubation`).
2. **Historique différent.** GitLab reçoit des commits "de travail"
   (sales, fréquents, expériences brutes). Gerrit reçoit des snapshots
   hebdo propres avec un `Change-Id`.
3. **Hooks différents.** Gerrit impose un hook `commit-msg` qui ajoute un
   `Change-Id` à chaque commit ; ce hook n'a rien à faire sur GitLab.

Donc deux dossiers côte à côte sur la VM B200 Coder :

```
~/
├── rl-gym-workout/                          # repo GitLab — travail quotidien
└── ai-agentic-commerce-incubation/          # clone Gerrit
    └── research/vadim-lagresle/             # mon sous-dossier
```

Le "push sur les deux" se fait en deux temps :

- **Daily** : `git push origin main` dans `rl-gym-workout/`
- **Weekly** : rsync ciblé vers `research/vadim-lagresle/`, commit propre,
  push pour review sur Gerrit

## À quoi sert Gerrit (en deux phrases)

Système de code review centralisé Criteo : chaque commit devient une
**Change** identifiée par un `Change-Id`, on ne push jamais sur master
directement, on push sur `refs/for/master` pour créer une Change qui
sera mergée après review. Pour notre cas (recherche, encadrant qui veut
juste une trace), c'est principalement un journal de bord institutionnel
qui survit à la fin du stage.

## Setup déjà fait (juste pour info, ne pas refaire)

- Clé SSH ed25519 générée (`~/.ssh/id_ed25519`), publique enregistrée
  dans Gerrit settings sous le label "coder-b200".
- Clone shallow : `~/ai-agentic-commerce-incubation/` (~2.6 Mo, négligeable).
- Hook `commit-msg` installé dans `.git/hooks/` (ajoute le `Change-Id`).
- `git config user.email v.lagresle@criteo.com` et `user.name "Vadim Lagresle"`.
- Sous-dossier `research/vadim-lagresle/` créé avec un README initial.

## Workflow hebdo (à faire chaque semaine)

```bash
cd ~/ai-agentic-commerce-incubation
git pull --rebase origin master

# Sync ciblé depuis le repo de travail vers le sous-dossier Gerrit.
# Adapter la liste à ce qui mérite review cette semaine.
rsync -av --delete \
  --include 'src/' --include 'src/**' \
  --include 'runs/' --include 'runs/**' \
  --include 'docs/' --include 'docs/**' \
  --include 'WORKLOG.md' \
  --exclude '*' \
  ~/rl-gym-workout/ research/vadim-lagresle/

git add research/vadim-lagresle/
git status   # vérifier ce qui change
git commit -m "research(vadim-lagresle): <description courte>"
git push origin HEAD:refs/for/master
```

Le `HEAD:refs/for/master` est crucial : sans ça (= `git push origin master`),
Gerrit rejette le push parce qu'on n'a pas le droit de toucher master
directement. `refs/for/master` veut dire "je propose ce commit pour
être mergé dans master".

## Convention de commit

- Format : `type(scope): description` (aligné sur le repo, ex.
  `feat(...)`, `fix(...)`, `research(...)`)
- **Première ligne < 50 caractères** (Gerrit warn sinon)
- Le `Change-Id` est ajouté automatiquement par le hook, ne pas le toucher
- Une description plus longue peut suivre après une ligne vide

Exemple :

```
research(vadim-lagresle): add exp7 results

Full FT N=8 on B200, 100 steps, Pass@1 = XX/100.
Details in research/vadim-lagresle/runs/exp7_b200_fullft/.
```

## Ce qu'on push / ce qu'on NE push PAS

**On push** :
- `src/` (scripts d'entraînement, d'eval, d'analyse)
- `runs/` configs et résultats agrégés (`config.yaml`, métriques finales)
- `docs/` notes méthodologiques, `RESULTS.md`
- `WORKLOG.md` (extrait ou complet)

**On NE push PAS** :
- `saves/` (checkpoints LoRA et FSDP — trop lourds)
- `models/` (modèle base Qwen)
- `logs/` (logs bruts d'entraînement)
- `external/` (submodules AgentGym + verl)
- `scratch/eval_logs*/` (logs d'eval bruts, volumineux)

Même règle que le `.gitignore` GitLab.

## Suivre une Change après push

1. Ouvrir le lien retourné par `git push` (ex.
   https://review.crto.in/c/agentic/ai-agentic-commerce-incubation/+/1927997)
2. **Reviewers** (panneau droit) → "Add reviewer" → ajouter l'encadrant
3. Suivre le statut : NEW → REVIEWED → MERGED (par Gerrit après approbation)

## Modifier une Change après review

Si l'encadrant demande des changements (ou si on veut amender) :

```bash
# édite le code
git add research/vadim-lagresle/
git commit --amend --no-edit   # garde le même Change-Id, crée un patchset 2
git push origin HEAD:refs/for/master
```

`--amend` est crucial : sans ça, on crée une nouvelle Change au lieu de
mettre à jour celle existante. Le `Change-Id` dans le message de commit
permet à Gerrit de reconnaître que c'est la même Change.

## Abandonner une Change

Depuis l'UI Gerrit : bouton "Abandon" en haut de la page de la Change.
Réversible (bouton "Restore") tant que pas de raisons fortes.

## Cas particuliers

- **Plusieurs Changes dans un push** : si on fait 2-3 commits avant de
  push, Gerrit crée 2-3 Changes liées en série. Pour le workflow hebdo
  monolithique recommandé ici, un seul commit par push suffit.
- **Conflit lors du rebase** : `git pull --rebase` peut avoir des
  conflits si quelqu'un d'autre a touché `research/`. Très improbable
  (les autres bossent dans `projects/` ou `workbench/`).
- **Token GitLab expiré** : sans rapport avec Gerrit. Le push GitLab
  passe par Coder qui injecte un token HTTPS, pas par cette clé SSH.
