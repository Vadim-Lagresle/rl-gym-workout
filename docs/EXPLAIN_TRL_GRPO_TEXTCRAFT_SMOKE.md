# Explication détaillée de `scratch/07_trl_grpo_textcraft_smoke.py`

Doc autonome qui consolide les deux explications fournies en chat (version longue pédagogique + cheatsheet courte). À relire quand tu reprends le projet ou pour préparer une discussion technique.

---

## Pourquoi "smoke" dans le nom du fichier ?

C'est un terme de génie logiciel qui vient de l'électronique : un **smoke test** est un test minimal qu'on fait juste pour voir si la chose ne prend pas feu — littéralement, à l'origine, on branchait un circuit nouvellement assemblé et on regardait s'il y avait de la fumée. En logiciel, ça désigne un test rapide et superficiel dont le seul but est de vérifier que la pile de base démarre, ne plante pas, et produit des sorties qui ressemblent à ce qu'on attend. Ce n'est pas un benchmark, ce n'est pas un run de production, c'est un "ça démarre, ça ne crash pas, on peut continuer".

Notre fichier est nommé `_smoke` parce qu'il a été conçu au départ pour ça : valider qu'on peut faire tourner un step ou deux de GRPO avec notre rollout custom, sans crash de mémoire et avec des métriques qui ont une tête raisonnable. Le nom est resté même après que le fichier soit devenu utilisable pour des runs de stabilité plus longs (10 ou 20 steps), pour garder en mémoire qu'il n'est pas encore le pipeline final de production.

---

## Vue d'ensemble du fichier

Le fichier sépare proprement deux responsabilités. Tout ce qui est numériquement délicat — la loss GRPO, l'optimizer, la mixed precision, LoRA, le ratio d'importance sampling — est entièrement délégué à TRL. Tout ce qui est spécifique à notre projet — la construction des prompts TextCraft, le rollout multi-tour avec env-in-the-loop, le reward shaping — est codé explicitement dans ce fichier. Le pont entre les deux mondes est la fonction `textcraft_rollout_func`, qui implémente le contrat technique exigé par TRL tout en encodant notre logique métier à l'intérieur. C'est pour ça que le fichier est court (moins de trois cents lignes) alors qu'il pilote un pipeline RL multi-tour complet : tout ce qui est lourd vit ailleurs.

---

## Version longue : fonction par fonction

### Le décor (lignes 1 à 25)

En haut, les imports et les constantes globales. On importe les briques classiques de l'écosystème Hugging Face — le tokenizer pour découper le texte en identifiants de tokens, `Dataset` pour empaqueter notre dataset en quelque chose que TRL sait avaler — et les briques spécifiques au projet : `LoraConfig` qui décrit le LoRA qu'on greffe sur le modèle, `GRPOConfig` et `GRPOTrainer` qui sont les deux objets centraux de TRL, et `TextCraftEnvClient` qui est le client HTTP qu'on a déjà rencontré dans les scripts d'eval.

Les constantes méritent un mot. `MODEL_PATH` pointe vers le checkpoint local de Qwen pour ne pas re-télécharger à chaque run. `TRAIN_PATH` pointe vers le dataset des items d'entraînement TextCraft. `ENV_SERVER_URL` est l'adresse du serveur HTTP qu'il faut avoir lancé à côté avant. `MAX_SIM_ROUNDS = 20` est le plafond de tours qu'on autorise par épisode pendant le rollout. `ITEM_TAG_RE` est une regex qui sert à un mécanisme d'aiguillage : on encode l'identifiant numérique de l'item dans le prompt sous la forme `<ITEM_IDX:42>` pour pouvoir, plus tard, retrouver à quel item on est en train de jouer quand le rollout est appelé.

### Les utilitaires de pré-traitement

`item_id_to_idx` est une petite fonction qui prend un identifiant textuel comme `textcraft_train_42` et qui en extrait l'entier `42`. Le serveur TextCraft attend un index numérique pour `reset(idx)`, alors que le dataset contient les identifiants sous forme texte. C'est juste un convertisseur.

`check_server` fait un appel HTTP `POST /create` au serveur TextCraft et lève une exception s'il ne répond pas correctement. C'est le garde-fou en début de run : si tu as oublié de lancer le serveur ou si l'adresse est mauvaise, tu plantes en deux secondes plutôt qu'au milieu d'un step de training, dix minutes plus tard. C'est ce qui nous a sauvés sur la race condition rencontrée hier.

`build_prompt_rows` est plus subtil. Il construit la liste de prompts qu'on va donner à TRL comme dataset d'entraînement. Pour chaque item du fichier `textcraft_train.json`, il fabrique une conversation de quatre messages. Le premier est le system prompt générique de Qwen ("You are a helpful assistant"). Le deuxième est le manuel TextCraft dans son entièreté — l'explication des règles du jeu, des actions disponibles, du format attendu, qu'on récupère depuis `probe.conversation_start[0]["value"]`. Le troisième est l'acquittement scripté du modèle ("OK, I'll follow your instructions and try my best to solve the task"), qu'on met en dur pour que la vraie première réponse de Qwen pendant le rollout soit déjà dans un état conversationnel cohérent. Le quatrième est, et c'est là que se joue l'astuce, un message contenant uniquement la chaîne `<ITEM_IDX:42>`. Ce n'est pas une vraie observation TextCraft, c'est un marqueur volontaire qu'on va remplacer par la vraie observation initiale plus tard, dans le `rollout_func`.

Pourquoi cette astuce ? Parce que TRL ne nous laisse pas facilement transporter de la métadonnée à côté du prompt. Le contrat est : tu mets ce que tu veux dans `prompt`, et c'est tout. Donc on encode l'index dans le contenu du dernier message, on le retrouvera côté rollout, et on le remplacera à la volée par la vraie observation après avoir fait `env.reset(idx)`. C'est un hack, mais il est lisible et confiné à deux endroits du code.

### Les utilitaires de post-traitement du texte généré

`completion_to_text` est défensif. TRL passe parfois la completion comme une chaîne, parfois comme une liste de messages chat (selon que tu lui demandes du texte plat ou structuré). Cette fonction prend l'un ou l'autre et retourne toujours une chaîne pour le code qui suit.

`count_actions` et `extract_actions` font la même chose en deux étages. `count_actions` compte combien de lignes commencent par `Action:` dans la réponse de l'agent. `extract_actions` retourne ces lignes nettoyées — espaces collés, première ligne uniquement, version normalisée. La nuance vient du serveur TextCraft : il s'attend à recevoir une seule action à la fois. Si Qwen écrit deux `Action: ...` dans la même réponse, le serveur n'en exécute qu'une et signale une erreur de format pour le reste. Cette information est utilisée à deux endroits différents : dans le rollout pour décider quelle action envoyer, et dans la reward pour pénaliser le multi-action.

`first_action_or_empty` est un wrapper qui retourne la première action extraite, ou la chaîne vide si Qwen n'a écrit aucune ligne `Action:`. C'est ce qu'on enverra au serveur si le modèle dérive.

### Le cœur : `textcraft_rollout_func`

C'est la fonction la plus dense du fichier. Elle est appelée par TRL une fois par step d'entraînement, avec en argument la liste des prompts du batch courant (chaque prompt étant la conversation à quatre messages qu'on a construite dans `build_prompt_rows`) et une référence au trainer lui-même (qui nous donne accès au tokenizer et à la machinerie de génération).

La toute première chose que fait la fonction, c'est l'initialisation des structures de bookkeeping. `n` est la taille du batch. `states` est une liste de listes : pour chaque sample du batch, on garde une copie mutable de la conversation, parce qu'on va y ajouter des messages au fur et à mesure des tours. `done`, `final_rewards` et `invalid_counts` sont les compteurs par sample qu'on remplit au fur et à mesure. `env_clients` accueillera un client HTTP TextCraft par sample.

Ensuite vient le bootstrap des environnements (lignes 109-120). On itère sur les `n` samples. Pour chacun, on lit le contenu du dernier message du prompt — qui est notre marqueur secret `<ITEM_IDX:N>`. On vérifie qu'il a bien le bon format avec la regex, on extrait l'entier, et on instancie un nouveau `TextCraftEnvClient` qu'on `reset` à cet item précis. On observe l'état initial du jeu (qui ressemble à "Crafting commands: craft <recipe> using <ingredients>; ... Goal: craft <item>"), on remplace le marqueur par cette vraie observation dans le state, et on stocke le client dans la liste. Quand cette boucle termine, chaque sample a son propre client TextCraft connecté au bon item, et chaque conversation a son observation initiale en place.

Les trois lignes 123-125 préparent les listes de sortie. Le contrat de `rollout_func` est de retourner trois choses centrales — `prompt_ids`, `completion_ids`, `logprobs` — chacune sous forme d'une liste de listes (une par sample). On les initialise vides, on les remplira au fur et à mesure.

Vient ensuite la grande boucle de tours, lignes 127-166. C'est là que la dynamique multi-tour vit. Pour chaque tour, on calcule d'abord la liste des samples encore actifs (ceux qui ne sont pas marqués `done`). Si plus aucun n'est actif, on sort de la boucle prématurément. Si on a au moins un sample actif, on entre dans la mécanique du tour.

D'abord on prépare les prompts d'entrée du modèle. Pour chaque sample actif, on rend la conversation courante (state complet : system + manuel + ack + observation initiale + tous les tours d'assistant et d'environnement déjà accumulés) sous forme de chaîne, en utilisant `apply_chat_template` du tokenizer Qwen. Ce template gère l'ajout des balises `<|im_start|>user` etc. qui sont spécifiques au format Qwen. On encode ensuite cette chaîne en identifiants de tokens. La subtilité importante : on passe `add_generation_prompt=True`, qui rajoute la balise `<|im_start|>assistant` à la fin pour signaler au modèle qu'il doit maintenant générer une réponse. Si on oubliait ça, le modèle continuerait à se prendre pour le user.

Au tour zéro et seulement à ce moment-là, on stocke le prompt encodé dans `prompt_ids_out[i]`. Pourquoi seulement le premier tour ? Parce que TRL veut une seule liste d'identifiants de prompt par sample, qui correspond à ce que le modèle voit "au départ" de l'épisode. Pour les tours suivants, le prompt s'allonge (il inclut tout l'historique multi-tour), mais ces extensions ne sont pas du prompt au sens TRL — elles font partie de la "completion" multi-tour. C'est un choix de modélisation important : on traite l'épisode entier comme une seule grosse completion, dont le prompt initial est l'état de la conversation au tour zéro.

Ligne 141, on délègue la génération à `trainer._generate_single_turn`. C'est une méthode interne de `GRPOTrainer` qu'on réutilise pour ne pas réécrire toute la logique de batching, de mixed precision et d'extraction des log-probs. Elle prend une liste de prompts encodés, et elle retourne pour chaque prompt deux choses : la liste des tokens générés, et la liste des log-probabilités associées (le log de la probabilité que le modèle a donné à chaque token au moment où il l'a échantillonné). Ces log-probs sont absolument cruciales pour la suite, parce que c'est elles qui permettent à TRL de calculer le ratio d'importance sampling dans la loss GRPO. Si on les zappait ou si on les bricolait, le gradient serait silencieusement faux.

Le bloc lignes 146-166 ferme le tour pour chaque sample actif. Pour chaque sample, on récupère ses tokens et ses log-probs. On décode les tokens en texte, on append cette réponse comme nouveau message d'assistant dans le state (pour que le tour suivant en tienne compte). On extrait la première action du texte avec notre utilitaire, et on l'envoie au serveur TextCraft en préfixant `Action: ` (le serveur veut ce préfixe pour parser proprement). Le serveur retourne un `step` qui contient trois choses : la nouvelle observation, le reward de ce step, et un drapeau `done` qui indique si l'épisode est terminé. On append cette nouvelle observation comme message d'utilisateur dans le state.

Puis on accumule. On étend `completion_ids_out[i]` avec les tokens générés à ce tour, et on étend `logprobs_out[i]` avec les log-probs correspondantes. Au bout de plusieurs tours, ces deux listes deviennent la concaténation complète de tout ce que le modèle a produit pendant tout l'épisode. C'est exactement ce que TRL attend : un seul long bloc de tokens pour la completion, accompagné des log-probs alignées un pour un.

Pour le reward final du sample, on prend le maximum entre ce qu'on avait avant et le reward du step courant. Comme TextCraft ne donne un reward non nul qu'en fin d'épisode (succès ou pas, sparse 0 ou 1), ce maximum capture proprement le résultat final dès qu'il survient. On compte aussi le nombre d'observations qui contiennent des messages d'erreur ("could not", "wrong item format"), pour le passer comme métrique à la reward function. Et on met à jour le drapeau `done` du sample.

Une fois tous les samples terminés ou la limite de vingt tours atteinte, on ferme tous les clients TextCraft pour libérer les ressources côté serveur (lignes 168-172). Puis on fait une dernière passe défensive (lignes 175-180) pour garantir que chaque sample a au moins un token de completion et un de prompt — sinon TRL plante avec une erreur d'array vide. Si jamais un sample n'a rien produit du tout, on met un seul token EOS pour être valide.

La fonction retourne un dictionnaire avec les trois champs imposés par le contrat TRL (`prompt_ids`, `completion_ids`, `logprobs`) plus deux champs supplémentaires (`episode_reward`, `invalid_steps`). Ces deux champs supplémentaires ne sont pas standard, mais TRL a un mécanisme appelé "extra reward kwargs" qui les forwarde automatiquement à la fonction de reward. C'est par ce canal qu'on transporte les métriques d'épisode jusqu'à la fonction qui en a besoin.

### La fonction de reward : `textcraft_reward`

Cette fonction est petite mais elle illustre une décision de design importante. Elle reçoit en argument les prompts du batch, les completions générées, et les deux extras qu'on a poussés depuis le rollout : `episode_reward` et `invalid_steps`. Elle doit retourner une liste de scalaires, un par sample.

Pour chaque sample, on part du reward d'épisode (qui est zéro ou un selon que l'agent a réussi à crafter l'objet final), et on lui ajoute un shaping fin. Si le modèle a généré exactement une action par message en moyenne, on ajoute un petit bonus de deux centièmes pour récompenser la bonne hygiène conversationnelle. Si zéro ou plus d'une action, on retire cinq centièmes pour pénaliser le format dégradé. Et on retire un centième par step invalide (chaque fois que le serveur a refusé une action). Ce shaping est volontairement faible en magnitude par rapport au reward sparse principal — l'idée n'est pas de réécrire le but du jeu, juste de donner au gradient une indication directionnelle quand l'agent fait n'importe quoi formellement.

Pourquoi calculer le nombre d'actions ici plutôt que dans le rollout ? Parce que le shaping s'appuie sur la completion finale (la concaténation de tous les tours de l'agent), pas sur chaque tour individuel. C'est plus propre de le faire au moment où on a la vue d'ensemble.

### Le main : tout brancher ensemble

`main` est l'orchestrateur. Il lit les arguments en ligne de commande pour qu'on puisse facilement varier la taille du run sans toucher au code (`--max-items`, `--max-steps`, `--num-generations`, `--run-name`). Il vérifie que le serveur TextCraft répond, construit le dataset à partir de nos prompts, charge le tokenizer Qwen et configure son token de padding (Qwen utilise EOS comme padding par défaut, c'est un détail mais sans ça le batching plante).

Vient ensuite la configuration de TRL elle-même, dans `GRPOConfig`. Chaque champ correspond à une décision qui mérite d'être consciente. `per_device_train_batch_size=1` parce qu'avec Qwen-3B et le rollout multi-tour qui mange beaucoup de tokens, on tient à peine un sample en mémoire. `learning_rate=1e-6` est volontairement très bas pour ne pas casser le modèle au premier step — c'est l'ordre de grandeur recommandé pour du fine-tuning RL. `num_generations=2` veut dire que pour chaque prompt, on génère deux completions différentes ; GRPO calcule l'avantage en comparant les rewards des completions au sein du même groupe (d'où "Group-Relative"), et il faut au moins deux completions pour qu'il y ait un groupe à comparer. `generation_batch_size=2` est aligné sur ça, parce que TRL impose la divisibilité. `max_completion_length=128` plafonne la longueur de chaque tour de génération à 128 tokens, suffisant pour une réponse "Thought / Action" raisonnable et indispensable pour ne pas exploser la mémoire. `bf16=True` active la demi-précision Brain. `gradient_checkpointing=True` réduit la mémoire activations en recalculant les forwards intermédiaires pendant le backward — coût en temps, gain en VRAM.

Le `peft_config` décrit le LoRA. Rang 16, alpha 32, dropout 0.05, sur tous les sept modules linéaires d'attention et de MLP de Qwen. On entraîne donc seulement ces matrices basse-dimension, qui font une vingtaine de méga-octets, le reste du modèle reste figé.

L'instanciation de `GRPOTrainer` plus bas est l'endroit où tout converge. On lui passe le chemin du modèle (TRL chargera lui-même Qwen depuis le disque et l'enverra sur GPU), notre fonction de reward, la config, le dataset, le tokenizer en tant que `processing_class` (TRL utilise ce mot pour rester cohérent avec son support multi-modal), la config LoRA pour qu'il greffe les adaptateurs au chargement, et enfin notre `rollout_func` custom qui remplace la génération par défaut. Tu peux voir cette dernière ligne comme l'override clé : c'est elle qui transforme un trainer GRPO standard single-turn en trainer multi-turn TextCraft-aware.

`trainer.train()` lance la boucle. À chaque step, TRL appelle notre `rollout_func`, récupère prompts/completions/logprobs, calcule notre reward, calcule l'avantage relatif, calcule la loss GRPO, fait un backward, fait un step d'optimizer. Et il recommence pour `max_steps` itérations.

Le bloc final, lignes 282-285, set deux variables d'environnement avant le démarrage. `TOKENIZERS_PARALLELISM=false` désactive le parallélisme des tokenizers, ce qui évite des warnings et des deadlocks quand on est dans un contexte multi-process. `VLLM_ATTENTION_BACKEND=XFORMERS` force vLLM à utiliser xformers comme backend d'attention, parce que le backend par défaut ne marche pas toujours bien sur certains drivers CUDA — c'est un héritage de tâtonnements précédents.

---

## Version courte (cheatsheet)

**`textcraft_rollout_func`** — Le cœur multi-tour. À chaque step de TRL, on reçoit un batch de prompts. Pour chaque sample on instancie un client TextCraft, on le `reset` sur l'item correspondant (récupéré via le marqueur `<ITEM_IDX:N>` planqué dans le prompt), et on remplace ce marqueur par la vraie observation initiale. Ensuite on entre dans une boucle de tours (max 20). À chaque tour, pour chaque sample encore actif, on rend la conversation en chat template, on appelle `trainer._generate_single_turn` pour obtenir les tokens générés et leurs log-probs, on extrait la première `Action:`, on l'envoie au serveur, on append l'observation reçue, on accumule tokens et log-probs dans des listes par sample. On retourne à TRL un dict avec `prompt_ids` (état initial), `completion_ids` (concat de tous les tours), `logprobs` (alignées un pour un, indispensables pour le ratio d'importance sampling), plus deux extras `episode_reward` et `invalid_steps` qui voyagent jusqu'à la reward.

**`textcraft_reward`** — La fonction de reward. Pour chaque sample elle part du reward sparse de fin d'épisode (0 ou 1), ajoute un petit bonus de +0.02 si le modèle a écrit exactement une action par message, retire 0.05 sinon, et retire 0.01 par step invalide signalé par le serveur. C'est du shaping volontairement faible pour donner une direction au gradient sans réécrire l'objectif principal.

**`main`** — L'orchestrateur. Lit les arguments, vérifie que le serveur TextCraft répond, construit le dataset de prompts, charge le tokenizer. Crée un `GRPOConfig` avec les paramètres mémoire-friendly (batch=1, max_completion_length=128, bf16, gradient checkpointing) et un `LoraConfig` (rang 16 sur tous les `*_proj`). Instancie `GRPOTrainer` en lui passant le modèle, la fonction de reward, la config LoRA, et — point clé — notre `rollout_func` qui remplace la génération par défaut. Lance `trainer.train()`.

**Bloc `__main__`** — Avant tout, set deux variables d'environnement : `TOKENIZERS_PARALLELISM=false` (évite warnings et deadlocks) et `VLLM_ATTENTION_BACKEND=XFORMERS` (force le backend d'attention compatible avec notre driver). Puis appelle `main`.

---

## Pour creuser

Le fichier le plus utile à lire ensuite est `~/miniconda3/envs/trl-grpo/lib/python3.10/site-packages/trl/trainer/grpo_trainer.py`. La méthode `compute_loss` y donne la formule GRPO en code, et `_generate_single_turn` (qu'on appelle dans notre rollout) est lisible sans trop de magie.
