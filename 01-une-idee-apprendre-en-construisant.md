# Chapitre 1 — Une idée : apprendre en construisant

*Ce chapitre répond à la question : pourquoi construire un petit modèle quand
on peut utiliser un grand ? À la fin, le lecteur connaît la chaîne complète du
projet et sait où se trouve chaque étape dans le dépôt.*

------------------------------------------------------------------------

## 1.1 La boîte noire

Les grands modèles de langage donnent l'impression d'être des boîtes noires.
On pose une question, une réponse apparaît, et les mécanismes qui rendent cela
possible restent invisibles. On peut lire des explications, suivre des cours,
regarder des schémas d'attention : tant que rien n'a été exécuté, ces
explications restent abstraites.

L'approche choisie ici est celle du laboratoire : **reconstruire une version
réduite, mais complète**. Réduire l'échelle ne signifie pas supprimer les
étapes. Au contraire, avec peu de données et peu de paramètres, chaque
décision devient visible. Quand une réponse est mauvaise, on peut remonter le
fil : est-ce le texte d'entraînement ? le découpage en tokens ? le format du
prompt ? le réglage de la génération ? la base de connaissances ?

> **Le principe du chapitre**
> Un petit modèle ne sert pas à rivaliser avec les systèmes industriels. Il
> sert à rendre chaque étape suffisamment simple pour être examinée, cassée
> et réparée.

C'est l'esprit de **PicoGPT1-Daisy**. Mais cette idée ne sort pas de nulle part :
elle vient d'un ensemble de ressources — le code open source de
`rasbt/LLMs-from-scratch` compris avec ChatGPT, des articles et, surtout, des
tutoriels YouTube — puis de deux dépôts qui racontent comment elle est devenue
un modèle. La section suivante les remet dans l'ordre ; le
[chapitre 2](02-une-inspiration-pedagogique.md) entre dans les choix techniques.

------------------------------------------------------------------------

## 1.2 Le point de départ : des ressources variées, puis deux dépôts

### Le code de *LLMs from Scratch*, pas le livre

Parmi les ressources qui ont nourri le projet, le dépôt de code
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) de
**Sebastian Raschka** occupe une place structurante. **Précision importante :
l'auteur n'a pas lu le livre.** Il a parcouru le code open source (sous licence
Apache-2.0) et s'est aidé de ChatGPT pour le comprendre, avant de reconstruire
les pièces dans `picogpt-python` (juin 2025). C'est de là que vient la méthode :
ne pas consommer un modèle comme une boîte noire, mais assembler les étapes les
unes après les autres pour voir comment du texte devient une réponse.

### Les articles et, surtout, les tutoriels YouTube

À ce code se sont ajoutées **beaucoup de lectures d'articles** et, plus encore,
**des tutoriels YouTube**. C'est même le canal le plus présent dans le parcours :
voir une architecture se monter et un modèle s'entraîner à l'écran rend concret
ce que la lecture seule laisse abstrait. L'auteur ne se souvient plus de tous
les titres ; le récit ne les inventera donc pas. Quelques traces subsistent
toutefois dans les README : le premier cite *Attention Is All You Need* et
l'article GPT-2, celui d'avril 2026 mentionne une vidéo. Le
[chapitre 2](02-une-inspiration-pedagogique.md) reprend ces éléments et la question
des licences.

> **Source.** L'origine est rapportée par l'auteur du projet ; les dépôts
> cités ci-dessous permettent de vérifier la suite, étape par étape. Les
> références de lectures retrouvées pourront être ajoutées au chapitre 2.

### Juin 2025 — `picogpt-python` : apprendre l'architecture

Premier dépôt : [nazimboudeffa/picogpt-python](https://github.com/nazimboudeffa/picogpt-python).
Tout y est écrit à la main dans `model.ipynb` : `LayerNorm`, `FeedForward`,
`MultiHeadAttention` et `GPTModel`. Le tokenizer vient de `tiktoken`
(encodage `gpt2`), les données sont des phrases anglaises simples
(`simple_english_phrases.txt`), et la configuration annonce 4 couches,
256 dimensions, 128 tokens de contexte et 4 têtes d'attention.

Le nom est déjà un clin d'œil : le README cite
[jaymody/picoGPT](https://github.com/jaymody/picoGPT) comme inspiration.
« PicoGPT » signifie « un petit GPT » ; `-python` marque la version Python de
cette exploration.

### Avril 2026 — `picogpt1-scratch-python` : le premier modèle, avec l'IA

Deuxième dépôt : [nazimboudeffa/picogpt1-scratch-python](https://github.com/nazimboudeffa/picogpt1-scratch-python).
Trois commits, tous datés du 19 avril 2026 : `first commit`, `train and test`,
puis l'ajout de `llama.cpp` en sous-module. La méthode change du tout au tout :

- le jeu **Alpaca** remplace les phrases simples ;
- l'architecture passe à **2 couches / 128 dimensions** ;
- l'entraînement est confié au `Trainer` de Hugging Face dans `train.ipynb` ;
- le modèle est converti en GGUF et essayé avec Ollama.

C'est ici que l'IA devient un partenaire de **développement** — elle servait
déjà, depuis 2025, d'aide à la compréhension du code ; le sujet du
[chapitre 3](03-programmer-avec-l-aide-de-l-ia.md). Et c'est ici que le « 1 »
apparaît. La description du dépôt est explicite :

> *« After PicoGPT basic for learning purpose, here is a first model, so no
> need to start from scratch now and after »*

Autrement dit : après le PicoGPT « pour apprendre », voici **un premier
modèle**. « PicoGPT1 » désigne donc la première génération de modèles
entraînés, et non une version 1.0 au sens commercial. La généalogie complète,
avec Jasmine et Daisy, est récapitulée dans le [sommaire](00-sommaire.md).

> **À compléter —** Confirme (ou corrige) cette interprétation du « 1 ». Si tu
> préfères une autre formulation, c'est ici qu'elle doit être écrite : le
> reste du livre s'y conformera.

------------------------------------------------------------------------

## 1.3 Neuf étapes, une seule chaîne

Le dépôt n'est pas un programme unique : c'est une succession de scripts,
chat et par programme, qui transforment du texte en outil utilisable. Le
tableau suivant est la carte du livre autant que celle du projet.

| # | Étape | Fichier principal | Question que l'étape résout |
| --- | --- | --- | --- |
| 1 | Installer l'environnement | `doc/02-installation.md` | De quoi ai-je besoin pour exécuter la chaîne ? |
| 2 | Choisir et filtrer des instructions | `prepare_dataset.py` | Quels exemples apprennent quelque chose de correct ? |
| 3 | Écrire une base fermée de faits | `knowledge.jsonl` | Sur quoi pourrai-je vérifier une réponse ? |
| 4 | Fabriquer des exemples ancrés et directs | `prepare_rag_dataset.py`, `prepare_facts_dataset.py` | Comment apprendre à répondre *et* à refuser ? |
| 5 | Tokeniser et masquer le prompt | `train.py` (`build_example`) | Que doit exactement apprendre le modèle ? |
| 6 | Entraîner | `train.py` | Comment mettre à jour les poids ? |
| 7 | Générer | `model_utils.py`, `test.py` | Comment obtenir une réponse lisible, sans boucle ? |
| 8 | Ancrer et évaluer | `rag.py`, `evaluate.py` | Comment éviter d'inventer, et comment le mesurer ? |
| 9 | Servir et exporter | `serve.py`, `export_gguf.py`, `publish_hf.py` | Comment rendre le modèle utilisable ailleurs ? |

Deux idées importantes se dégagent déjà.

**D'abord, la séparation entre données d'entraînement et informations
consultables.** `knowledge.jsonl` est versionné : c'est une source de
vérité, petite et contrôlée. Les fichiers `train.txt`, `rag_train.txt` et
`facts_train.txt` sont générés puis ignorés par Git. Modifier un fait dans la
base ne suffit pas : il faut régénérer les exemples, puis réentraîner.

**Ensuite, le refus fait partie de l'apprentissage.** Le projet n'apprend pas
seulement à répondre ; il apprend aussi à dire `Je ne sais pas.` lorsque la
question sort du périmètre de la base. C'est l'un des apports les plus
concrets de l'étape RAG (chapitre 10).

------------------------------------------------------------------------

## 1.4 Deux époques dans une même histoire

Le brouillon de ce livre décrivait un modèle de **6,86 millions de
paramètres**, entraîné sur **745 exemples**, en **10 époques**, avec une loss
finale d'environ **4,54**. Ces chiffres sont exacts — mais ils décrivent la
première époque du projet, avant le 21 novembre 2026. Le `README.md` de
`picogpt1-daisy` affichait encore les mêmes valeurs avant le commit `12c8df5`.

Le 21 novembre, le projet change de stratégie : au lieu d'apprendre la langue
depuis des poids aléatoires sur quelques milliers d'exemples, il part de
**GPT-2 pré-entraîné** et se contente d'un ajustement (*fine-tuning*).

| | Première époque (2 couches / 128 dims) | Époque actuelle (`pretrained`) |
| --- | --- | --- |
| Commande | Aucune option CLI (configuration dans `train.py`) | `python train.py` |
| Poids de départ | Aléatoires | `gpt2` (124 M paramètres) |
| Architecture | 2 couches, 128 dimensions, 2 têtes | 12 couches, 768 dimensions, 12 têtes |
| Paramètres | ≈ 6,86 M | 124 439 808 |
| Exemples | 745 (500 + 245) | 2 442 (2 000 + 245 + 197) |
| Époques de référence | 10 | 5 |
| Loss rapportée | ≈ 4,54 | 1,58 en moyenne, ≈ 1,05 en fin de course |
| Matériel | CPU | RTX 3070, CUDA + `--fp16`, 250,8 s |

**À ne pas confondre.** L'option `--init scratch`, ajoutée au commit `12c8df5`,
ne reproduit pas la première époque : elle utilise 4 couches / 256 dimensions /
4 têtes, soit ≈ 16,1 M paramètres (`doc/09-architecture.md`). C'est un outil
de comparaison rétrospectif, pas un retour en arrière.

Le code lui-même conserve la mémoire des deux époques. L'en-tête de
`train.py` (lignes 1-12) l'explique sans détour :

```python
"""Entraîne le mini GPT-2, ou ajuste un GPT-2 pré-entraîné.

Usage :
    python train.py                          # GPT-2 pré-entraîné (recommandé)
    python train.py --init scratch           # ancien comportement, depuis zéro
    python train.py --epochs 5 --lr 3e-5     # réglages personnalisés

Le modèle pré-entraîné est celui qui donne des réponses cohérentes dans le
GGUF / llama.app : avec 745 exemples, un modèle entraîné de zéro (même plus
grand) ne peut pas apprendre la langue correctement. Le mode ``scratch`` est
conservé pour la démonstration pédagogique.
"""
```

**Lecture commentée.** La phrase clé est la dernière : avec 745 exemples, un
modèle aléatoire ne peut pas apprendre la langue. Ce n'est pas une opinion
écrite après coup, c'est la leçon tirée d'un entraînement réel, conservée dans
le code pour éviter de la refaire. Le mode `scratch` reste disponible
précisément parce que le projet est pédagogique : on peut voir ce que donne
un entraînement naïf, puis comparer.

> **Leçon du chapitre**
> « Entraîné de zéro » peut désigner deux choses : la démarche d'apprentissage
> (reconstruire soi-même la chaîne) et l'initialisation des poids (aléatoire).
> Le livre garde le premier sens ; le projet utilise désormais le second pour
> la comparaison.

------------------------------------------------------------------------

## 1.5 Le parcours du lecteur

La séquence complète tient en huit commandes dans le bloc ci-dessous. Le
`doc/README.md` de `320a572` en affiche sept et présente `python serve.py`
séparément. Elle suppose le dépôt cloné et l'environnement virtuel activé
(chapitre 5).

```bash
python -m pip install torch transformers datasets huggingface_hub scikit-learn

python prepare_dataset.py --max-examples 2000   # train.txt : 2 000 exemples Alpaca filtrés
python prepare_rag_dataset.py                   # rag_train.txt : 245 exemples ancrés
python prepare_facts_dataset.py                 # facts_train.txt : 197 Q/R directes
python train.py                                 # ajustement de GPT-2

python test.py --prompt "Quelle est la capitale de la France ?" --rag
python evaluate.py
python serve.py                                 # API locale sur le port 8000
```

Trois points à remarquer avant d'aller plus loin.

1. **`scikit-learn` n'est pas optionnel** dès qu'on touche au RAG. Sans lui,
   `rag.py` ne peut pas construire son index TF-IDF.
2. **Les poids ne sont pas dans Git.** Le `.gitignore` exclut
   `*.safetensors`, `*.gguf`, les fichiers `*_train.txt` et les checkpoints.
   Après un clonage, il faut soit entraîner, soit télécharger le modèle
   publié sur Hugging Face.
3. **La génération ajoute le saut de ligne final.** Le modèle a été entraîné
   avec `prompt` + `\n` + `réponse` + `<|endoftext|>` ; `generate()` ajoute
   donc `\n` s'il manque. Ce détail apparemment minuscule explique beaucoup de réponses
   « qui démarrent au milieu de la ligne ».

------------------------------------------------------------------------

## 1.6 Vérifier par soi-même

Avant de lire la suite, quelques commandes pour ancrer le récit dans la
réalité du dépôt :

```bash
# Voir la chronologie réelle des étapes
git log --format='%h %ad %s' --date=short --reverse

# Compter les faits de la base fermée (30 lignes attendues)
(Get-Content knowledge.jsonl | Where-Object { $_.Trim() }).Count

# Compter les exemples de chaque jeu généré : les scripts écrivent une ligne
# vide ENTRE les blocs, donc n exemples donnent n-1 lignes vides (d'où le + 1).
(Get-Content train.txt | Where-Object { $_ -match '^$' }).Count + 1
(Get-Content rag_train.txt | Where-Object { $_ -match '^$' }).Count + 1
(Get-Content facts_train.txt | Where-Object { $_ -match '^$' }).Count + 1

# Voir les options réellement disponibles pour l'entraînement
python train.py --help
```

> **À compléter —** Dans ta version du livre, ajoute ici ce que tu as
> ressenti en voyant le premier entraînement « depuis zéro » produire des
> réponses incohérentes, avant la bascule vers GPT-2 pré-entraîné. C'est le
> moment du récit où le chiffre 4,54 devient une expérience plutôt qu'une
> statistique.

------------------------------------------------------------------------

## 1.7 Ce qu'il faut retenir

- Le projet est un **laboratoire** : l'objectif est de comprendre la chaîne,
  pas de fabriquer un assistant généraliste.
- L'idée vient d'un ensemble de ressources — le **code** open source de
  `rasbt/LLMs-from-scratch` étudié avec ChatGPT (le livre n'a pas été lu), des
  articles et surtout des tutoriels YouTube — puis passe par `picogpt-python`
  (juin 2025, architecture écrite à la main) et `picogpt1-scratch-python`
  (avril 2026, premier modèle avec l'IA) avant Jasmine et Daisy.
- La chaîne compte **neuf étapes** et **treize fichiers** (certaines étapes en
  regroupent plusieurs), décrits dans douze fiches `doc/` et un index.
- Les données sont de deux natures : une **base fermée versionnée**
  (`knowledge.jsonl`) et des **jeux générés** non versionnés.
- Le projet a connu **deux époques** : 6,86 M paramètres depuis zéro
  (2 couches / 128 dims), puis 124 M par ajustement de GPT-2. Le mode
  `scratch` de Daisy (4 couches / 256 dims, ≈ 16,1 M) sert à la comparaison.
- Le **refus de répondre** est une fonctionnalité à part entière, pas un
  effet de bord.

**Suite :** [Chapitre 2 — Une inspiration pédagogique](02-une-inspiration-pedagogique.md),
où l'on sépare ce qui vient du code de référence, des articles et des tutoriels
de ce qui est propre à Daisy.
