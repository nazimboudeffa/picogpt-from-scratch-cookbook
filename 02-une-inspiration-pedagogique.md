# Chapitre 2 — Une inspiration pédagogique

*Ce chapitre répond à la question : qu'est-ce qui vient des ressources
pédagogiques existantes, qu'est-ce qui a été adapté, qu'est-ce qui est né
dans le projet ? À la fin, le lecteur sait lire `train.py` comme un choix
d'architecture plutôt que comme une recette tombée du ciel.*

------------------------------------------------------------------------

## 2.1 Un cadre parmi d'autres : code, articles et tutoriels

La référence écrite la plus clairement identifiée n'est pas le livre, mais le
**code open source** du dépôt
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), publié
par **Sebastian Raschka** sous licence Apache-2.0. **L'auteur n'a pas lu le
livre** : il a parcouru le code et s'est aidé de ChatGPT pour le comprendre.
L'apprentissage s'est aussi nourri de **nombreux articles** et, **surtout, de
tutoriels YouTube**, dont il ne se souvient plus de tous les titres. Ce chapitre
présente le cadre apporté par ce code, sans faire croire qu'il a tout apporté.

Le dépôt de code accompagne le lecteur dans la construction progressive d'un
modèle de langage de type GPT. Il aborde notamment :

- la préparation du texte et la tokenisation ;
- les mécanismes d'attention ;
- l'assemblage d'une architecture GPT ;
- le préentraînement sur un corpus ;
- l'ajustement sur une tâche particulière (*fine-tuning*).

Pour PicoGPT1-Daisy, l'apport principal est **un ordonnancement des
questions**. Plutôt que d'utiliser un modèle prêt à l'emploi comme une API,
on se demande successivement : comment représenter le texte ? comment
découper les exemples ? que doit-on masquer dans les labels ? comment
s'arrêter lors de la génération ?

> **Inspiration n'est pas reproduction — mais elle a d'abord été imitation.**
> Le projet ne recopie pas le code de la référence dans sa version finale. Il
> retient la progression pédagogique, puis fait ses propres choix. Nuance
> historique importante : dans `picogpt-python` (juin 2025), les blocs
> d'attention **étaient** écrits dans le notebook (`MultiHeadAttention` dans
> `model.ipynb`), en s'inspirant de près du code amont. C'était la partie
> littéralement « from scratch » de l'apprentissage — au sens où l'étudiant
> reconstruisait les blocs, pas au sens où il les aurait inventés. Le passage
> au `Trainer` en avril 2026, puis à
> `GPT2LMHeadModel` dans Daisy, déplace ensuite l'effort de l'implémentation
> des blocs vers le chemin des données.

Cette distinction est centrale pour l'honnêteté du récit. Un lecteur qui
chercherait dans `train.py` d'aujourd'hui une implémentation manuelle du
*self-attention* n'en trouverait plus : elle est déléguée à `GPT2LMHeadModel`.
Ce qui est reconstruit désormais, ce n'est pas le tenseur d'attention, c'est
**le parcours complet de données à outil**.

------------------------------------------------------------------------

## 2.2 La généalogie complète : cinq dépôts

Le projet ne commence pas avec Jasmine. Il passe par plusieurs dépôts publics,
chacun conservant une étape de l'apprentissage. Les dates ci-dessous
proviennent des commits.

| Période | Dépôt | Avancée | Trace vérifiable |
| --- | --- | --- | --- |
| Juin 2025 | [`picogpt-python`](https://github.com/nazimboudeffa/picogpt-python) | Architecture écrite à la main : 4 couches, 256 dimensions, 128 tokens de contexte, 4 têtes ; tokenizer `tiktoken`, phrases anglaises simples | `model.ipynb` : `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `GPTModel` |
| 19 avr. 2026 | [`picogpt1-scratch-python`](https://github.com/nazimboudeffa/picogpt1-scratch-python) | Premier modèle entraîné : Alpaca, 2 couches / 128 dimensions, `Trainer`, GGUF et Ollama | 3 commits ; `train.ipynb` (452 lignes au HEAD, 458 au commit `b033165`), `PicoGPT1-Scratch/` |
| 20-21 nov. 2026 | [`picogpt1-jasmine`](https://github.com/nazimboudeffa/picogpt1-jasmine) | Passage aux scripts Python, publication Hugging Face, documentation en 11 fiches + un index | 6 commits (le 6ᵉ le 21 nov. à 00 h 24) ; `MODEL_CARD.md`, `doc/` |
| 20-21 nov. 2026 | [`picogpt-kit`](https://github.com/nazimboudeffa/picogpt-kit) | RAG, abstention, décodage anti-boucles, puis GPT-2 pré-entraîné 124 M | 6 commits à `320a572` ; `rag.py`, `knowledge.jsonl`, `train.py` |
| 2026 | [`picogpt-from-scratch-cookbook`](https://github.com/nazimboudeffa/picogpt-from-scratch-cookbook) | Ce livre : mise en récit et vérification des chiffres | `00-sommaire.md`, chapitres 01-04 |

Le premier README cite *Attention Is All You Need*, l'article GPT-2 et
[jaymody/picoGPT](https://github.com/jaymody/picoGPT), mais **pas** le dépôt de
Raschka. Celui d'avril 2026 mentionne une vidéo et le dépôt
[hassanhabib/LLMfromNothing](https://github.com/hassanhabib/LLMfromNothing).
Ces traces confirment ce que récapitule le [chapitre 1](01-une-idee-apprendre-en-construisant.md) :
à côté du code de Raschka, il y a eu **beaucoup d'articles** — dont les
références exactes ne sont plus toutes connues — et **des tutoriels YouTube**,
qui ont pesé au moins autant dans la manière de voir le modèle se construire.
Le dépôt de Raschka n'est donc pas l'unique point de départ : c'est la référence
open source la plus clairement identifiée d'un parcours plus large.

> **Licence et attribution — à régulariser dans `picogpt-python`.** Le code amont
> est sous **Apache-2.0** (`Copyright (c) Sebastian Raschka`). Le `model.ipynb`
> de juin 2025 reprend des briques très proches du dépôt amont (`GPTDatasetV1`,
> `LayerNorm`, `GELU`, `MultiHeadAttention`, `GPTModel`), alors que le README de
> `picogpt-python` ne le mentionne pas et déclare une licence MIT. Apache-2.0
> autorise ce réemploi, mais exige de conserver l'avis de copyright, de fournir
> un exemplaire de la licence Apache et de signaler les fichiers modifiés. C'est
> un point à corriger dans le dépôt `picogpt-python` ;
> ce livre se contente de le documenter.

> **Décision éditoriale proposée — à confirmer par l'auteur.**
> « PicoGPT » est un clin d'œil au petit GPT de jaymody, cité dans le premier
> README. Le « 1 » apparaît avec `picogpt1-scratch-python`, dont la
> description dit : *« After PicoGPT basic for learning purpose, here is a
> first model, so no need to start from scratch now and after »*. Dans ce
> livre, on entendra donc **PicoGPT1 = la première génération de modèles
> entraînés**, après la phase où le but était seulement d'apprendre
> l'architecture. Ce n'est pas un numéro de version au sens strict, plutôt un
> nom de famille : Jasmine et Daisy en sont les étapes suivantes.

### Le tableau Jasmine → Daisy

Le 20 novembre 2026, le commit `1c00747` — « Documenter les differences avec
PicoGPT1-Jasmine » — ajoute 19 lignes à `MODEL_CARD.md`. C'est une trace
précieuse : le projet prend le temps de se situer par rapport à son propre
passé.

À cette date, le tableau comparatif suivant était écrit :

| Élément | `PicoGPT1-Jasmine` (ancien) | `PicoGPT1-Daisy` (à ce moment) |
| --- | --- | --- |
| Architecture | GPT-2 : 2 couches, 128 dims, 2 têtes, contexte 256 | Identique (6,86 M paramètres) |
| Jeu d'entraînement | 500 exemples Alpaca bruts | 500 Alpaca **filtrés** + 245 ancrés = **745 exemples** |
| Données | Erreurs factuelles et doublons | Erreurs retirées (eau à 115 °C, fausse définition du « mole », `<nooutput>`), doublons supprimés |
| Décodage | Glouton, boucles fréquentes | Échantillonnage par défaut (`top_k=40`, `no_repeat_ngram_size=3`, `repetition_penalty=1.15`), arrêt à l'EOS ou au premier `\n` |
| Ancrage | Aucun | `knowledge.jsonl` (30 faits), TF-IDF, copie contrainte, `Je ne sais pas.` hors base |
| Entraînement | 10 epochs, ≈ 1 250 steps, loss ≈ 4,95 | 10 epochs, ≈ 1 860 steps, loss ≈ 4,54 |

La conclusion du `MODEL_CARD.md` de l'époque tient en une phrase : *même
architecture, mais données nettoyées, décodage corrigé et ancrage
anti-hallucination*. Le changement le plus important n'était donc pas le
modèle, mais **le comportement autour du modèle**.

Le lendemain, le commit `12c8df5` change cette donne : l'architecture passe à
un GPT-2 complet pré-entraîné. Le tableau ci-dessus doit donc être lu comme
une photographie datée, pas comme l'état actuel. C'est exactement le genre de
piège que ce livre cherche à éviter : dater chaque chiffre.

> **À compléter —** Pourquoi le nom « Daisy » plutôt qu'une simple version 2 ?
> S'il y a une anecdote derrière le renommage de `b5f05c6`, c'est ici qu'elle
> appartient. Les commits ne la racontent pas.

------------------------------------------------------------------------

## 2.3 Inspiré, adapté, propre au projet

La matrice suivante répartit les choix de `train.py` et des scripts voisins.
Les références de lignes permettent de vérifier chaque affirmation.

| Ce qui est… | Élément | Où le voir |
| --- | --- | --- |
| **Inspiré du code de référence** | Tokenizer GPT-2, vocabulaire de 50 257 tokens | `train.py:138` (chargement), `train.py:149` (affichage), `doc/09-architecture.md` |
| | Format instruction → réponse avec token de fin | `train.py:55`, `train.py:141-148` |
| | Masquage des tokens du prompt dans les labels | `train.py:59-60` |
| | Boucle d'entraînement avec `Trainer` | `train.py:188-214` |
| **Adapté au projet** | Troncature à 256 tokens malgré 1 024 positions disponibles | `train.py:34,140,162-163` |
| | `DataCollatorForSeq2Seq` avec `label_pad_token_id=-100` | `train.py:201-206` |
| | Deux modes d'initialisation (`pretrained` / `scratch`) | `train.py:103-106,159-181` |
| | Conservations du tokenizer local pour éviter une requête réseau | `train.py:132-138` |
| **Propre à PicoGPT1-Daisy** | Base fermée de 30 faits en français | `knowledge.jsonl`, `rag.py:16` |
| | Score TF-IDF minimal et abstention `Je ne sais pas.` | `rag.py:18,23,65-75` |
| | Décodage anti-boucles et arrêt au saut de ligne | `model_utils.py:21-40,98-115` |
| | Génération de 197 réponses directes pour llama.app | `prepare_facts_dataset.py` |
| | Export GGUF + publication Hugging Face | `export_gguf.py`, `publish_hf.py` |

Deux lignes de cette table méritent un commentaire.

**Le masquage des labels** est une technique classique du *fine-tuning*
d'instructions : on calcule la perte uniquement sur la réponse. Le modèle
n'est pas pénalisé pour avoir écrit le prompt, il apprend à le compléter.

**Le choix du `Trainer`** est un compromis assumé. Écrire sa propre boucle
d'entraînement aurait été plus « from scratch » au sens littéral, mais aurait
déplacé l'attention du projet : au lieu de comprendre le chemin des données,
le lecteur aurait passé son temps à déboguer un chargeur de batchs.

------------------------------------------------------------------------

## 2.4 Premier extrait commenté : `build_example()`

Voici le premier morceau de code réellement commenté du livre. Il se trouve
dans `train.py`, lignes 53-66, et tient en une dizaine de lignes — assez pour
servir de microcosme du projet.

```python
def build_example(tokenizer, prompt, completion, max_len=MAX_LEN):
    """Tokenise un couple prompt/réponse et masque le prompt dans les labels."""
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    tokens = tokenizer(text, truncation=True, max_length=max_len)
    input_ids = tokens["input_ids"]

    prefix_ids = tokenizer(f"{prompt}\n", truncation=True, max_length=max_len)["input_ids"]
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }
```

Commentaire ligne par ligne :

1. **`text = f"{prompt}\n{completion}{tokenizer.eos_token}"`** — l'exemple appris est une
   seule chaîne : le prompt, un saut de ligne, la réponse, puis le token de fin
   `<|endoftext|>`. Le saut de ligne est donc **fourni**, pas généré. C'est
   pour cela que `generate()` devra l'ajouter plus tard (chapitre 9).
2. **`truncation=True, max_length=256`** — au-delà de 256 tokens, l'exemple
   est coupé. Comme le modèle hérite de 1 024 positions, les positions
   supplémentaires existent mais ne sont jamais utilisées en entraînement.
3. **`input_ids`** — la suite d'identifiants que le modèle recevra.
4. **`prefix_ids`** — on re-tokenise uniquement `prompt\n` pour connaître sa
   longueur exacte en tokens. Compter les caractères ne suffirait pas : un
   token GPT-2 peut couvrir plusieurs lettres.
5. **`labels = [-100] * len(prefix_ids) + …`** — `-100` est la valeur
   conventionnelle que PyTorch ignore dans le calcul de la perte. Les tokens
   du prompt deviennent des cibles « invisibles » ; seuls ceux de la réponse
   et du token de fin comptent.
6. **`attention_mask`** — transmis tel quel au collateur, qui s'occupe du
   padding des batchs.

Un exemple concret avec la question de la capitale :

| Position | Contenu | `input_ids` | `labels` |
| --- | --- | --- | --- |
| 1…n | `Quelle est la capitale…\n` | identifiants du prompt | `-100` |
| n+1 | `Paris` | identifiant(s) de `Paris` | appris |
| n+2 | `<|endoftext|>` | 50256 | appris |

C'est cette ligne `labels` qui explique pourquoi le modèle, à la génération,
ne sait pas produire le `\n` séparateur : il n'a jamais été entraîné à le
prédire.

------------------------------------------------------------------------

## 2.5 Ce que l'inspiration n'apporte pas

Il faut dire clairement ce que le projet **ne démontre pas** :

- il n'implémente pas l'attention à la main ;
- il ne préentraîne pas un modèle sur un grand corpus (le mode `scratch`
  n'apprend que sur quelques milliers d'exemples) ;
- il ne propose pas de nouvelle architecture ;
- il ne remplace pas une évaluation rigoureuse par une démonstration.

Ce qu'il démontre, en revanche, est moins spectaculaire et plus utile : on
peut assembler, avec des scripts lisibles, une chaîne où chaque étape a une
entrée, une sortie et une vérification possible.

> **Leçon du chapitre**
> L'inspiration pédagogique fournit un ordre de questionnement. Les choix
> d'implémentation, eux, doivent être expliqués à partir du code réel et des
> expériences du projet — pas attribués à une référence externe.

------------------------------------------------------------------------

## 2.6 À compléter

- **Comment le dépôt de code de Raschka a-t-il été découvert** (recommandation,
  recherche, vidéo) ? Quels fichiers ou notebooks as-tu parcourus, et dans quel
  ordre ? Le livre lui-même n'ayant pas été lu, c'est cette question qu'il faut
  documenter.
- **Quels articles et quels tutoriels YouTube** ont accompagné cette lecture ?
  Quelques repères sont déjà rassemblés dans
  l'[annexe G](annexe-g-sources-et-references/annexe-g-sources-et-references.md)
  (dont le tutoriel de Andrej Karpathy). Ceux dont tu te souviens encore
  peuvent être ajoutés ici ; les autres resteront décrits sans titre plutôt
  que d'être inventés.
- **Confirmer l'interprétation de « PicoGPT1 »** proposée plus haut. S'il y a
  une autre raison au « 1 », elle est à écrire ici.
- Dans `picogpt-python`, qu'est-ce qui a été **reproduit fidèlement** du dépôt
  de code (et avec quelle attribution Apache) et qu'est-ce qui a été simplifié
  dès le départ ?
- Qu'a-t-il été **décidé de ne pas reproduire** ensuite, et pourquoi (temps,
  complexité, pertinence) ?
- Le passage de 2 couches / 128 dimensions (Jasmine) à 12 couches / 768
  dimensions a-t-il été discuté avant le commit `12c8df5`, ou est-ce une
  réponse à un essai infructueux ?
- Y a-t-il eu une tentative de boucle d'entraînement écrite à la main avant
  d'adopter `Trainer` ?

**Suite :** [Chapitre 3 — Programmer avec l'aide de l'IA](03-programmer-avec-l-aide-de-l-ia.md),
où l'on raconte la chronologie des décisions à partir des commits.
