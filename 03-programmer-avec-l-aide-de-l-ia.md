# Chapitre 3 — Programmer avec l'aide de l'IA

*Ce chapitre répond à la question : comment raconter honnêtement une
collaboration avec l'IA quand les prompts exacts n'ont pas été conservés ? À
la fin, le lecteur dispose d'une méthode de documentation et de sept zones à
compléter avec ses propres souvenirs.*

> **Avertissement méthodologique**
> Les six commits de `picogpt1-daisy` prouvent **ce qui a changé** et **quand**.
> Ils ne contiennent ni les prompts ni les réponses de l'assistant. Ce chapitre
> ne reconstitue donc que ce que les fichiers permettent d'affirmer. Chaque
> zone `À compléter` doit être remplie à partir de notes ou de souvenirs
> réels — jamais inventée pour arrondir le récit.

------------------------------------------------------------------------

## 3.1 Le rôle du développeur reste essentiel

Le brouillon initial le disait déjà : utiliser l'IA pour programmer ne signifie
pas que le développement s'est déroulé sans intervention humaine. Un projet
logiciel demande de choisir une direction, de comprendre les besoins, de
vérifier le comportement du code et de corriger ce qui ne fonctionne pas.

Dans le cas de PicoGPT1-Daisy, cette responsabilité se voit dans les
résultats :

| Responsabilité humaine | Trace visible dans le dépôt |
| --- | --- |
| Définir un objectif | Le passage d'une génération libre à une réponse ancrée (`9884277`), puis à des réponses directes (`320a572`) |
| Guider les échanges | Le choix de conserver deux modes d'initialisation (`train.py:103-106`) plutôt d'en supprimer un |
| Examiner les propositions | Les commentaires qui expliquent pourquoi une technique a été **abandonnée** (voir §3.3) |
| Lancer les tests et constater les erreurs | Les métriques de `evaluate.py` et les valeurs rapportées dans `doc/12-rag.md` |
| Décider des corrections | Le seuil `DEFAULT_MIN_SCORE = 0.50`, le retrait du décodage contraint, la régénération du GGUF |

L'IA a pu aider à écrire ou modifier des fichiers Python, à expliquer une
fonction, à suggérer des commandes, à analyser un message d'erreur ou à
imaginer des tests — comme le décrivait le brouillon. Ce qui est certain, ce
sont les **artefacts** : scripts, documentation, commits. L'attribution
précise de chaque ligne reste à documenter avec les vrais échanges.

------------------------------------------------------------------------

## 3.2 La chronologie commence avant Daisy

### Avril 2026 : la première collaboration IA

Les six commits de Daisy ont une préhistoire. Le 19 avril 2026, le dépôt
[`picogpt1-scratch-python`](https://github.com/nazimboudeffa/picogpt1-scratch-python)
est créé en trois commits : `first commit`, `train and test`, puis ajout de
`llama.cpp` en sous-module. Le contraste avec `picogpt-python` (juin 2025) est
net :

| | Juin 2025 | Avril 2026 |
| --- | --- | --- |
| Code du modèle | Écrit à la main (`MultiHeadAttention`, `GPTModel`) | Délégué à `GPT2LMHeadModel` via `Trainer` |
| Données | `simple_english_phrases.txt` | `tatsu-lab/alpaca` |
| Architecture | 4 couches / 256 dims / contexte 128 | 2 couches / 128 dims |
| Interface | Notebooks | Notebooks + GGUF + Ollama |
| Taille du checkpoint | — | `model.safetensors` de 27,5 Mo |

C'est ce saut — de l'architecture écrite à la main vers un pipeline
`Trainer` + Alpaca + GGUF — qui marque l'entrée de l'IA comme partenaire de
développement. Les prompts de cette période ne sont pas conservés dans Git ;
seuls les artefacts le sont. C'est pourquoi cette étape est racontée ici
comme un **changement de méthode**, pas comme une conversation reconstituée.

> **À compléter — échange 0 (le plus ancien).** Quelle question as-tu posée
> pour passer de `model.ipynb` à `train.ipynb` + `Trainer` ? Est-ce l'IA qui a
> proposé Alpaca, ou étais-tu parti chercher ce jeu toi-même ?

### `b5f05c6` — 20 novembre : nommer et stabiliser

*« Renommage PicoGPT1-Daisy et decodage anti-boucles »* — 31 fichiers, dont
les scripts, la documentation découpée en douze fiches et deux notebooks.

Le dépôt n'est pas encore un modèle au point : c'est un état initial complet
et versionné. Le choix le plus intéressant concerne le décodage. Le modèle
bouclait en mode glouton ; le code adopté par la suite documentera la
solution dans `model_utils.py` :

```python
class StopAtNewline(StoppingCriteria):
    """Arrête la génération dès qu'une nouvelle ligne apparaît dans la réponse.

    Le modèle a été entraîné avec ``prompt\\nréponse<|endoftext|>``. Un second
    saut de ligne annonce une nouvelle paire question/réponse inventée : on
    s'en sert comme critère d'arrêt, en plus de l'EOS.
    """
```

**Commentaire.** Le critère d'arrêt n'est pas un réglage miracle : il exploite
le format d'entraînement. Le modèle ne sait pas « s'arrêter parce qu'il a
fini » ; il s'arrête parce qu'un saut de ligne survenant après la réponse
ressemble au début d'une nouvelle invention.

> **À compléter — échange 1.** Quel prompt a conduit à ce critère
> `StopAtNewline` ? Quel exemple de boucle avait été observé juste avant ?

### `9884277` — 20 novembre : l'ancrage change de nature

*« Ajout du pipeline RAG et reduction des hallucinations »* — 21 fichiers,
+947/−142 lignes. Apparition de `rag.py`, `evaluate.py`, `knowledge.jsonl`,
`prepare_rag_dataset.py` et du chapitre `doc/12-rag.md`.

C'est le commit le plus structurant. Le modèle ne répond plus seulement :
il doit copier un extrait d'un texte récupéré, ou s'abstenir. Le code de
`rag.py` pose déjà les trois fonctions qui resteront jusqu'à la fin :
`retrieve()`, `build_prompt()` et `is_supported()`.

> **À compléter — échange 2.** Comment le principe du RAG a-t-il été introduit
> dans la conversation ? Quelle alternative a été écartée (plus de données,
> plus d'époques, simple prompt système) ?

### `1c00747` — 20 novembre, 23 h 09 : se situer

*« Documenter les differences avec PicoGPT1-Jasmine »* — un seul fichier,
`MODEL_CARD.md`, +19 lignes. Le tableau comparatif repris au chapitre 2.

Ce commit ne change aucune ligne de code. Il change le **récit** : le projet
s'explique par rapport à son prédécesseur. C'est un rappel utile — écrire la
documentation fait partie du développement, pas d'une phase finale.

> **À compléter — échange 3.** Le tableau comparatif a-t-il été demandé tel
> quel, ou est-il né d'une question plus générale du type « qu'est-ce qui a
> changé ? » ?

### `66e4623` — 20 novembre : choisir des scripts plutôt que des notebooks

*« Supprimer les notebooks obsolètes et nettoyer la documentation »* —
6 fichiers, +3/−590 lignes, principalement `train.ipynb` (452 lignes au HEAD,
458 au commit `b033165`) et `test.ipynb` (125 lignes).

C'est une décision de forme aux conséquences pédagogiques : le projet sera
lisible comme une suite de fichiers Python exécutables, sans dépendre d'un
environnement de notebooks. Le brouillon de notre livre mentionnait encore
ces notebooks dans son annexe : c'est l'un des indices qui dataient le
manuscrit.

> **À compléter — échange 4.** Pourquoi les notebooks sont-ils devenus
> « obsolètes » ? Y avait-il eu des problèmes de reproductibilité ou de
> dépendances ?

### `12c8df5` — 21 novembre : changer d'échelle

*« Passer a GPT-2 pre-entraine, ajouter le support GPU et regenerer la
documentation »* — 17 fichiers, +328/−167. `train.py` est profondément
remanié : `--init`, `--pretrained-model`, `--fp16`, architectures `scratch`
configurables. Onze des treize fichiers `doc/` sont modifiés ; les deux autres
(`05-generation.md`, `06-serveur-local.md`) restent inchangés.

Le code garde la justification de ce choix dans son en-tête (chapitre 1) :
avec 745 exemples, un modèle entraîné de zéro ne peut pas apprendre la langue
correctement. On passe de 6,86 M à 124 M paramètres, de 4,54 à 1,05 de loss
journalisée.

> **À compléter — échange 5.** Quelle a été l'erreur ou l'observation
> déclenchante ? Un essai avec un modèle `scratch` plus grand a-t-il été
> tenté avant de conclure à l'échec de cette voie ?

### `320a572` — 21 novembre : fiabiliser l'ancrage

*« Ajouter les reponses directes et fiabiliser le RAG »* — 16 fichiers,
+248/−106. Nouveau script `prepare_facts_dataset.py` (144 lignes), 197
réponses directes, et surtout une **décision de retrait** dans `rag.py`.

Le chapitre 9 détaillera le mécanisme. Ici, ce qui compte, c'est la méthode :
une technique sophistiquée a été enlevée parce qu'un test a montré qu'elle
nuisait au résultat.

> **À compléter — échange 6.** Quel message d'erreur ou quelle sortie a
> montré que la contrainte de décodage tronquait « 100 °C » en « 100 » ?

------------------------------------------------------------------------

## 3.3 Les traces de décisions dans le code

C'est ici que l'IA et le développeur laissent les traces les plus lisibles :
les commentaires et les docstrings qui expliquent un **échec**. On n'écrit ce
genre de note que lorsqu'on a essayé, observé, puis renoncé. Les extraits
ci-dessous sont reproduits sans leur indentation ; le premier (`a`) est une
docstring, les autres des commentaires `#`.

### a. La contrainte de décodage, abandonnée

`rag.py`, lignes 98-102 :

```python
Anciennement, le décodage était contraint aux tokens du contexte. Avec le
nouveau jeu de réponses directes (`facts_train.txt`), cette contrainte
tronquait les réponses (« 100 » au lieu de « 100 °C ») et pouvait provoquer
une abstention à tort. La génération déterministe donne désormais la
réponse complète ; `is_supported()` conserve la garantie d'ancrage.
```

**Ce que cela raconte.** La première idée était élégante : n'autoriser que
les tokens présents dans le contexte, puisque la réponse doit être un extrait.
Le test a montré un effet de bord — la réponse s'arrêtait trop tôt. La
garantie a été déplacée **après** la génération (`is_supported()`) plutôt
qu'**pendant**. C'est une leçon de conception : une contrainte stricte peut
réduire la qualité d'une sortie qu'elle devait protéger.

### b. L'anti-répétition qui cassait la beam search

`rag.py`, lignes 109-110 :

```python
# `no_repeat_ngram_size=3`, utile en échantillonnage libre, casse la beam
# search sur ce modèle (réponses du type « convertir la mitre des lumières »).
```

**Ce que cela raconte.** Un réglage bénéfique dans un mode de génération
(`do_sample=True`) devient nuisible dans un autre (`num_beams=5`). Le code
fixe donc explicitement `no_repeat_ngram_size=0` dans le chemin RAG. Noter
l'exemple de sortie absurde dans le commentaire est très utile : il transforme
un réglage abstrait en souvenir concret.

### c. L'import paresseux de `scikit-learn`

`rag.py`, lignes 38-39 :

```python
# Import paresseux : test.py / serve.py restent utilisables sans RAG
# même si scikit-learn n'est pas installé.
```

**Ce que cela raconte.** Une dépendance a été déplacée à l'intérieur de la
classe pour ne pas bloquer les usages qui n'en ont pas besoin. C'est le type
de correction qu'on obtient après avoir vu un `ModuleNotFoundError` chez un
utilisateur — ou dans son propre environnement.

### d. Le tokenizer local prioritaire

`train.py`, lignes 132-133 :

```python
# Le tokenizer local est prioritaire : même vocabulaire que les poids
# existants, et aucune requête réseau (les erreurs TLS sont fréquentes).
```

**Ce que cela raconte.** Deux problèmes à la fois : une cohérence de
vocabulaire et des erreurs réseau intermittentes. La solution tient en trois
lignes, mais elle a nécessité d'observer les deux symptômes.

### e. L'abstention mal placée

`prepare_rag_dataset.py`, lignes 65-69 :

```python
# Peu d'exemples de refus : uniquement pour des questions hors base.
# Les anciennes versions utilisaient items[:5] (des questions couvertes),
# ce qui apprenait au modèle à refuser des questions auxquelles il sait
# répondre. Les questions hors base sont aussi gérées par is_supported()
# dans rag.py à l'inférence.
```

**Ce que cela raconte.** Le premier jeu d'abstention prenait les cinq
premières questions *de la base*. Résultat : le modèle apprenait à refuser
des questions auxquelles il devait répondre. La correction consiste à choisir
des questions **hors base**. C'est probablement l'erreur la plus instructive
du projet : le bug n'était pas dans le code, il était dans la *sémantique des
données*.

------------------------------------------------------------------------

## 3.4 Ce que l'IA a fait, ce que le développeur a fait

| Étape | IA (probable, à confirmer) | Développeur (attesté par les artefacts) |
| --- | --- | --- |
| Architecture du pipeline | Proposer une organisation en scripts | Décider de l'ordre d'exécution et du format des fichiers |
| Écriture des scripts | Générer ou modifier Python | Relire, exécuter, corriger, documenter les renoncements |
| Documentation | Rédiger les fiches `doc/` | Vérifier chaque chiffre, régénérer après changement de stratégie |
| Erreurs | Analyser des messages (TLS, `sklearn`, boucles) | Confronter au comportement réel et choisir la correction |
| Évaluation | Imaginer des métriques | Décider de ce qui compte : exactitude, ancrage, abstention |

Le tableau est volontairement prudent dans sa colonne de gauche : sans les
prompts, on ne peut pas affirmer qui a écrit quelle phrase. Ce que l'on peut
affirmer, c'est le **travail de vérification** qui a suivi — et il est
largement visible.

------------------------------------------------------------------------

## 3.5 Modèle de fiche d'échange

Pour compléter ce chapitre sans rien inventer, recopier le modèle suivant
pour chaque échange dont tu te souviens :

```markdown
### Échange n° — à compléter

- **Date / commit :** 20 ou 21 novembre 2026 / ……………
- **Objectif :** ce que je voulais obtenir
- **Prompt exact :** (copier-coller depuis l'historique si possible)
- **Réponse de l'IA (extrait utile) :** …
- **Fichier(s) généré(s) ou modifié(s) :** …
- **Vérification effectuée :** commande lancée + résultat observé
- **Erreur rencontrée :** message exact ou comportement bizarre
- **Décision retenue :** garder / modifier / abandonner, et pourquoi
- **Leçon :** une phrase que tu pourrais redire à un lecteur
```

Les sept emplacements à remplir ont été signalés dans la section 3.2. Il n'est
pas nécessaire d'être exhaustif : trois échanges bien documentés valent mieux
que vingt reconstitués.

------------------------------------------------------------------------

## 3.6 Ce qu'il faut retenir

- L'historique Git donne la **chronologie des décisions**, pas celle des
  conversations. Les deux ne doivent pas être confondues.
- Les commits racontent une progression nette : stabiliser le décodage →
  ancrer → se positionner → simplifier l'outillage → changer d'échelle →
  fiabiliser l'ancrage.
- Les commentaires de code sont des **traces d'expériences** : ils notent des
  solutions abandonnées avec l'exemple de sortie qui a échoué.
- Le développeur reste responsable des objectifs, des tests et des
  arbitrages ; l'IA est un partenaire de production, pas une source de
  vérité sur l'historique du projet.

**Suite :** chapitre 4 (à écrire) — installer l'environnement et retrouver,
sur sa propre machine, les versions et les fichiers décrits ici.
