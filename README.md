# De LLMs from Scratch à PicoGPT1-Daisy

## Construire un mini modèle de langage avec l'aide de l'intelligence artificielle

*Carnet de projet --- première édition de travail*

------------------------------------------------------------------------

## Avant-propos

Ce petit livre raconte la naissance et la construction de
**PicoGPT1-Daisy**, un projet expérimental autour des modèles de
langage. L'idée n'est pas de prétendre avoir créé un grand modèle
comparable aux systèmes industriels. Il s'agit plutôt de comprendre, en
construisant, les grandes étapes qui permettent de passer de données
textuelles à un petit modèle capable de générer du texte et de dialoguer
dans un environnement simple.

Le projet s'inscrit dans une démarche d'apprentissage inspirée notamment
du travail pédagogique de **Sebastian Raschka** et de son dépôt [LLMs
from Scratch](https://github.com/rasbt/LLMs-from-scratch). Cette
référence aide à aborder les principes d'un modèle de langage construit
progressivement. PicoGPT1-Daisy constitue toutefois une réalisation
personnelle, avec ses propres choix, ses scripts, ses données et ses
expériences.

Un autre élément important de cette aventure est l'utilisation de
l'intelligence artificielle comme partenaire de développement. L'IA a
participé au travail de programmation et de réflexion. Le rôle du
développeur reste essentiel : définir les objectifs, guider les
échanges, examiner les propositions, lancer les tests, constater les
erreurs et décider des corrections à apporter. Les détails précis de
cette collaboration seront enrichis au fil des prochaines étapes.

## Le projet en quelques mots

PicoGPT1-Daisy est présenté comme un mini GPT-2 entraîné de zéro, depuis
la préparation des données jusqu'à une interface de discussion
utilisable dans un navigateur. Le dépôt rassemble plusieurs étapes :
préparation d'un jeu de données, constitution d'une petite base de
connaissances, préparation d'exemples pour le RAG, entraînement du
modèle, génération de texte, évaluation, mise à disposition via un
serveur HTTP local et pistes d'export ou de publication.

-   **Dépôt du projet :**
    <https://github.com/nazimboudeffa/picogpt1-daisy>
-   **Référence pédagogique :**
    <https://github.com/rasbt/LLMs-from-scratch>

## Chapitre 1 --- Une idée : apprendre en construisant

Les grands modèles de langage peuvent donner l'impression d'être des
boîtes noires. On leur pose une question et ils produisent une réponse,
mais les mécanismes qui rendent cela possible restent souvent
invisibles. Pour mieux comprendre ces systèmes, une approche consiste à
reconstruire une version réduite de leur fonctionnement.

C'est l'esprit de PicoGPT1-Daisy : réduire l'échelle sans abandonner les
étapes importantes. Même un petit modèle oblige à rencontrer les
questions fondamentales : comment représenter le texte, comment préparer
les exemples, comment organiser l'apprentissage, comment mesurer les
résultats et comment transformer un modèle entraîné en outil utilisable.

Le projet ne vise donc pas seulement à obtenir un chatbot. Il sert aussi
de laboratoire. Chaque script représente une étape du parcours et permet
d'isoler une partie du problème.

## Chapitre 2 --- Une inspiration pédagogique

Le dépôt **LLMs from Scratch** de Sebastian Raschka accompagne le
lecteur dans la construction progressive d'un modèle de langage de type
GPT. Il aborde notamment la préparation du texte, la tokenisation, les
mécanismes d'attention, l'architecture GPT, le préentraînement et
l'ajustement sur des tâches particulières.

Cette démarche est une source d'inspiration pour PicoGPT1-Daisy :
apprendre en suivant les étapes qui composent un modèle plutôt que de se
limiter à utiliser un modèle déjà prêt. Il faut néanmoins distinguer une
inspiration conceptuelle d'une reproduction exacte. Les choix
d'implémentation, l'organisation du dépôt, les données et les
expérimentations propres à PicoGPT1-Daisy doivent être décrits à partir
du code et de l'expérience réelle du projet.

L'un des objectifs de ce livre sera justement de préciser, étape par
étape, ce qui vient de la lecture et de l'inspiration, ce qui a été
adapté, et ce qui a été imaginé ou développé spécifiquement pour
PicoGPT1-Daisy.

## Chapitre 3 --- Programmer avec l'aide de l'IA

PicoGPT1-Daisy a été développé avec l'aide d'outils d'intelligence
artificielle. Cette collaboration fait partie intégrante de l'histoire
du projet. Elle ne signifie pas que le développement s'est déroulé sans
intervention humaine : un projet logiciel demande de choisir une
direction, de comprendre les besoins, de vérifier le comportement du
code et de corriger ce qui ne fonctionne pas.

Dans ce type de démarche, l'IA peut aider à proposer une architecture,
écrire ou modifier des fichiers Python, expliquer une fonction, suggérer
des commandes, analyser un message d'erreur ou imaginer des tests. Le
développeur doit ensuite confronter ces propositions à la réalité :
exécuter le programme, observer les résultats, vérifier les hypothèses
et décider de la suite.

Le récit détaillé de cette collaboration sera complété avec les exemples
réels de prompts, les fichiers générés ou modifiés, les erreurs
rencontrées et les décisions prises. Il est important de ne pas
reconstituer artificiellement une chronologie : les étapes seront
racontées à partir des commits, des notes et des souvenirs du
développement.

## Chapitre 4 --- Préparer les données

Un modèle de langage apprend à partir d'exemples. La préparation des
données est donc une étape déterminante. Le dépôt PicoGPT1-Daisy décrit
une préparation d'un jeu de données de type Alpaca, avec filtrage, ainsi
que l'ajout d'exemples d'ancrage liés à une petite base de connaissances
en français.

Le projet mentionne une base de connaissances contenant trente faits en
français, stockés dans `knowledge.jsonl`. Des scripts dédiés servent à
préparer les données et à construire des exemples destinés à l'approche
RAG. Cette séparation entre les données d'entraînement et les
informations consultables constitue un point important à expliquer dans
la version détaillée du livre.

La qualité des données compte autant que leur quantité. Un petit corpus
peut être utile pour une expérience pédagogique, mais il limite
nécessairement la diversité des formulations et les capacités générales
du modèle. Les résultats doivent donc être interprétés à l'échelle de
l'expérience.

## Chapitre 5 --- Entraîner un mini GPT

Le cœur du projet est un modèle de langage de petite taille, présenté
dans le dépôt comme une architecture de type GPT-2. Les chiffres
indiqués dans la documentation actuelle donnent un ordre de grandeur :
environ **6,86 millions de paramètres**, **745 exemples
d'entraînement**, **10 époques** et environ **1 860 étapes**, pour une
loss finale annoncée autour de **4,54**.

Ces chiffres décrivent une expérience précise et ne constituent pas, à
eux seuls, une preuve de qualité conversationnelle. La loss permet de
suivre un aspect de l'apprentissage, mais elle ne suffit pas à
déterminer si les réponses sont pertinentes, fiables ou utiles. C'est
pourquoi le projet comporte également des tests et une évaluation
orientée vers la base de connaissances.

Le script `train.py` joue un rôle central dans cette phase. Le dépôt
contient également des notebooks destinés à accompagner l'entraînement
et les essais. Dans la version enrichie, nous décrirons les paramètres
réellement utilisés, les difficultés d'exécution, les ajustements
successifs et les observations faites pendant l'apprentissage.

## Chapitre 6 --- Relier le modèle à une base de connaissances

Un modèle entraîné sur un petit corpus ne peut pas mémoriser
parfaitement toutes les informations et répondre correctement à toutes
les questions. Le projet explore donc une approche de **génération
augmentée par récupération**, appelée **RAG** (*Retrieval-Augmented
Generation*). Le principe général consiste à rechercher des informations
pertinentes dans une source externe, puis à les fournir au modèle comme
contexte pour produire une réponse.

Dans PicoGPT1-Daisy, la base de connaissances est volontairement réduite
et structurée. Cette simplicité permet de tester un comportement
essentiel : répondre lorsque l'information est présente et éviter
d'inventer une réponse lorsque la base ne contient pas l'information
demandée.

La documentation du dépôt annonce une évaluation de **30 réponses
ancrées sur 30** et **10 abstentions correctes sur 10** questions hors
base. Ces résultats sont ceux rapportés par le projet ; ils devront être
accompagnés, dans la version détaillée, des questions de test, de la
méthode d'évaluation et de ses limites.

## Chapitre 7 --- Passer du modèle au petit outil

Un modèle devient plus intéressant lorsqu'il peut être utilisé
facilement. PicoGPT1-Daisy prévoit une génération en ligne de commande
et un serveur HTTP local permettant d'interagir avec le modèle depuis un
navigateur. Cette étape transforme les scripts d'expérimentation en une
petite application accessible.

Le dépôt comprend notamment des outils pour charger le modèle, générer
du texte et démarrer le serveur. Il mentionne également un export au
format **GGUF** et une démarche de publication sur **Hugging Face**,
suivie d'essais avec **llama.app**. Ces étapes ouvrent la voie à des
tests dans d'autres environnements, mais elles nécessitent chacune des
vérifications de compatibilité et de qualité.

## Chapitre 8 --- Ce que le projet montre, et ce qu'il ne montre pas

PicoGPT1-Daisy montre qu'il est possible d'assembler, à petite échelle,
plusieurs briques importantes d'un projet de modèle de langage :
données, entraînement, génération, récupération d'informations,
évaluation et interface. Il constitue un support concret pour apprendre
et expérimenter.

Il ne faut toutefois pas confondre démonstration technique et modèle
généraliste. Avec quelques millions de paramètres et un corpus limité,
les capacités restent modestes. Le modèle peut produire des réponses
incohérentes, mal suivre une instruction ou manquer de connaissances. Le
RAG améliore l'accès à des informations ciblées, mais dépend de la
qualité de la base et du mécanisme de recherche.

Reconnaître ces limites n'enlève rien à l'intérêt du projet. Au
contraire, elles permettent de mieux comprendre les exigences d'un
système plus robuste : davantage de données, des évaluations plus
variées, une meilleure mesure des erreurs, une documentation
reproductible et des tests indépendants.

## Conclusion --- Un projet qui reste ouvert

PicoGPT1-Daisy est à la fois un résultat logiciel et un parcours
d'apprentissage. L'inspiration de **LLMs from Scratch** a donné un cadre
pour explorer les mécanismes des modèles de langage. L'assistance de
l'IA a contribué au travail de programmation. Les choix, les essais et
les vérifications ont permis de transformer cette exploration en un
dépôt organisé autour d'un mini GPT expérimental.

Ce livre est une première base. Il ne cherche pas encore à figer chaque
détail historique. Il pose les grandes étapes et réserve une place aux
explications techniques, aux commandes, aux extraits de code, aux
prompts et aux difficultés réellement rencontrées. La suite consistera à
reprendre le développement dans l'ordre, à vérifier chaque étape et à
raconter ce qui s'est effectivement passé.

## Annexe --- Carte du dépôt

  -----------------------------------------------------------------------
  Fichier                             Rôle indiqué
  ----------------------------------- -----------------------------------
  `prepare_dataset.py`                Préparation et filtrage du jeu de
                                      données

  `prepare_rag_dataset.py`            Préparation des exemples liés à
                                      l'approche RAG

  `train.py`                          Entraînement du modèle

  `rag.py`                            Fonctions associées à la
                                      récupération d'informations

  `evaluate.py`                       Évaluation des réponses et du
                                      comportement d'abstention

  `model_utils.py`                    Utilitaires autour du modèle

  `test.py`                           Essais et génération

  `serve.py`                          Serveur HTTP local

  `export_gguf.py`                    Export vers le format GGUF

  `publish_hf.py`                     Outils de publication sur Hugging
                                      Face

  `train.ipynb` / `test.ipynb`        Notebooks d'entraînement et de test

  `MODEL_CARD.md` et `doc/`           Documentation du modèle et
                                      documents du projet
  -----------------------------------------------------------------------

## Sources et références

-   Projet PicoGPT1-Daisy :
    <https://github.com/nazimboudeffa/picogpt1-daisy>
-   Sebastian Raschka, *LLMs from Scratch* :
    <https://github.com/rasbt/LLMs-from-scratch>

Les descriptions et chiffres techniques de cette première version sont
repris de la documentation du dépôt du projet. Ils devront être
revérifiés au moment de détailler les étapes et de figer une édition
destinée à la publication.
