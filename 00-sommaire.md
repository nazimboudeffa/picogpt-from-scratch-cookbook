# Sommaire et notes éditoriales

*Version de travail — sources vérifiées sur le dépôt `picogpt1-daisy` à l'état
du commit `320a572` (21 novembre 2026, 01 h 25).*

> **Édition figée.** Le dépôt source a été renommé
> [`picogpt-kit`](https://github.com/nazimboudeffa/picogpt-kit) le 21 novembre
> à 03 h 45 (`f24cc33`) et compte désormais seize commits, avec le modèle
> `PicoGPT1-Iris` 1.1.0. Les corrections apportées à ce manuscrit visent
> uniquement à rendre chaque affirmation exacte pour `320a572` ; elles ne
> changent pas le périmètre du récit. L'état courant (`71e30d9`) est
> volontairement hors sujet.

------------------------------------------------------------------------

## 1. Pourquoi ce sommaire a été corrigé

Le brouillon initial (`README.md` de ce dépôt) n'est pas faux : il décrit
fidèlement une **étape ancienne** du projet. Il correspond aux chiffres
encore affichés dans le `README.md` de `picogpt1-daisy` avant le 21 novembre
2026, c'est-à-dire à l'époque où le modèle était entraîné **de zéro** avec une
configuration 2 couches / 128 dimensions / 2 têtes. L'option `--init scratch`
n'existe pas encore à ce stade : elle sera ajoutée par `12c8df5` pour rejouer
une comparaison. Depuis, deux commits ont changé la donne :

- `12c8df5` — « Passer a GPT-2 pre-entraine, ajouter le support GPU et
  regenerer la documentation » ;
- `320a572` — « Ajouter les reponses directes et fiabiliser le RAG ».

Le livre garde le fil du brouillon, mais raconte l'état actuel. Chaque chiffre
est désormais associé à sa source vérifiable.

| Élément | Brouillon (première époque) | État actuel (`320a572`) | Où vérifier |
| --- | --- | --- | --- |
| Initialisation | Poids aléatoires, « depuis zéro » | GPT-2 **pré-entraîné** (`--init pretrained` par défaut) | `train.py:103-106,159-167` |
| Paramètres | ≈ 6,86 millions (2 couches / 128 dims) | **124 439 808** | brouillon : `MODEL_CARD.md` @ `1c00747` ; actuel : `doc/09-architecture.md`, `README.md` §13 |
| Exemples | 745 (500 Alpaca + 245 ancrés) | **2 442** (2 000 + 245 + 197) | `README.md` §6, `doc/03-dataset.md` |
| Jeux de données | `train.txt`, `rag_train.txt` | + `facts_train.txt` | `train.py:27-31` |
| Époques | 10 | 5 pour l'entraînement de référence (3 par défaut en fine-tuning, 10 en `scratch`) | `doc/04-entrainement.md` |
| Étapes | ≈ 1 860 | ≈ 3 050 | `doc/04-entrainement.md` |
| Loss | ≈ 4,54 (finale) | 1,58 (moyenne), ≈ 1,05 (dernière journalisée) | `doc/04-entrainement.md` |
| Matériel | CPU | CUDA + `--fp16`, 250,8 s sur RTX 3070 | `doc/04-entrainement.md` |
| Notebooks | `train.ipynb`, `test.ipynb` | Supprimés au commit `66e4623` ; scripts Python directs | `git ls-tree HEAD` |
| Dépendance RAG | non mentionnée | `scikit-learn` requise | `rag.py:38-41`, `README.md` §3 |

**Nuance importante.** Le mode « depuis zéro » n'a pas disparu : il reste
disponible avec `python train.py --init scratch`. Le titre du livre garde donc
tout son sens — il désigne une **démarche d'apprentissage**, pas
l'initialisation des poids du dernier entraînement. Les deux époques sont
racontées dans le [chapitre 1](01-une-idee-apprendre-en-construisant.md) et
le chapitre 8 (à écrire).

### La généalogie : des ressources d'apprentissage aux premiers dépôts

Le brouillon ne disait presque rien de l'origine. Elle est pourtant essentielle
pour comprendre le nom « PicoGPT1 » et le rôle de l'IA. **L'auteur n'a pas lu le
livre de Raschka** : il a étudié le **code open source** du dépôt associé, avec
l'aide de ChatGPT. À cela s'ajoutent **beaucoup d'articles** — dont il ne se
souvient plus de tous les titres — et, **surtout, des tutoriels YouTube**. La
chronologie vérifiée des dépôts est la suivante :

| Date | Dépôt / ressource | Ce qui est appris ou construit | Lien |
| --- | --- | --- | --- |
| source | Code `rasbt/LLMs-from-scratch` (Apache-2.0) | Référence structurante : le code étudié et compris avec ChatGPT, sans lecture du livre | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
| source | ChatGPT | Aide à la compréhension du code dès 2025, puis à la production à partir d'avril 2026 (chapitre 3) | — |
| source | Articles (références en partie oubliées) et tutoriels YouTube | L'autre versant — et le plus présent — de l'apprentissage : voir les mécanismes expliqués et le code se construire | [annexe G](annexe-g-sources-et-references/annexe-g-sources-et-references.md) pour les références retrouvées ; le reste à compléter |
| juin 2025 | `picogpt-python` | Première implémentation : attention multi-têtes, `LayerNorm`, `FeedForward` et `GPTModel` écrits à la main, tokenizer `tiktoken`, phrases anglaises simples, notebooks | [nazimboudeffa/picogpt-python](https://github.com/nazimboudeffa/picogpt-python) |
| avril 2026 | `picogpt1-scratch-python` | Avec l'IA : premier modèle entraîné sur Alpaca, `Trainer` Hugging Face, 2 couches / 128 dimensions, conversion GGUF et Ollama | [nazimboudeffa/picogpt1-scratch-python](https://github.com/nazimboudeffa/picogpt1-scratch-python) |
| 20 nov. 2026 | `picogpt1-jasmine` | Passage aux scripts Python, publication Hugging Face, documentation complète | [nazimboudeffa/picogpt1-jasmine](https://github.com/nazimboudeffa/picogpt1-jasmine) |
| 20-21 nov. 2026 | `picogpt1-daisy` | RAG, abstention, décodage anti-boucles, puis GPT-2 pré-entraîné 124 M | [nazimboudeffa/picogpt1-daisy](https://github.com/nazimboudeffa/picogpt1-daisy) |

**Décision éditoriale proposée.** « PicoGPT1 » n'est pas un numéro de version
au sens strict : c'est le nom du **premier modèle entraîné**, par opposition
au PicoGPT de 2025 qui servait à apprendre l'architecture. La page du dépôt
d'avril 2026 l'écrit elle-même : *« After PicoGPT basic for learning purpose,
here is a first model, so no need to start from scratch now and after »*.
Le chapitre 2 propose une formulation pour le livre ; elle reste à confirmer
par l'auteur.

------------------------------------------------------------------------

## 2. La colonne vertébrale : six commits

L'historique Git est la source la plus fiable pour dater les étapes. Il ne
contient pas les prompts échangés avec l'IA, mais il prouve les changements,
les ordres et les reprises.

| Date | Commit | Message | Changement visible | Chapitre |
| --- | --- | --- | --- | --- |
| 20 nov. 2026 | `b5f05c6` | Renommage PicoGPT1-Daisy et decodage anti-boucles | État initial versionné : 31 fichiers, scripts + `doc/` + notebooks | 1, 3, 9 |
| 20 nov. 2026 | `9884277` | Ajout du pipeline RAG et reduction des hallucinations | +947/−142 lignes ; `rag.py`, `evaluate.py`, `knowledge.jsonl`, `doc/12-rag.md` | 10, 11, 3 |
| 20 nov. 2026 | `1c00747` | Documenter les differences avec PicoGPT1-Jasmine | `MODEL_CARD.md` +19 lignes : tableau comparatif | 2 |
| 20 nov. 2026 | `66e4623` | Supprimer les notebooks obsolètes et nettoyer la documentation | −590 lignes (notebooks retirés) | 3, 5 |
| 21 nov. 2026 | `12c8df5` | Passer a GPT-2 pre-entraine, ajouter le support GPU et regenerer la documentation | 17 fichiers, +328/−167 ; bascule `pretrained`, `--fp16`, doc régénérée | 8, 3 |
| 21 nov. 2026 | `320a572` | Ajouter les reponses directes et fiabiliser le RAG | 16 fichiers, +248/−106 ; `prepare_facts_dataset.py`, contrainte de décodage abandonnée | 10, 3 |

------------------------------------------------------------------------

## 3. Table des matières détaillée

### [Chapitre 1 — Une idée : apprendre en construisant](01-une-idee-apprendre-en-construisant.md) ✅ rédigé

- La boîte noire des grands modèles et le choix d'un laboratoire réduit.
- **L'origine réelle : le code open source de `rasbt/LLMs-from-scratch`, étudié
  avec ChatGPT (le livre n'a pas été lu), de nombreux articles et surtout des
  tutoriels YouTube, puis les dépôts `picogpt-python` (juin 2025) et
  `picogpt1-scratch-python` (avril 2026).**
- La chaîne complète, du texte brut au chat navigateur : neuf étapes, treize
  fichiers, une question par étape.
- Les deux époques du projet : 6,86 M pour la première architecture (2 couches
  / 128 dims), puis 124 M par ajustement ; ce que cela change dans le récit.
  Le mode `scratch` conservé dans Daisy utilise une autre configuration
  (4 couches / 256 dims, ≈ 16,1 M) et sert à la comparaison.
- Le parcours du lecteur et les premières commandes de vérification.

### [Chapitre 2 — Une inspiration pédagogique](02-une-inspiration-pedagogique.md) ✅ rédigé

- Ce qu'apporte le **code** de `rasbt/LLMs-from-scratch` (pas le livre), avec
  l'aide de ChatGPT, ainsi que les articles lus et les tutoriels YouTube — et
  ce que le projet ne reprend pas.
- La note sur la licence Apache-2.0 et l'attribution à régulariser dans
  `picogpt-python`.
- La généalogie complète en cinq dépôts, de l'attention reconstruite en 2025 au
  `Trainer` de 2026 ; ce que signifie « PicoGPT1 ».
- La filiation `PicoGPT1-Jasmine` → `PicoGPT1-Daisy` documentée dans le
  commit `1c00747`.
- Matrice « inspiré / adapté / propre au projet » reliée aux lignes de code.
- Premier extrait commenté : `build_example()` et le masquage des labels.

### [Chapitre 3 — Programmer avec l'aide de l'IA](03-programmer-avec-l-aide-de-l-ia.md) ✅ rédigé

- Méthode : définir un objectif, examiner une proposition, tester, corriger.
- ChatGPT comme aide à la compréhension du code dès 2025, puis comme partenaire
  de production à partir d'avril 2026.
- OpenCode avec le modèle **Big Pickle**, pour ce manuscrit et le développement
  courant de `picogpt-kit` (novembre 2026).
- Les six commits racontés comme une suite de décisions.
- Les « traces de décisions » laissées dans les commentaires du code
  (contrainte abandonnée, beam search, erreurs TLS, etc.).
- Modèle de fiche d'échange et zones explicitement laissées à compléter.

### [Chapitre 4 — Atelier : de zéro à une première réponse](04-atelier-de-zero-a-une-reponse.md) ✅ rédigé

- Un GPT miniature écrit à la main : tokenizer de caractères, lots décalés,
  attention causale, deux blocs, `AdamW`, génération.
- Six blocs de code à recopier dans un seul `mon_gpt.py` ; sortie vérifiée
  (loss 3,287 → 0,064, Python 3.13.2 / `torch` 2.7.1).
- Code **pédagogique écrit pour le livre**, explicitement distinct du code de
  `picogpt1-daisy` (convention n° 1 ci-dessous).
- Tableau de correspondance avec le projet réel : 106 264 paramètres contre
  124 439 808, 24 caractères contre 50 257 tokens BPE.
- Pourquoi un modèle sur cinq phrases mémorise et ne répond pas : ouverture
  vers le préentraînement, le `Trainer` et le RAG.

### Chapitre 5 — Installer et reproduire l'environnement ✍️ à écrire

- Python 3.10+, testé en 3.13 sur Windows 11 ; `torch`, `transformers`,
  `datasets`, `huggingface_hub`, `scikit-learn`.
- Environnement virtuel, vérification par `python -m py_compile`, versions.
- Pourquoi les poids ne sont pas versionnés et les deux façons de repartir.
- Sources : `doc/02-installation.md`, `README.md` §3-4, `.gitignore`.

### Chapitre 6 — Préparer les données : trois scripts, trois jeux ✍️ à écrire

- `prepare_dataset.py` : Alpaca, filtres, déduplication, `BAD_PATTERNS`.
- `knowledge.jsonl` : les 30 faits, le format `id`/`text`/`question`/`answer`.
- `prepare_rag_dataset.py` : 8 variantes ancrées + 5 abstentions = 245.
- `prepare_facts_dataset.py` : variantes mécaniques, alias, calculs = 197.
- Format de bloc deux lignes et pourquoi le parseur est aussi strict.
- Sources : les trois scripts, `doc/03-dataset.md`, `doc/12-rag.md`.

### Chapitre 7 — Le format appris : tokenisation et masquage ✍️ à écrire

- `build_example()` ligne à ligne : EOS, PAD, troncature à 256.
- Pourquoi les tokens du prompt reçoivent `-100`.
- Le `chat_template` sauvegardé et réutilisé dans le GGUF.
- Sources : `train.py:53-66,138-148`, `doc/09-architecture.md`.

### Chapitre 8 — Entraîner : deux époques du projet ✍️ à écrire

- Mode `scratch` (4 couches / 256 dims / 4 têtes) versus mode `pretrained`.
- Les options CLI, le `Trainer`, les checkpoints, la durée observée.
- Comparer les deux jeux de résultats : 4,54 contre 1,05.
- Sources : `train.py`, `doc/04-entrainement.md`, commit `12c8df5`.

### Chapitre 9 — Générer : le modèle seul ✍️ à écrire

- `test.py`, `load_model()`, `generate()` pas à pas.
- Le `\n` garanti, `StopAtNewline`, échantillonnage et anti-boucles.
- Pièges : `generation_config.max_length`, décodage glouton, PowerShell.
- Sources : `model_utils.py`, `test.py`, `doc/05-generation.md`.

### Chapitre 10 — Ancrer : le pipeline RAG ✍️ à écrire

- TF-IDF sur caractères (`char_wb`, 3-5), indexation de `text + question`.
- Seuil `DEFAULT_MIN_SCORE = 0.50`, construction du prompt, `is_supported()`.
- Pourquoi le décodage contraint a été retiré au dernier commit.
- Ce que le GGUF contient et ne contient pas.
- Sources : `rag.py`, `prepare_rag_dataset.py`, `doc/12-rag.md`, commit `320a572`.

### Chapitre 11 — Évaluer sans se tromper soi-même ✍️ à écrire

- `evaluate.py` : exactitude, ancrage, abstentions, retrieval.
- Les résultats annoncés (30/30, 10/10) et leurs limites.
- Pourquoi il n'y a pas de jeu de validation.
- Sources : `evaluate.py`, `doc/12-rag.md`, `doc/11-limites.md`.

### Chapitre 12 — Servir, exporter, publier ✍️ à écrire

- `serve.py` : API locale avec la bibliothèque standard, `POST /generate`.
- `export_gguf.py` : convertisseur épinglé `b4000`, sortie F16.
- `publish_hf.py` : retries TLS, `MODEL_CARD.md` renommé `README.md`.
- llama.app et le piège de la régénération du GGUF après réentraînement.
- Sources : les trois scripts, `doc/06`, `07`, `08`, `MODEL_CARD.md`.

### Chapitre 13 — Ce que le projet montre et ce qu'il ne montre pas ✍️ à écrire

- Forces : chaîne complète reproductible, abstention vérifiable.
- Limites : base fermée, 256 tokens, pas de validation, GGUF sans RAG.
- Pistes : jeu de validation, quantification `Q4_K_M`, extension de la base.
- Sources : `doc/11-limites.md`, `README.md` §15, `MODEL_CARD.md`.

### Annexes

- [A. `mon_gpt.py` complet et commenté (chapitre 4)](annexe-a-mon-gpt-complet/annexe-a-mon-gpt-complet.md) ✅ rédigé
- [B. Petit chat avec `mon_gpt`](annexe-b-petit-chat/annexe-b-petit-chat.md) ✅ rédigé
- C. Carte complète du dépôt (fichiers versionnés / générés) ✍️ à écrire.
- D. Chronologie Git commentée et commandes pour la reconstituer ✍️ à écrire.
- E. Glossaire : token, EOS, label `-100`, TF-IDF, RAG, GGUF, beam search ✍️ à écrire.
- F. Scripts complets annotés (`train.py`, `rag.py`, `model_utils.py`) ✍️ à écrire.
- [G. Sources et références](annexe-g-sources-et-references/annexe-g-sources-et-references.md) ✅ rédigé.

------------------------------------------------------------------------

## 4. Conventions de rédaction

1. **Le code est réel.** Les extraits sont repris des fichiers de
   `picogpt1-daisy` ; les numéros de lignes indiqués permettent de les
   retrouver. Aucun code « d'illustration » n'est présenté comme du code du
   projet. Seule exception explicitement signalée : le chapitre 4, où le
   code pédagogique écrit pour le livre est présenté comme tel, sans jamais
   être attribué au dépôt.
2. **Les commandes sont exécutables telles quelles.** Quand une nuance
   Windows existe (PowerShell et le `\n` des prompts), elle est signalée
   dans un encadré.
3. **Les chiffres sont sourcés.** Un nombre avancé dans le récit renvoie à un
   fichier ou à un commit. Les valeurs du brouillon sont conservées comme
   témoignage de la première époque, jamais comme valeurs courantes.
4. **Rien n'est inventé sur la collaboration IA.** Les commits ne contiennent
   pas les prompts. Les zones `> **À compléter —**` indiquent précisément ce
   qui doit venir de notes ou de souvenirs personnels.
5. **Chaque chapitre se termine par un récapitulatif** et une ouverture vers
   le suivant.

------------------------------------------------------------------------

## 5. Correspondance avec le brouillon initial

| Brouillon | Nouveau livre |
| --- | --- |
| Avant-propos | Chapitre 1 + notes éditoriales ci-dessus |
| Ch. 1 « Une idée… » | Chapitre 1 |
| Ch. 2 « Une inspiration… » | Chapitre 2 |
| Ch. 3 « Programmer avec l'IA » | Chapitre 3 |
| — (ajout pour le livre) | Chapitre 4 (atelier from scratch) |
| Ch. 4 « Préparer les données » | Chapitres 6 et 7 |
| Ch. 5 « Entraîner un mini GPT » | Chapitre 8 |
| Ch. 6 « Relier à une base… » | Chapitres 10 et 11 |
| Ch. 7 « Passer du modèle à l'outil » | Chapitre 12 |
| Ch. 8 « Ce que le projet montre… » | Chapitre 13 |
| — (ajout pour le livre) | Annexe A (script de l'atelier) |
| — (ajout pour le livre) | Annexe B (petit chat avec `mon_gpt`) |
| — (ajout pour le livre) | Annexe G (sources et références) |
| Annexe « Carte du dépôt » | Annexe C |

------------------------------------------------------------------------

## 6. État d'avancement

- [x] Sommaire corrigé et vérifié sur les sources
- [x] Généalogie des cinq dépôts vérifiée (2025-2026)
- [x] Chapitres 1, 2 et 3 (avec l'origine du projet)
- [x] Chapitre 4 (atelier : code pédagogique vérifié, six blocs exécutables)
- [x] Annexe A (script intégral commenté, exécuté et vérifié)
- [x] Annexe B (petit chat interactif avec `mon_gpt`, exécuté et vérifié)
- [x] Annexe G (sources et références : Vaswani, PyTorch, Karpathy, Raschka)
- [ ] Chapitres 5 à 13 et annexes C à F
- [x] Relecture des extraits de code par rapport aux fichiers sources
- [x] Édition figée sur `320a572` : corrections factuelles appliquées le 22 novembre 2026
- [ ] Ajout des prompts réels dans le chapitre 3 et des souvenirs de development
  (les références retrouvées sont désormais dans l'annexe G)
- [ ] Figer une édition destination publication
