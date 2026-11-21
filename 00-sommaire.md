# Sommaire et notes éditoriales

*Version de travail — sources vérifiées sur le dépôt `picogpt1-daisy` à l'état
du commit `320a572` (21 novembre 2026).*

------------------------------------------------------------------------

## 1. Pourquoi ce sommaire a été corrigé

Le brouillon initial (`README.md` de ce dépôt) n'est pas faux : il décrit
fidèlement une **étape ancienne** du projet. Il correspond aux chiffres
encore affichés dans le `README.md` de `picogpt1-daisy` avant le 21 novembre
2026, c'est-à-dire à l'époque où le modèle était entraîné **de zéro**
(`--init scratch`). Depuis, deux commits ont changé la donne :

- `12c8df5` — « Passer a GPT-2 pre-entraine, ajouter le support GPU et
  regenerer la documentation » ;
- `320a572` — « Ajouter les reponses directes et fiabiliser le RAG ».

Le livre garde le fil du brouillon, mais raconte l'état actuel. Chaque chiffre
est désormais associé à sa source vérifiable.

| Élément | Brouillon (première époque) | État actuel (`320a572`) | Où vérifier |
| --- | --- | --- | --- |
| Initialisation | Poids aléatoires, « depuis zéro » | GPT-2 **pré-entraîné** (`--init pretrained` par défaut) | `train.py:103-106,159-167` |
| Paramètres | ≈ 6,86 millions | **124 439 808** | `doc/09-architecture.md`, `README.md` §13 |
| Exemples | 745 (500 Alpaca + 245 ancrés) | **2 442** (2 000 + 245 + 197) | `README.md` §6, `doc/03-dataset.md` |
| Jeux de données | `train.txt`, `rag_train.txt` | + `facts_train.txt` | `train.py:27-31` |
| Époques | 10 | 5 pour l'entraînement de référence (3 par défaut en fine-tuning, 10 en `scratch`) | `doc/04-entrainement.md` |
| Étapes | ≈ 1 860 | ≈ 3 050 | `doc/04-entrainement.md` |
| Loss | ≈ 4,54 (finale) | 1,58 (moyenne), ≈ 1,05 (dernière journalisée) | `doc/04-entrainement.md` |
| Matériel | CPU | CUDA + `--fp16`, 250,8 s sur RTX 3070 | `doc/04-entrainement.md` |
| Notebooks | `train.ipynb`, `test.ipynb` | Supprimés au commit `66e4623` ; scripts Python directs | `git ls-tree HEAD` |
| Dépendance RAG | non mentionnée | `scikit-learn` requise | `rag.py:41`, `README.md` §3 |

**Nuance importante.** Le mode « depuis zéro » n'a pas disparu : il reste
disponible avec `python train.py --init scratch`. Le titre du livre garde donc
tout son sens — il désigne une **démarche d'apprentissage**, pas
l'initialisation des poids du dernier entraînement. Les deux époques sont
racontées dans le [chapitre 1](01-une-idee-apprendre-en-construisant.md) et
le chapitre 7 (à écrire).

### La généalogie : du livre aux premiers dépôts

Le brouillon ne disait presque rien de l'origine. Elle est pourtant essentielle
pour comprendre le nom « PicoGPT1 » et le rôle de l'IA. La chronologie
vérifiée est la suivante :

| Date | Dépôt | Ce qui est appris ou construit | Lien |
| --- | --- | --- | --- |
| source | *LLMs from Scratch*, S. Raschka | Le point de départ intellectuel : construire un GPT étape par étape | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
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
| 20 nov. 2026 | `b5f05c6` | Renommage PicoGPT1-Daisy et decodage anti-boucles | État initial versionné : 31 fichiers, scripts + `doc/` + notebooks | 1, 3, 8 |
| 20 nov. 2026 | `9884277` | Ajout du pipeline RAG et reduction des hallucinations | +947/−142 lignes ; `rag.py`, `evaluate.py`, `knowledge.jsonl`, `doc/12-rag.md` | 9, 10, 3 |
| 20 nov. 2026 | `1c00747` | Documenter les differences avec PicoGPT1-Jasmine | `MODEL_CARD.md` +19 lignes : tableau comparatif | 2 |
| 20 nov. 2026 | `66e4623` | Supprimer les notebooks obsolètes et nettoyer la documentation | −590 lignes (notebooks retirés) | 3, 4 |
| 21 nov. 2026 | `12c8df5` | Passer a GPT-2 pre-entraine, ajouter le support GPU et regenerer la documentation | 17 fichiers, +328/−167 ; bascule `pretrained`, `--fp16`, doc régénérée | 7, 3 |
| 21 nov. 2026 | `320a572` | Ajouter les reponses directes et fiabiliser le RAG | 16 fichiers, +248/−106 ; `prepare_facts_dataset.py`, contrainte de décodage abandonnée | 9, 3 |

------------------------------------------------------------------------

## 3. Table des matières détaillée

### [Chapitre 1 — Une idée : apprendre en construisant](01-une-idee-apprendre-en-construisant.md) ✅ rédigé

- La boîte noire des grands modèles et le choix d'un laboratoire réduit.
- **L'origine réelle : le livre de Raschka, puis les dépôts `picogpt-python`
  (juin 2025) et `picogpt1-scratch-python` (avril 2026).**
- La chaîne complète, du texte brut au chat navigateur : neuf étapes, neuf
  fichiers, une question par étape.
- Les deux époques du projet : 6,86 M paramètres puis 124 M ; ce que cela
  change dans le récit.
- Le parcours du lecteur et les premières commandes de vérification.

### [Chapitre 2 — Une inspiration pédagogique](02-une-inspiration-pedagogique.md) ✅ rédigé

- Ce que apporte *LLMs from Scratch* de Sebastian Raschka, et ce que le
  projet n'en reprend pas.
- La généalogie complète en cinq dépôts, de l'attention écrite à la main en
  2025 au `Trainer` de 2026 ; ce que signifie « PicoGPT1 ».
- La filiation `PicoGPT1-Jasmine` → `PicoGPT1-Daisy` documentée dans le
  commit `1c00747`.
- Matrice « inspiré / adapté / propre au projet » reliée aux lignes de code.
- Premier extrait commenté : `build_example()` et le masquage des labels.

### [Chapitre 3 — Programmer avec l'aide de l'IA](03-programmer-avec-l-aide-de-l-ia.md) ✅ rédigé

- Méthode : définir un objectif, examiner une proposition, tester, corriger.
- Les six commits racontés comme une suite de décisions.
- Les « traces de décisions » laissées dans les commentaires du code
  (contrainte abandonnée, beam search, erreurs TLS, etc.).
- Modèle de fiche d'échange et zones explicitement laissées à compléter.

### Chapitre 4 — Installer et reproduire l'environnement ✍️ à écrire

- Python 3.10+, testé en 3.13 sur Windows 11 ; `torch`, `transformers`,
  `datasets`, `huggingface_hub`, `scikit-learn`.
- Environnement virtuel, vérification par `python -m py_compile`, versions.
- Pourquoi les poids ne sont pas versionnés et les deux façons de repartir.
- Sources : `doc/02-installation.md`, `README.md` §3-4, `.gitignore`.

### Chapitre 5 — Préparer les données : trois scripts, trois jeux ✍️ à écrire

- `prepare_dataset.py` : Alpaca, filtres, déduplication, `BAD_PATTERNS`.
- `knowledge.jsonl` : les 30 faits, le format `id`/`text`/`question`/`answer`.
- `prepare_rag_dataset.py` : 8 variantes ancrées + 5 abstentions = 245.
- `prepare_facts_dataset.py` : variantes mécaniques, alias, calculs = 197.
- Format de bloc deux lignes et pourquoi le parseur est aussi strict.
- Sources : les trois scripts, `doc/03-dataset.md`, `doc/12-rag.md`.

### Chapitre 6 — Le format appris : tokenisation et masquage ✍️ à écrire

- `build_example()` ligne à ligne : EOS, PAD, troncature à 256.
- Pourquoi les tokens du prompt reçoivent `-100`.
- Le `chat_template` sauvegardé et réutilisé dans le GGUF.
- Sources : `train.py:53-66,138-148`, `doc/09-architecture.md`.

### Chapitre 7 — Entraîner : deux époques du projet ✍️ à écrire

- Mode `scratch` (4 couches / 256 dims / 4 têtes) versus mode `pretrained`.
- Les options CLI, le `Trainer`, les checkpoints, la durée observée.
- Comparer les deux jeux de résultats : 4,54 contre 1,05.
- Sources : `train.py`, `doc/04-entrainement.md`, commit `12c8df5`.

### Chapitre 8 — Générer : le modèle seul ✍️ à écrire

- `test.py`, `load_model()`, `generate()` pas à pas.
- Le `\n` garanti, `StopAtNewline`, échantillonnage et anti-boucles.
- Pièges : `generation_config.max_length`, décodage glouton, PowerShell.
- Sources : `model_utils.py`, `test.py`, `doc/05-generation.md`.

### Chapitre 9 — Ancrer : le pipeline RAG ✍️ à écrire

- TF-IDF sur caractères (`char_wb`, 3-5), indexation de `text + question`.
- Seuil `DEFAULT_MIN_SCORE = 0.50`, construction du prompt, `is_supported()`.
- Pourquoi le décodage contraint a été retiré au dernier commit.
- Ce que le GGUF contient et ne contient pas.
- Sources : `rag.py`, `prepare_rag_dataset.py`, `doc/12-rag.md`, commit `320a572`.

### Chapitre 10 — Évaluer sans se tromper soi-même ✍️ à écrire

- `evaluate.py` : exactitude, ancrage, abstentions, retrieval.
- Les résultats annoncés (30/30, 10/10) et leurs limites.
- Pourquoi il n'y a pas de jeu de validation.
- Sources : `evaluate.py`, `doc/12-rag.md`, `doc/11-limites.md`.

### Chapitre 11 — Servir, exporter, publier ✍️ à écrire

- `serve.py` : API locale avec la bibliothèque standard, `POST /generate`.
- `export_gguf.py` : convertisseur épinglé `b4000`, sortie F16.
- `publish_hf.py` : retries TLS, `MODEL_CARD.md` renommé `README.md`.
- llama.app et le piège de la régénération du GGUF après réentraînement.
- Sources : les trois scripts, `doc/06`, `07`, `08`, `MODEL_CARD.md`.

### Chapitre 12 — Ce que le projet montre et ce qu'il ne montre pas ✍️ à écrire

- Forces : chaîne complète reproductible, abstention vérifiable.
- Limites : base fermée, 256 tokens, pas de validation, GGUF sans RAG.
- Pistes : jeu de validation, quantification `Q4_K_M`, extension de la base.
- Sources : `doc/11-limites.md`, `README.md` §15, `MODEL_CARD.md`.

### Annexes ✍️ à écrire

- A. Carte complète du dépôt (fichiers versionnés / générés).
- B. Chronologie Git commentée et commandes pour la reconstituer.
- C. Glossaire : token, EOS, label `-100`, TF-IDF, RAG, GGUF, beam search.
- D. Scripts complets annotés (`train.py`, `rag.py`, `model_utils.py`).
- E. Sources et références.

------------------------------------------------------------------------

## 4. Conventions de rédaction

1. **Le code est réel.** Les extraits sont repris des fichiers de
   `picogpt1-daisy` ; les numéros de lignes indiqués permettent de les
   retrouver. Aucun code « d'illustration » n'est présenté comme du code du
   projet.
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
| Ch. 4 « Préparer les données » | Chapitres 5 et 6 |
| Ch. 5 « Entraîner un mini GPT » | Chapitre 7 |
| Ch. 6 « Relier à une base… » | Chapitres 9 et 10 |
| Ch. 7 « Passer du modèle à l'outil » | Chapitre 11 |
| Ch. 8 « Ce que le projet montre… » | Chapitre 12 |
| Annexe « Carte du dépôt » | Annexe A |

------------------------------------------------------------------------

## 6. État d'avancement

- [x] Sommaire corrigé et vérifié sur les sources
- [x] Généalogie des cinq dépôts vérifiée (2025-2026)
- [x] Chapitres 1, 2 et 3 (avec l'origine du projet)
- [ ] Chapitres 4 à 12 et annexes
- [ ] Relecture des extraits de code par rapport aux fichiers sources
- [ ] Ajout des prompts réels dans le chapitre 3 et des souvenirs de development
- [ ] Figer une édition destination publication
