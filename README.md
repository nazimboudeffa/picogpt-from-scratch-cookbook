# Livre — PicoGPT from Scratch

Version de travail — **édition figée sur PicoGPT1-Daisy**. Le contenu est
rédigé à partir du dépôt réel
[`picogpt1-daisy`](https://github.com/nazimboudeffa/picogpt1-daisy), dont
l'état de référence est le commit `320a572` (21 novembre 2026, 01 h 25).
Ce dépôt a été renommé
[`picogpt-kit`](https://github.com/nazimboudeffa/picogpt-kit) le même jour à
03 h 45 (`f24cc33`) ; l'ancienne URL redirige vers lui. Le manuscrit reste
aligné sur `320a572`, pas sur l'état courant (`71e30d9`, PicoGPT1-Iris 1.1.0).

**Point d'entrée : [00-sommaire.md](00-sommaire.md)** — plan corrigé,
généalogie des dépôts, chronologie des commits et conventions de rédaction.

Ce manuscrit est relu et corrigé avec **OpenCode** (modèle **Big Pickle**), qui
sert aussi au développement courant de `picogpt-kit` ; le chapitre 3 précise ce
rôle par rapport aux outils utilisés en 2025 et 2026.

Le récit prend pour point de départ un ensemble de ressources d'apprentissage :
le **code open source** de
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — le livre
associé n'a pas été lu —, exploré avec l'aide de ChatGPT, de nombreux articles
(dont [l'article sur arxiv.org](https://arxiv.org/abs/1706.03762) ; les autres
références exactes n'ont pas toutes été conservées) et surtout des tutoriels
YouTube, dont celui de
[Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY). Il suit ensuite
les dépôts `picogpt-python` (juin 2025), `picogpt1-scratch-python` (avril 2026),
`picogpt1-jasmine` et `picogpt-kit` (novembre 2026).

## Articles, documentation et vidéo

Les ressources citées dans le récit sont regroupées dans
l'[annexe G](annexe-g-sources-et-references/annexe-g-sources-et-references.md).
Les repères les plus directs sont :

- **Article** — Vaswani et al., [*Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762),
  le texte fondateur de l'attention et du Transformer, cité dans le premier
  README de `picogpt-python`.
- **Documentation** — [PyTorch, documentation officielle](https://pytorch.org/docs/stable/index.html),
  référence des tenseurs et des briques `torch.nn` utilisées dans l'atelier.
- **Vidéo** — Andrej Karpathy,
  [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY),
  un GPT construit en direct, de la tokenisation à la génération.
- **Livre et code** — Sebastian Raschka, *Build a Large Language Model (From
  Scratch)* (Manning, 2024) : seul le
  [dépôt de code associé](https://github.com/rasbt/LLMs-from-scratch) a été
  étudié, avec l'aide de ChatGPT — le livre n'a pas été lu.

## Licences et sources

- Le manuscrit de ce dépôt est publié sous **Apache-2.0** (voir `LICENSE`, texte
  générique sans nom de titulaire renseigné). Apache-2.0 demande à toute personne
  qui redistribue le livre de conserver les avis de copyright, de fournir un
  exemplaire de la licence et de signaler les fichiers modifiés.
- Le code de référence
  [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) est sous
  **Apache-2.0** (Copyright © Sebastian Raschka). Le **livre** associé (Manning,
  2024) n'est pas couvert par cette licence et **n'a pas été lu** : c'est le code
  open source qui a été étudié, avec l'aide de ChatGPT.
- Les scripts de `picogpt1-daisy` / `picogpt-kit` appartiennent à un dépôt sans
  licence déclarée. Le chapitre 4 et les annexes présentent du code pédagogique
  écrit pour le livre, et non du code du projet.
- `model.ipynb` de `picogpt-python` (juin 2025) reprend des structures très
  proches du code amont ; l'attribution Apache et la copie de la licence restent
  à ajouter dans ce dépôt (voir chapitre 2, §2.2).
- Les repères bibliographiques sont détaillés dans
  l'[annexe G](annexe-g-sources-et-references/annexe-g-sources-et-references.md).

Chapitres déjà rédigés :

1. [Une idée : apprendre en construisant](01-une-idee-apprendre-en-construisant.md)
2. [Une inspiration pédagogique](02-une-inspiration-pedagogique.md)
3. [Programmer avec l'aide de l'IA](03-programmer-avec-l-aide-de-l-ia.md)
4. [Atelier : de zéro à une première réponse](04-atelier-de-zero-a-une-reponse.md)

Annexes rédigées :

- [A. `mon_gpt.py` complet et commenté (chapitre 4)](annexe-a-mon-gpt-complet/annexe-a-mon-gpt-complet.md)
- [B. Petit chat avec `mon_gpt`](annexe-b-petit-chat/annexe-b-petit-chat.md)
- [G. Sources et références](annexe-g-sources-et-references/annexe-g-sources-et-references.md)

Les autres chapitres sont annoncés dans le sommaire et restent à écrire.

## Les deux exemples à lancer

Les deux exemples pédagogiques du livre sont autonomes : une seule dépendance,
`torch`, aucun téléchargement de poids. Le code est à recopier depuis les annexes.

1. [`mon_gpt.py`](annexe-a-mon-gpt-complet/README.md) — le script intégral de
   l'atelier : `python -m py_compile mon_gpt.py`, puis `python mon_gpt.py`.
   Le modèle apprend cinq phrases françaises et continue « la capitale de la
   france est ».
2. [`mon_gpt_chat.py`](annexe-b-petit-chat/README.md) — le même modèle avec une
   boucle de conversation : `python mon_gpt_chat.py` (`quit` pour quitter).

La création d'un environnement virtuel, sous Windows/PowerShell comme sous
Linux/macOS, est expliquée dans le
[README de l'annexe A](annexe-a-mon-gpt-complet/README.md).

> Les fichiers cités dans le livre (`train.py`, `rag.py`, `knowledge.jsonl`,
> `doc/…`) appartiennent au dépôt `picogpt1-daisy`, aujourd'hui `picogpt-kit`.
> Le manuscrit est versionné dans `picogpt-from-scratch-cookbook` ; l'ancienne
> URL `picogpt-from-scratch` redirige vers ce dépôt.
