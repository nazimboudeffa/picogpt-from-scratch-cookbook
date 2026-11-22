# Annexe G — Sources et références

*Cette annexe rassemble les repères bibliographiques cités ou retrouvés au fil
du manuscrit. Elle ne remplace pas les sources du projet — commits, fichiers et
README de `picogpt1-daisy` — qui restent la référence pour les faits datés.*

> **Précision conservée.** Le livre de Sebastian Raschka **n'a pas été lu** :
> c'est le **dépôt de code open source** associé qui a été étudié, avec l'aide de
> ChatGPT. Les entrées ci-dessous sont des repères : le récit n'attribue à
> aucune d'elles un choix d'implémentation qui ne soit pas vérifiable dans le
> code du projet.

------------------------------------------------------------------------

## G.1 Les quatre références

### Vaswani et al. — *Attention Is All You Need* (2017)

- Article : [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- Ce que le livre en retient : le texte d'origine du mécanisme d'attention et
  de l'architecture Transformer. C'est l'une des références citées dans le
  premier README de `picogpt-python` (chapitre 2, §2.2) ; le chapitre 4 montre
  ce que « reconstruire l'attention » signifie concrètement.

### PyTorch — documentation officielle

- Documentation : [pytorch.org/docs/stable](https://pytorch.org/docs/stable/index.html).
- Ce que le livre en retient : la référence des briques utilisées dans le code
  pédagogique — tenseurs, `torch.nn`, `torch.nn.functional`, optimiseurs et
  `manual_seed`. Le chapitre 4 et les annexes A et B s'appuient sur cette API.

### Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*

- Vidéo : [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- Ce que le livre en retient : un exemple de construction d'un GPT « en direct,
  en code », de la tokenisation à la génération. Il appartient à la famille des
  tutoriels YouTube qui, avec les articles, ont rendu l'architecture concrète
  (chapitre 1, §« Les articles et, surtout, les tutoriels YouTube », et
  chapitre 2, §2.1).

### Sebastian Raschka — *Build a Large Language Model (From Scratch)*

- Livre (Manning, 2024) ; code associé :
  [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch),
  sous licence **Apache-2.0** (Copyright © Sebastian Raschka).
- Ce que le livre en retient : l'ordonnancement des questions — représenter le
  texte, découper les exemples, masquer les labels, assembler le modèle,
  l'entraîner, puis l'ajuster. Le dépôt de code a été parcouru en juin 2025 et
  compris avec l'aide de ChatGPT ; l'ouvrage imprimé, lui, n'a pas été lu. La
  note d'attribution à régulariser dans `picogpt-python` est détaillée au
  chapitre 2, §2.2.

------------------------------------------------------------------------

## G.2 Ce qui reste ouvert

- Les autres articles et tutoriels vus en 2025 et 2026, dont le titre n'a pas
  été retrouvé, ne sont pas inventés : ils restent décrits sans titre au
  chapitre 2, §2.6.
- Les prompts réellement échangés avec l'IA ne figurent pas dans les commits ;
  le chapitre 3 attend toujours ces notes personnelles.
- Pour toute affirmation datée, la source première reste la généalogie des
  dépôts et la chronologie des commits du [sommaire](../00-sommaire.md), §1 et §2.

**Retour :** [Sommaire](../00-sommaire.md) ; **annexe précédente :** F — Scripts
complets annotés (à écrire).
