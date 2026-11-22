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

Le récit prend pour point de départ un ensemble de ressources d'apprentissage :
le **code open source** de
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — le livre
associé n'a pas été lu —, exploré avec l'aide de ChatGPT, de nombreux articles
(dont les références exactes n'ont pas toutes été conservées) et surtout des
tutoriels YouTube. Il suit ensuite les dépôts `picogpt-python` (juin 2025),
`picogpt1-scratch-python` (avril 2026), `picogpt1-jasmine` et `picogpt1-daisy`
(novembre 2026).

## Licences et sources

- Le manuscrit de ce dépôt est publié sous **Apache-2.0** (voir `LICENSE`, texte
  générique sans nom de titulaire renseigné). Les versions déjà diffusées sous
  CC0 1.0 le restent : la renonciation CC0 étant irrévocable, ce changement vaut
  pour la version présente et les suivantes. Apache-2.0 demande à toute personne
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

Chapitres déjà rédigés :

1. [Une idée : apprendre en construisant](01-une-idee-apprendre-en-construisant.md)
2. [Une inspiration pédagogique](02-une-inspiration-pedagogique.md)
3. [Programmer avec l'aide de l'IA](03-programmer-avec-l-aide-de-l-ia.md)
4. [Atelier : de zéro à une première réponse](04-atelier-de-zero-a-une-reponse.md)

Annexe rédigée :

- [A. `mon_gpt.py` complet et commenté (chapitre 4)](annexe-a-mon-gpt-complet/annexe-a-mon-gpt-complet.md)
- [B. Petit chat avec `mon_gpt`](annexe-b-petit-chat/annexe-b-petit-chat.md)

Les autres chapitres sont annoncés dans le sommaire et restent à écrire.

> Les fichiers cités dans le livre (`train.py`, `rag.py`, `knowledge.jsonl`,
> `doc/…`) appartiennent au dépôt `picogpt1-daisy`, aujourd'hui `picogpt-kit`.
> Le manuscrit est versionné dans `picogpt-from-scratch-cookbook` ; l'ancienne
> URL `picogpt-from-scratch` redirige vers ce dépôt.
