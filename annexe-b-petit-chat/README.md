# Annexe B — Petit chat avec `mon_gpt`

Ce dossier contient un script interactif qui reprend exactement `mon_gpt`
de l'[annexe A](../annexe-a-mon-gpt-complet/README.md) : même corpus, même
tokenizer, même architecture, même entraînement. Seule la fin change : le
prompt figé est remplacé par une boucle de conversation.

- Le code complet et ses commentaires :
  [annexe-b-petit-chat.md](annexe-b-petit-chat.md)
- Le chapitre correspondant :
  [chapitre 4 — Atelier](../04-atelier-de-zero-a-une-reponse.md)

## Lancer le chat

Recopier le bloc Python de l'annexe dans un fichier `mon_gpt_chat.py`, puis :

```bash
python mon_gpt_chat.py
```

Tape `quit` (ou `exit`, ou `q`) pour quitter ; `Ctrl+C` fonctionne aussi.
Le modèle attend une phrase commençant comme celles du corpus, par exemple
`la capitale de la france est`. Une question inédite produira du charabia :
c'est la limite de l'entraînement « from scratch », pas un bug.

## Environnement virtuel

Comme pour l'annexe A, installe `torch` dans un environnement virtuel plutôt
que dans l'installation globale de Python. La procédure Windows/PowerShell
et Linux/macOS est détaillée dans le
[README de l'annexe A](../annexe-a-mon-gpt-complet/README.md) — remplace
simplement le nom du script par `mon_gpt_chat.py`.

Retour au [sommaire](../00-sommaire.md).
