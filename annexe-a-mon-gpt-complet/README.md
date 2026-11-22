# Annexe A — `mon_gpt.py` commenté

Ce dossier contient la version intégrale et commentée du script de l'atelier
du [chapitre 4](../04-atelier-de-zero-a-une-reponse.md) :
[annexe-a-mon-gpt-complet.md](annexe-a-mon-gpt-complet.md).

Pour l'exécuter, recopier le bloc Python de l'annexe dans un fichier
`mon_gpt.py`, puis lancer :

```bash
python -m py_compile mon_gpt.py
python mon_gpt.py
```

Une seule dépendance : `torch`. Sortie vérifiée avec Python 3.13.2 et
torch 2.7.1 (graine fixée par `torch.manual_seed(0)`).

## Environnement virtuel

Mieux vaut installer `torch` dans un environnement virtuel que dans
l'installation globale de Python. Depuis ce dossier, sous **Windows /
PowerShell** :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
```

Sous **Linux / macOS** :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

Si PowerShell refuse l'activation de `.ps1`, l'autoriser pour la session
courante avant de relancer la commande :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

On quitte l'environnement avec `deactivate`. Le dossier `.venv` ne doit pas
être versionné ; le chapitre 5 reviendra sur ce point.

Voir aussi : [annexe B — petit chat avec `mon_gpt`](../annexe-b-petit-chat/README.md).

Retour au [sommaire](../00-sommaire.md).
