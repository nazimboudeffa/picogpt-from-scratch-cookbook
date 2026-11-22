# Annexe A — `mon_gpt.py` complet et commenté (chapitre 4)

*Cette annexe reproduit en un seul bloc le script présenté en six étapes dans
le [chapitre 4](../04-atelier-de-zero-a-une-reponse.md). Elle est prête à être
recopiée et exécutée.*

> **Nature du code.** Comme dans le chapitre 4, il s'agit d'un **exemple
> pédagogique écrit pour ce livre**, et non d'un fichier de
> `picogpt1-daisy`. La convention n° 1 du [sommaire](../00-sommaire.md) reste
> respectée : le code du projet est étudié aux chapitres 5 à 12.

## A.1 Mode d'emploi

Il est recommandé de travailler dans un environnement virtuel : la marche à
suivre est dans le [README du dossier](README.md).

1. Recopier **tout** le bloc ci-dessous dans un fichier nommé `mon_gpt.py`.
2. Vérifier la syntaxe : `python -m py_compile mon_gpt.py`.
3. Exécuter : `python mon_gpt.py`.

Une seule dépendance est nécessaire : `torch` (aucun `transformers`, aucun
`datasets`, aucun téléchargement de poids). Vérification faite avec
**Python 3.13.2** et **torch 2.7.1**, en CPU, en environ 7 secondes. La graine
est fixée à zéro : la sortie doit être identique à celle du chapitre 4.

## A.2 Le script intégral

```python
"""mon_gpt.py — script complet de l'atelier du chapitre 4.

Recopier ce fichier tel quel puis lancer :

    python mon_gpt.py

Le modèle part de poids aléatoires, apprend cinq phrases françaises par
cœur, puis continue une phrase commençant par « la capitale de la france est ».
Il s'agit d'une démonstration pédagogique : aucun code de picogpt1-daisy.
"""

import torch                       # tenseurs, tirages aléatoires, optimiseurs
import torch.nn as nn              # briques : Linear, Embedding, LayerNorm, Module
from torch.nn import functional as F  # fonctions sans état : softmax, cross_entropy

# ---------------------------------------------------------------------------
# 1. Reproductibilité
# ---------------------------------------------------------------------------
# Les poids initiaux et l'ordre des lots sont tirés au hasard. Fixer la graine
# rend l'exécution identique d'une fois sur l'autre ; c'est indispensable
# pour comparer une sortie documentée avec une sortie observée.
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 2. Corpus
# ---------------------------------------------------------------------------
# Cinq phrases, répétées vingt fois : 2 560 caractères au total. Répéter le
# même texte est volontaire — avec si peu de données, le modèle mémorise.
# C'est exactement la limite que le chapitre 4 demande de constater.
text = """
la capitale de la france est paris.
la capitale du japon est tokyo.
l eau bout a 100 degres.
le ciel est bleu.
je ne sais pas.
""" * 20

# ---------------------------------------------------------------------------
# 3. Tokenizer minimal (caractère → entier)
# ---------------------------------------------------------------------------
# Un token = un caractère, y compris l'espace et le saut de ligne.
# `sorted(set(text))` donne 24 symboles : c'est la taille du vocabulaire.
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}        # caractère → identifiant
itos = {i: c for c, i in stoi.items()}            # identifiant → caractère
encode = lambda s: [stoi[c] for c in s]           # chaîne → liste d'identifiants
decode = lambda ids: "".join(itos[i] for i in ids)  # inverse

# Le corpus entier devient un seul tenseur d'identifiants (dtype long = entiers).
data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------------------------------------------------------
# 4. Hyperparamètres et fabrication des lots
# ---------------------------------------------------------------------------
# batch_size  : 16 fenêtres traitées en parallèle
# block_size  : 48 caractères de contexte (assez pour une phrase entière)
# steps       : 400 mises à jour des poids
batch_size, block_size, steps = 16, 48, 400


def batch():
    """Renvoie x (fenêtres) et y (les mêmes fenêtres décalées d'un caractère).

    À chaque position, la cible est le caractère suivant. C'est ce décalage
    qui transforme du texte brut en tâche d'apprentissage supervisée.
    """
    # Choisir 16 points de départ au hasard dans le corpus.
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    # x : ce que le modèle voit.
    x = torch.stack([data[i:i + block_size] for i in starts])
    # y : ce qu'il doit prédire ; note le +1 sur les deux bornes.
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


# ---------------------------------------------------------------------------
# 5. Attention causale multi-têtes, écrite à la main
# ---------------------------------------------------------------------------
class TeteAttention(nn.Module):
    """Une tête d'attention : requêtes, clés, valeurs et masque causal."""

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        # Une seule projection produit les trois matrices q, k et v.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Projection de sortie, après recombinaison des têtes.
        self.proj = nn.Linear(d_model, d_model)
        # torch.tril garde le triangle inférieur : la position t ne peut
        # regarder que les positions 0..t, jamais le futur.
        mask = torch.tril(torch.ones(block_size, block_size))
        # register_buffer : le masque suit le modèle (CPU/GPU) sans être
        # un paramètre entraînable.
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape                       # batch, contexte, dimensions
        q, k, v = self.qkv(x).split(C, dim=2)   # découper en trois blocs

        def split_heads(t):
            """Réorganiser (B, T, C) en (B, n_heads, T, head_size)."""
            return t.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # Scores d'attention : similarité requête/clé, mise à l'échelle.
        att = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        # Les cases interdites deviennent -inf ; après softmax elles valent 0.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # Mélange des valeurs selon les poids d'attention.
        out = att @ v
        # Recombinaison des têtes, puis retour à la forme (B, T, C).
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# 6. Bloc Transformer et modèle complet
# ---------------------------------------------------------------------------
class Bloc(nn.Module):
    """Attention + réseau fully connected, avec LayerNorm et résidus."""

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.att = TeteAttention(d_model, n_heads, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Le MLP élargit puis réduit : 64 → 256 → 64 dimensions.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # x = x + sous-couche : la connexion résiduelle laisse passer le
        # signal d'origine et rend l'optimisation plus stable.
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Deux blocs, une tête linéaire sur le vocabulaire : 106 264 paramètres."""

    def __init__(self, vocab_size, block_size, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.block_size = block_size
        # Embedding du caractère lu.
        self.tok = nn.Embedding(vocab_size, d_model)
        # Embedding de position : l'attention seule ne sait pas où l'on est.
        self.pos = nn.Embedding(block_size, d_model)
        # Empilement de deux blocs identiques.
        self.blocks = nn.Sequential(
            *[Bloc(d_model, n_heads, block_size) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        # Un score par caractère possible (24 ici) à la position suivante.
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Positions 0..T-1 ; `device=idx.device` évite un plantage sur GPU.
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            # cross_entropy attend (N, vocab) et (N) : on aplatit les deux.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()             # la génération ne calcule pas de gradients
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Ajoute un caractère à la fois, en échantillonnant selon les scores."""
        for _ in range(max_new_tokens):
            # On ne garde que les block_size derniers caractères.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Diviser par temperature adoucit (ou durcit) la distribution.
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # tirage pondséré
            idx = torch.cat((idx, nxt), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 7. Entraînement
# ---------------------------------------------------------------------------
model = MiniGPT(len(chars), block_size)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(steps):
    x, y = batch()
    _, loss = model(x, y)         # 1. calcul de l'erreur
    opt.zero_grad()               # 2. oubli des gradients précédents
    loss.backward()               # 3. rétropropagation
    opt.step()                    # 4. mise à jour des poids
    # Journaliser le début, la fin et une mesure intermédiaire tous les 100 pas.
    if step % 100 == 0 or step == steps - 1:
        print(f"etape {step:3d} - loss {loss.item():.3f}")

# ---------------------------------------------------------------------------
# 8. Première génération
# ---------------------------------------------------------------------------
# Le prompt est une chaîne vue pendant l'entraînement : le modèle peut donc
# continuer avec « paris. » puis enchaîner sur une autre phrase mémorisée.
prompt = "la capitale de la france est "
out = model.generate(torch.tensor([encode(prompt)]), max_new_tokens=60)
print(decode(out[0].tolist()))
```

## A.3 Sortie attendue

```text
etape   0 - loss 3.287
etape 100 - loss 0.249
etape 200 - loss 0.077
etape 300 - loss 0.051
etape 399 - loss 0.064
la capitale de la france est paris.
la capitale du japon est tokyo.
l eau bout a 100 degr
```

La réponse s'arrête au milieu de `degres` uniquement parce que
`max_new_tokens=60`. Le modèle ne raisonne pas : il rejoue des motifs vus.
C'est le point de départ du chapitre 5 et du passage au code réel de
`train.py`.

## A.4 Repères

| Section du script | Section du chapitre 4 | Ce qu'il faut retenir |
| --- | --- | --- |
| Corpus et tokenizer | 4.2 | 2 560 caractères, 24 tokens |
| Lots | 4.3 | `y` est décalé d'un caractère |
| `TeteAttention` | 4.4 | le masque triangulaire interdit le futur |
| `Bloc` et `MiniGPT` | 4.5 | résidus, LayerNorm, tête de 24 scores |
| Entraînement | 4.6 | `cross_entropy` + `AdamW`, 400 pas |
| `generate()` et sortie | 4.7 | échantillonnage à température 0,8 |

**Retour :** [Chapitre 4 — Atelier : de zéro à une première réponse](../04-atelier-de-zero-a-une-reponse.md) ;
**Sommaire :** [00-sommaire.md](../00-sommaire.md).
