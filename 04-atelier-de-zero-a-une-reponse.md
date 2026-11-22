# Chapitre 4 — Atelier : de zéro à une première réponse

*Ce chapitre répond à la question : que faut-il écrire, concrètement, pour
qu'un modèle parti de poids aléatoires produise une suite de texte cohérente ?
À la fin, le lecteur a exécuté un GPT miniature d'environ 106 000 paramètres
et il sait exactement ce qui, dans cette chaîne, sera délégué à Hugging Face
dans le projet réel.*

> **Nature du code de cet atelier.** Les extraits des chapitres 1 à 3 sont
> repris de `picogpt-kit`. Ceux-ci **ne le sont pas** : ils ont été écrits
> pour ce livre afin de rendre visible ce que `GPT2LMHeadModel` cache. Ils sont
> présentés comme un **exemple pédagogique**, jamais comme du code du dépôt.
> La convention n° 1 du [sommaire](00-sommaire.md) reste donc respectée : le
> code du projet est étudié aux chapitres 5 à 12, le code d'apprentissage est
> signalé comme tel ici.

------------------------------------------------------------------------

## 4.1 Le périmètre : cinq phrases et un vocabulaire de caractères

Le projet raconté jusqu'ici fonctionne parce qu'il part de GPT-2
pré-entraîné. Pour voir ce que « partir de zéro » veut dire, on fait l'inverse :
**aucun poids pré-entraîné, aucun tokenizer téléchargé, aucun jeu de données
externe**. Le corpus tient dans cinq phrases françaises répétées vingt fois,
soit 2 560 caractères, et le vocabulaire est l'ensemble des caractères
rencontrés (24 symboles).

Le but n'est pas de fabriquer un assistant : c'est de rendre chaque mécanisme
exécutable. On construit successivement un tokenizer, des lots, une attention
causale, deux blocs Transformer, une boucle d'optimisation, puis une fonction
de génération. C'est la même liste d'étapes que le chapitre 1, mais réduite à
un seul fichier.

| Étape de l'atelier | Ce que fait le code | Où le projet réel s'en charge |
| --- | --- | --- |
| Tokenizer | Caractère → entier, avec `stoi`/`itos` | Tokenizer GPT-2 BPE, `train.py:138` (chapitre 7) |
| Lots | Fenêtres de 48 caractères, cible décalée d'un | `build_example()`, `train.py:53-66` (chapitre 7) |
| Architecture | Attention, `LayerNorm`, MLP écrits à la main | `GPT2LMHeadModel` (chapitre 2) |
| Entraînement | `AdamW` + `cross_entropy` sur 400 pas | `Trainer` Hugging Face (chapitre 8) |
| Génération | Échantillonnage avec température | `model_utils.py` + `rag.py` (chapitres 9 et 10) |

Il n'est pas question de rivaliser avec les 124 439 808 paramètres de Daisy.
On vise seulement un comportement observable : après entraînement, le modèle
doit continuer `la capitale de la france est ` par `paris.`, puis enchaîner
avec une autre phrase du corpus. C'est ce que signifie « fonctionne un peu »
ici : **reproduire des motifs vus, pas raisonner**.

------------------------------------------------------------------------

## 4.2 Bloc 1 — le corpus et le tokenizer

Les six blocs de code qui suivent se recopient **dans cet ordre**, dans un
seul fichier nommé `mon_gpt.py`. Aucun autre fichier n'est nécessaire.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(0)

text = """
la capitale de la france est paris.
la capitale du japon est tokyo.
l eau bout a 100 degres.
le ciel est bleu.
je ne sais pas.
""" * 20

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
```

Quatre lignes suffisent au tokenizer : `chars` est la liste triée des
caractères uniques, `stoi` et `itos` sont les deux sens de la correspondance,
et `data` est le corpus entier sous forme d'identifiants. `torch.manual_seed(0)`
rend la démonstration reproductible : les poids initiaux et l'ordre des lots
seront les mêmes à chaque exécution.

Ce tokenizer est volontairement naïf. Un token GPT-2 peut couvrir plusieurs
lettres ; ici un token = une lettre, y compris l'espace et le saut de ligne.
Le chapitre 7 montrera pourquoi re-tokeniser `prompt\n` en entier est
indispensable avec un vrai tokenizer.

------------------------------------------------------------------------

## 4.3 Bloc 2 — prédire le caractère suivant

```python
batch_size, block_size, steps = 16, 48, 400

def batch():
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y
```

`x` contient 16 fenêtres de 48 caractères ; `y` contient les **mêmes fenêtres
décalées d'un caractère**. Le modèle voit donc une fenêtre comme
`la capitale de la france est par...` et doit prédire
`a capitale de la france est pari...` : à chaque position, la cible est le
caractère suivant.

Un exemple sur une seule fenêtre, avec des caractères au lieu d'identifiants :

| Position | 1 | 2 | 3 | 4 | … |
| --- | --- | --- | --- | --- | --- |
| Entrée `x` | `l` | `a` | ` ` | `c` | … |
| Cible `y` | `a` | ` ` | `c` | `a` | … |

Le masque causal, ajouté au bloc suivant, garantit qu'à la position 3 le modèle
ne peut pas regarder la cible `c`. Sans ce masque, il pourrait copier la
réponse au lieu de l'apprendre. Ici toutes les fenêtres ont la même longueur :
pas de padding, donc pas de labels `-100` — ce mécanisme du projet réel est
expliqué au chapitre 7.

------------------------------------------------------------------------

## 4.4 Bloc 3 — l'attention causale écrite à la main

```python
class TeteAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
```

La projection `qkv` produit trois vues du même texte : *requête*, *clé* et
*valeur*. Le produit `q @ kᵀ` donne les scores d'attention, divisés par
`√head_size` pour éviter des valeurs trop grandes. La matrice triangulaire
inférieure `torch.tril` interdit les positions futures : les scores masqués
deviennent `-inf`, puis `softmax` les transforme en zéro. Enfin, on recombine
les têtes et on projette le résultat.

C'est exactement le genre de blocs qui étaient écrits à la main dans
`picogpt-python` en juin 2025 (`MultiHeadAttention` dans `model.ipynb`, chapitre 2). Dans le Daisy de `320a572`, ce code n'existe plus : il est fourni par
`GPT2LMHeadModel`. L'atelier permet donc de voir une fois ce que le projet
délègue ensuite.

------------------------------------------------------------------------

## 4.5 Bloc 4 — empiler les blocs et prévoir le vocabulaire

```python
class Bloc(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.att = TeteAttention(d_model, n_heads, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[Bloc(d_model, n_heads, block_size) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
```

Un bloc = attention + réseau fully connected, chaque fois précédé d'une
`LayerNorm` et entouré d'une connexion résiduelle (`x = x + …`). Le modèle
ajoute une embedding de position, car l'attention seule ne sait pas où se
trouve un caractère dans la phrase. La tête linéaire produit un score pour
chacun des 24 caractères possibles : la **loss** compare ces scores au
caractère suivant.

`generate()` est volontairement simple : on prend le dernier caractère prédit,
on l'ajoute à la séquence, et on recommence. `temperature=0.8` assouplit la
distribution ; une température très basse donnerait une réponse plus
déterministe, mais aussi plus encline à boucler — le chapitre 9 montrera
comment `StopAtNewline` et `no_repeat_ngram_size` traitent ce problème dans le
projet réel.

------------------------------------------------------------------------

## 4.6 Bloc 5 — la boucle d'entraînement

```python
model = MiniGPT(len(chars), block_size)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(steps):
    x, y = batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == steps - 1:
        print(f"etape {step:3d} - loss {loss.item():.3f}")
```

Quatre lignes composent le cœur de l'apprentissage : calcul de la loss,
remise à zéro des gradients, rétropropagation, mise à jour des poids. Il n'y a
ni checkpoint ni planificateur de taux d'apprentissage : ce sont des
simplifications assumées, pas des oublis. Le `Trainer` du chapitre 8 fait la
même chose, avec en plus la gestion des checkpoints, du GPU et du fp16.

------------------------------------------------------------------------

## 4.7 Bloc 6 — la première réponse

```python
prompt = "la capitale de la france est "
out = model.generate(torch.tensor([encode(prompt)]), max_new_tokens=60)
print(decode(out[0].tolist()))
```

Une version intégralement commentée, prête à être recopiée d'un seul bloc,
est reproduite en [annexe A](annexe-a-mon-gpt-complet/annexe-a-mon-gpt-complet.md).
Une variante interactive, qui garde le même modèle mais remplace le prompt
figé par une boucle de conversation, est proposée en
[annexe B](annexe-b-petit-chat/annexe-b-petit-chat.md). Le script complet
s'exécute avec les deux commandes suivantes :

```bash
python -m py_compile mon_gpt.py   # vérification syntaxique, chapitre 5
python mon_gpt.py
```

Sortie observée lors de la vérification de ce chapitre (Python 3.13.2,
`torch` 2.7.1, graine fixée à 0, exécution CPU en environ 7 secondes) :

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

La loss passe de 3,287 à 0,064 : les poids aléatoires ont été ajustés jusqu'à
reproduire presque parfaitement le corpus. La réponse commence bien par
`paris.`, puis continue avec une autre phrase apprise. Elle s'arrête au milieu
de `degres` uniquement parce qu'on a demandé 60 caractères.

> **À ne pas sur-interpréter.** Ce modèle n'a pas « compris » la géographie :
> il a mémorisé cinq phrases. Pose-lui une question absente du corpus et il
> produira une suite de caractères sans rapport. C'est précisément le
> problème que le projet traite ensuite avec un préentraînement, un jeu de
> données plus large et le RAG.

------------------------------------------------------------------------

## 4.8 Ce que l'atelier montre, et ce qu'il ne montre pas

| | Cet atelier | `picogpt-kit` à `320a572` |
| --- | --- | --- |
| Initialisation | Poids aléatoires | GPT-2 pré-entraîné (`--init pretrained`) |
| Paramètres | 106 264 | 124 439 808 |
| Vocabulaire | 24 caractères | 50 257 tokens BPE |
| Données | 5 phrases × 20 = 2 560 caractères | 2 442 exemples (2 000 + 245 + 197) |
| Contexte | 48 caractères | 256 tokens |
| Entraînement | 400 pas, `AdamW`, CPU | 5 époques, ≈ 3 050 pas, CUDA + `--fp16` |
| Résultat | Continue un texte mémorisé | Répond avec ancrage et abstention |

Trois leçons survivent à la réduction d'échelle :

1. **La boucle est courte.** Une dizaine de lignes suffisent pour transformer
   du texte en probabilités, puis des probabilités en caractères. Le reste du
   projet est surtout du réglage, de la donnée et de la vérification.
2. **Le format d'entraînement commande la sortie.** Le modèle continue ce
   qu'on lui a montré. Dans l'atelier, il continue des phrases ; dans Daisy,
   il a appris `prompt` + `\n` + `réponse` + `<|endoftext|>` — d'où le `\n`
   ajouté à la génération (chapitre 9).
3. **« From scratch » a deux sens.** Ici, les poids *et* l'architecture sont
   construits à la main. Le chapitre 1 explique pourquoi le livre garde le
   premier sens (la démarche) alors que le projet utilise le second
   (l'initialisation aléatoire) seulement pour comparer.

Ce que l'atelier ne démontre pas : l'attention à grande échelle, la
généralisation à des questions inédites, l'évaluation rigoureuse ni
l'ancrage. Ces sujets sont traités aux chapitres 8, 10 et 11.

------------------------------------------------------------------------

## 4.9 Ce qu'il faut retenir

- Un GPT complet, même minuscule, tient dans une centaine de lignes et
  **106 264 paramètres** ; la difficulté du projet réel vient des données et
  du comportement, pas du nombre de lignes de l'architecture.
- L'attention causale se résume à une matrice triangulaire qui met les
  positions futures à `-inf` avant le `softmax`.
- Entraîner, c'est décaler la cible d'un caractère, calculer une
  `cross_entropy` et laisser `AdamW` corriger les poids.
- Un modèle entraîné sur cinq phrases **mémorise** ; il ne répond pas. C'est la
  limite que le passage à GPT-2 pré-entraîné puis au RAG vient corriger.

**Suite :** chapitre 5 — *Installer et reproduire l'environnement* (à écrire,
comme les suivants) — on y retrouvera `torch`, les versions et la vérification
par `python -m py_compile` utilisés dans cet atelier, avant d'étudier le code
réel de `train.py`.
