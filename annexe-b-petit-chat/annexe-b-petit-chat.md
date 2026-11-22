# Annexe B — Petit chat avec `mon_gpt`

*Cette annexe transforme le script du chapitre 4 en une petite conversation
interactive. Elle garde exactement `mon_gpt` — même corpus, même tokenizer,
même architecture, même entraînement — et ne change que la dernière partie :
le prompt figé devient une saisie au clavier.*

> **Nature du code.** Comme les annexes A et le chapitre 4, il s'agit d'un
> **exemple pédagogique écrit pour ce livre**, et non d'un fichier de
> `picogpt1-daisy`. La convention n° 1 du [sommaire](../00-sommaire.md) reste
> respectée : le code du projet est étudié aux chapitres 5 à 12.

## B.1 Pourquoi un chat, et pourquoi maintenant

Le chapitre 4 se termine par une seule réponse, obtenue avec un prompt écrit
en dur. Une boucle `input()` → `generate()` → `print()` suffit pour donner
l'impression d'un assistant, sans introduire le moindre concept nouveau. C'est
le prolongement naturel de l'atelier ; le chapitre 5 peut ensuite rester
consacré à l'installation.

Le bloc fourni ici est à recopier dans un fichier nommé `mon_gpt_chat.py`.
Il est **autonome** : on le lance directement, sans importer l'annexe A. Cela évite
de modifier `mon_gpt` et garde les deux annexes indépendantes. On pourra
revenir sur ce chat au chapitre 9, quand le projet réel montre son `test.py`,
pour mesurer le chemin parcouru entre ce jouet et la génération ancrée.

## B.2 Le script intégral

Le bloc ci-dessous est le script complet, à recopier tel quel dans
`mon_gpt_chat.py`.

```python
"""mon_gpt_chat.py — petit chat interactif avec le modèle du chapitre 4.

Ce script reprend exactement `mon_gpt` (annexe A) : même corpus, même
tokenizer, même architecture, même entraînement. Seule la dernière partie
change : le prompt figé est remplacé par une boucle de conversation.

Usage :
    python mon_gpt_chat.py

Tape `quit` (ou `exit`, ou `q`) pour quitter. Le modèle ayant été entraîné
sur cinq phrases, il ne sait répondre qu'à ce qu'il a mémorisé : c'est le
but de l'annexe — voir la limite de « from scratch » en action.
"""

import torch                       # tenseurs, tirages aléatoires, optimiseurs
import torch.nn as nn              # briques : Linear, Embedding, LayerNorm, Module
from torch.nn import functional as F  # softmax, cross_entropy

# ---------------------------------------------------------------------------
# 1. Reproductibilité
# ---------------------------------------------------------------------------
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 2. Corpus : cinq phrases répétées vingt fois (2 560 caractères)
# ---------------------------------------------------------------------------
text = """
la capitale de la france est paris.
la capitale du japon est tokyo.
l eau bout a 100 degres.
le ciel est bleu.
je ne sais pas.
""" * 20

# ---------------------------------------------------------------------------
# 3. Tokenizer minimal : un token = un caractère (24 symboles)
# ---------------------------------------------------------------------------
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------------------------------------------------------
# 4. Hyperparamètres et lots
# ---------------------------------------------------------------------------
batch_size, block_size, steps = 16, 48, 400


def batch():
    """Fenêtres de 48 caractères et cible décalée d'un caractère."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


# ---------------------------------------------------------------------------
# 5. Attention causale multi-têtes
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. Bloc Transformer et modèle : identique à mon_gpt (106 264 paramètres)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 7. Entraînement : exactement les 400 pas de l'annexe A
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 8. La seule nouveauté : une boucle de conversation
# ---------------------------------------------------------------------------
# Ici, pas de prompt écrit en dur dans le script : c'est l'utilisateur qui
# le tape. On encode directement sa saisie, sans ajouter de `\n` : le modèle
# a appris des phrases continues et doit continuer sur la même ligne.
print("Petit chat avec mon_gpt — tape 'quit' pour quitter.")
try:
    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        reponse = model.generate(torch.tensor([encode(question)]),
                                 max_new_tokens=60)
        print(decode(reponse[0].tolist()))
except (KeyboardInterrupt, EOFError):
    # Ctrl+C ou fin d'entrée (redirection de fichier) : sortie propre.
    print("\nAu revoir.")
```

## B.3 Exemple de session

Après les 400 pas d'entraînement (les mêmes que dans l'annexe A), le chat
affiche son invitation. On tape une phrase commençant comme une phrase du
corpus, puis `quit` pour sortir :

```text
etape   0 - loss 3.287
etape 100 - loss 0.249
etape 200 - loss 0.077
etape 300 - loss 0.051
etape 399 - loss 0.064
Petit chat avec mon_gpt — tape 'quit' pour quitter.
> la capitale de la france est
la capitale de la france est paris.
la capitale du japon est tokyo.
l eau bout a 100 degr
> quit
```

## B.4 Ce que le chat montre — et sa limite

- Le modèle continue **ce qu'on lui donne**, y compris une phrase inachevée :
  c'est exactement la tâche apprise au chapitre 4.
- Aucun `
` n'est ajouté au prompt : dans le corpus, une phrase suit
  immédiatement la précédente après un saut de ligne, mais la saisie de
  l'utilisateur est traitée comme le début d'une ligne.
- Posez une question absente des cinq phrases mémorisées : vous obtiendrez
  une suite de caractères sans rapport. Le chat ne sait ni raisonner, ni
  refuser poliment, ni consulter une base — le RAG et l'abstention arrivent
  aux chapitres 10 et 11.
- `Ctrl+C` et la fin d'entrée (`EOFError`) terminent proprement la session.

**Retour :** [Chapitre 4 — Atelier](../04-atelier-de-zero-a-une-reponse.md) ;
**annexe précédente :** [Annexe A — `mon_gpt.py` commenté](../annexe-a-mon-gpt-complet/annexe-a-mon-gpt-complet.md) ;
**sommaire :** [00-sommaire.md](../00-sommaire.md).
