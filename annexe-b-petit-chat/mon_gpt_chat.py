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
