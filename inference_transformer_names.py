"""
Generate Indian names with the transformer trained by indian_names_transformer.py.

Set the constants below, then run:  python predict_inidan_names/inference_transformer_names.py
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- settings: edit these ----
CKPT_PATH = "./predict_indian_names/transformer_names_45500.pt" # Path(__file__) /  # e.g. "transformer_names_4000.pt" for a specific step
NUM_NAMES = 50
PREFIX = ""          # names start with this, e.g. "pra"; "" = anything
TEMPERATURE = 1.0    # lower = safer, more common names; higher = more creative
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
stoi, itos = ckpt["stoi"], ckpt["itos"]

# hyperparameters come from the checkpoint, so they always match training
_cfg = ckpt["config"]
n_emb = _cfg["n_emb"]
num_heads = _cfg["num_heads"]
n_layer = _cfg["n_layer"]
block_size = _cfg["block_size"]
vocab_size = _cfg["vocab_size"]
dropout = _cfg["dropout"]

encode = lambda s: [stoi[c] for c in s]
decode = lambda idxs: "".join(itos[i] for i in idxs)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(n_emb, head_size)
        self.keys = nn.Linear(n_emb, head_size)
        self.values = nn.Linear(n_emb, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x)
        k = self.keys(x)

        wei = q @ k.transpose(-2,-1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.values(x)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, n_emb, num_heads):
        super().__init__()

        head_size = n_emb // num_heads
        self.multi_attension = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.multi_attension(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class Transormer(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)
        self.positional_embeddings = nn.Embedding(block_size, n_emb)

        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        token_emb = self.embedding_layer(idx)
        positional_emb = self.positional_embeddings(torch.arange(T, device=device))

        x = token_emb + positional_emb

        attention = self.blocks(x)

        x = self.ln_f(attention)
        logits = self.linear(x)
        return logits

    @torch.no_grad
    def generate_name(self, prefix="", temperature=1.0):
        """One name, one character at a time, until the model writes '.' (end)."""
        idx = torch.tensor([[0] + encode(prefix)], dtype=torch.long, device=device)
        while idx.shape[1] < block_size:
            logits = self(idx)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            if ix.item() == 0:
                break
            idx = torch.cat((idx, ix), dim=1)
        return decode(idx[0, 1:].tolist())


model = Transormer(n_emb)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()


if __name__ == "__main__":
    prefix = PREFIX.strip().lower()
    unknown = sorted({c for c in prefix if c not in stoi or c == "."})
    if unknown:
        raise SystemExit(f"Unsupported characters in PREFIX: {' '.join(unknown)}")

    print(f"Model: {CKPT_PATH} (step {ckpt['step']})")
    for _ in range(NUM_NAMES):
        print(model.generate_name(prefix, TEMPERATURE).capitalize())
