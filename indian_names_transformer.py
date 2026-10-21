"""
Indian name generator with the GPT (transformer) architecture from chess_gpt_move_tokens.py.

Same data and split as predict-indian-names-wavenet-arch.ipynb (first 80% train, next 10% val,
last 10% test, file order), so the losses are directly comparable with the WaveNet results
(train 1.90 / val 2.01 / test 2.01).

Model size is chosen for the dataset, not to match WaveNet: ~0.8M training tokens
(91K names x ~9 characters incl. the end token) -> ~200K params, ~4 tokens per param.
Small character datasets are trained for many passes, so overfitting is guarded by dropout and
early stopping: the exact val loss is checked every eval_interval steps, only the best model
is saved, and training stops after `patience` checks without improvement.

Each training example is one complete name, like the chess model's one-game-per-example:
    x:  .  p  r  a  k  a  s  h        ('.' = start)
    y:  p  r  a  k  a  s  h  .        ('.' = end)
padded to the longest name in the batch; padding is ignored in the loss (-100).

Run from anywhere:  python predict_inidan_names/indian_names_transformer.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DATA_PATH = HERE.parent / "generate_names" / "names_dataset.txt"
CKPT_PATH = HERE / "transformer_names_19500.pt"

# Hyper parameters
max_iters = 50000   # upper limit; early stopping usually ends training before this
batch_size = 128     # names per batch
eval_interval = 500
patience = 4         # stop after this many evals (x eval_interval steps) without a better val loss
learning_rate = 1e-3
n_emb = 64
num_heads = 4
n_layer = 4
dropout = 0.2
eval_iters = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

IGNORE = -100        # F.cross_entropy skips targets with this value


# Reading input (same parsing as the WaveNet notebook: lines look like "name: prakash")
with open(DATA_PATH) as f:
    lines = f.read().split("\n")
words = [parts[1] for parts in (l.split(" ") for l in lines) if len(parts) == 2]

# vocab: '.' (index 0) marks both the start and the end of a name
chars = sorted(set("".join(words)))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda idxs: "".join(itos[i] for i in idxs)

# Longest example is ".name" (start token + all letters), which must fit in block_size
block_size = max(len(w) for w in words) + 1

# train / val / test splits, by name, in file order like the notebook
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
splits = {"train": words[:n1], "val": words[n1:n2], "test": words[n2:]}
train_tokens = sum(len(w) + 1 for w in splits["train"])   # every letter + the end '.' is a prediction
print(f"names: {len(words):,} (train {n1:,}, val {n2 - n1:,}, test {len(words) - n2:,}), "
      f"vocab: {vocab_size}, block_size: {block_size}, train tokens: {train_tokens:,}")


def make_batch(names):
    """Whole names, padded to the longest in the batch. x starts with '.', y ends with '.'."""
    T = max(len(w) for w in names) + 1
    x = torch.zeros((len(names), T), dtype=torch.long)          # pad x with '.', never seen by real tokens
    y = torch.full((len(names), T), IGNORE, dtype=torch.long)
    for row, w in enumerate(names):
        ids = encode(w)
        x[row, :len(ids) + 1] = torch.tensor([0] + ids)
        y[row, :len(ids) + 1] = torch.tensor(ids + [0])
    return x.to(device), y.to(device)


def get_batch(split, batch_size):
    names = splits[split]
    ix = torch.randint(0, len(names), (batch_size,)).tolist()
    return make_batch([names[i] for i in ix])


@torch.no_grad
def evaluate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size)
            logits, loss = m(x, y)
            losses[k] = loss
        out[split] = losses.mean()
    m.train()
    return out


@torch.no_grad
def full_split_loss(split, chunk=4096):
    """Exact loss over every character of a split, averaged per character like the WaveNet notebook."""
    m.eval()
    names = splits[split]
    total, count = 0.0, 0
    for i in range(0, len(names), chunk):
        x, y = make_batch(names[i:i+chunk])
        logits, _ = m(x)
        total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum").item()
        count += (y != IGNORE).sum().item()
    m.train()
    return total / count


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

        q = self.query(x) # x = (B,T,n_emb), self.query = (n_emb, head_size), q = (B, T, head_size)
        k = self.keys(x) # x = (B,T,n_emb), self.keys = (n_emb, head_size), k = (B, T, head_size)

        wei = q @ k.transpose(-2,-1)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.values(x) # (B, T, head_size)

        out = wei @ v # (B, T, head_size)
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
        self.ln1 = nn.LayerNorm(n_emb)                 # Layer normalization before self-attention
        self.ln2 = nn.LayerNorm(n_emb)                 # Layer normalization before feed-forward

    def forward(self, x):
        x = x + self.multi_attension(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class Transormer(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)  # (vocab_size, n_emb)
        self.positional_embeddings = nn.Embedding(block_size, n_emb) # (block_size, n_emb)

        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, vocab_size) # (n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.embedding_layer(idx) # (B, T, n_emb)
        positional_emb = self.positional_embeddings(torch.arange(T, device=device)) # (T, n_emb)

        x = token_emb + positional_emb # (B, T, n_emb)

        attention = self.blocks(x)

        x = self.ln_f(attention)
        logits = self.linear(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)   # ignores IGNORE (-100) targets
        return logits, loss

    @torch.no_grad
    def generate_name(self, prefix="", temperature=1.0):
        """One name, one character at a time, until the model writes '.' (end)."""
        idx = torch.tensor([[0] + encode(prefix)], dtype=torch.long, device=device)
        while idx.shape[1] < block_size:
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            if ix.item() == 0:
                break
            idx = torch.cat((idx, ix), dim=1)
        return decode(idx[0, 1:].tolist())


# training the model
def save_checkpoint(m, opt, step, path):
    torch.save({
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "stoi": stoi,
        "itos": itos,
        "config": {
            "n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
            "block_size": block_size, "vocab_size": vocab_size, "dropout": dropout,
        },
    }, path)
    print(f"Model saved to {path}")


def train(m, opt, start_step, num_iters):
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params:,} ({train_tokens / n_params:.1f} train tokens per param)")

    m.train()
    best_val, bad_evals = float("inf"), 0
    for step in range(start_step, start_step + num_iters):
        x, y = get_batch("train", batch_size)

        logits, loss = m(x, y)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % eval_interval == 0:
            train_loss = evaluate_loss()["train"]
            val_loss = full_split_loss("val")   # exact, over every val name: used for early stopping
            save_checkpoint(m, opt, step, HERE / f"transformer_names_{step}.pt")   # every eval, to resume/compare
            if val_loss < best_val:
                best_val, bad_evals = val_loss, 0
                save_checkpoint(m, opt, step, CKPT_PATH)   # transformer_names.pt = best model so far
                mark = " (best)"
            else:
                bad_evals += 1
                mark = f" (no improvement {bad_evals}/{patience})"
            print(f"Loss at step {step}: train {train_loss:.4f}, val {val_loss:.4f}{mark}")
            if bad_evals >= patience:
                print(f"Early stopping: val loss hasn't improved for {patience * eval_interval} steps")
                break

    # Report on the best model, not the last one
    best = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    m.load_state_dict(best["model_state_dict"])
    print(f"\nBest model: step {best['step']}, val {best_val:.4f}")

    # Exact losses over the full splits, comparable with the WaveNet notebook
    for split in ["train", "val", "test"]:
        print(f"{split} loss: {full_split_loss(split):.4f}")

    m.eval()
    print("\nSample names:")
    for _ in range(20):
        print(m.generate_name().capitalize())
    m.train()
    return m


def train_from_scratch(num_iters=max_iters):
    global m   # evaluate_loss() uses the global m
    m = Transormer(n_emb).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    return train(m, opt, 0, num_iters)


def train_from_checkpoint(path=CKPT_PATH, num_iters=max_iters):
    global m   # evaluate_loss() uses the global m
    ckpt = torch.load(path, map_location=device, weights_only=True)

    config = {"n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
              "block_size": block_size, "vocab_size": vocab_size, "dropout": dropout}
    if ckpt["config"] != config:
        raise ValueError(f"Checkpoint config {ckpt['config']} does not match script config {config}")
    if ckpt["stoi"] != stoi:
        raise ValueError("Checkpoint vocabulary does not match the training data")

    m = Transormer(n_emb).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resuming from {path} at step {ckpt['step']}")
    return train(m, opt, ckpt["step"], num_iters)


if __name__ == "__main__":
    # train_from_scratch()
    train_from_checkpoint()
