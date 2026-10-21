import os

import gradio as gr
import torch
import torch.nn.functional as F

try:
    import spaces  # available on Hugging Face ZeroGPU Spaces
    gpu = spaces.GPU
except ImportError:  # running locally
    gpu = lambda fn: fn


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out


class Embedding:
    def __init__(self, num_embeddings, embeddings_size):
        self.weight = torch.randn((num_embeddings, embeddings_size))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out


class Flatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        batch_size, block_size, n_emb = x.shape
        x = x.view(batch_size, block_size // self.n, n_emb * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out


class Linear:
    def __init__(self, fanin, fanout, bias=True):
        self.weight = torch.randn((fanin, fanout)) / (fanin**0.5)
        self.bias = torch.randn(fanout) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # inference only: always use running stats
        xhat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavenet.pt")
print("model path:", model_path)
ckpt = torch.load(model_path, map_location="cpu")
itos = ckpt["itos"]
stoi = {s: i for i, s in itos.items()}
block_size = ckpt["block_size"]
vocab_size = len(itos)

# Use the SAME hyperparameters and layer order as in training
n_emb = 50
n_hidden = 300

model = Sequential([
    Embedding(vocab_size, n_emb),
    Flatten(2), Linear(n_emb * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Flatten(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Flatten(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

for key, t in ckpt["state"].items():
    i, name = key.split(".")
    setattr(model.layers[int(i)], name, t)


@torch.no_grad()
def generate_one(prefix="", temperature=1.0, max_len=30):
    context = [0] * block_size
    out = []
    for ch in prefix:
        context = context[1:] + [stoi[ch]]
        out.append(stoi[ch])
    while len(out) < max_len:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits / temperature, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:  # end-of-name token
            break
        context = context[1:] + [ix]
        out.append(ix)
    return "".join(itos[i] for i in out)


# ZeroGPU Spaces require at least one @spaces.GPU function; the model itself runs on CPU
@gpu
def generate_names(num_names, prefix, temperature):
    prefix = (prefix or "").strip().lower()
    unknown = sorted({ch for ch in prefix if ch not in stoi or stoi[ch] == 0})
    if unknown:
        raise gr.Error(f"Unsupported characters in prefix: {' '.join(unknown)}")
    names = [generate_one(prefix, temperature).capitalize() for _ in range(int(num_names))]
    return "\n".join(names)


with gr.Blocks() as iface:
    gr.Markdown(
        """
        # 🇮🇳 Indian Name Generator

        A character-level language model trained on Indian names, based on the **WaveNet architecture**.
        It generates new, name-like words one character at a time.

        ### Model details:
        - **All layers are custom built** from scratch in PyTorch (Embedding, Flatten, Linear,
          BatchNorm1d, Tanh, Sequential) — no `torch.nn` layers used
        - **WaveNet architecture**: characters are fused hierarchically, 2 at a time, over a
          context of 8 previous characters
        - **Embedding size (n_emb):** 50
        - **Hidden units (n_hidden):** 300
        - **Total parameters:** 401,277
        - **Dataset:** [swami93/indian-names-1.5M](https://huggingface.co/datasets/swami93/indian-names-1.5M)
        - **Training notebook:** [Kaggle — predict-indian-names-wavenet-arch](https://www.kaggle.com/code/eluriadarsh/predict-indian-names-wavenet-arch)

        ### Performance (cross-entropy loss):
        | Split | Loss |
        |---|---|
        | Training | 1.90 |
        | Validation | 2.01 |
        | Test | 2.01 |

        ### How to use:
        - Choose how many names to generate
        - Optionally give a starting prefix (e.g. `pra`, `su`, `ka`)
        - **Temperature**: lower = safer, more common-sounding names; higher = more creative
        """
    )

    with gr.Row():
        with gr.Column():
            num_names = gr.Slider(1, 50, value=10, step=1, label="Number of names")
            prefix = gr.Textbox(label="Starts with (optional)", placeholder="e.g. pra")
            temperature = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Temperature")
            submit_btn = gr.Button("✨ Generate Names", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Generated names", lines=15)

    gr.Examples(
        examples=[[10, "", 1.0], [10, "pra", 0.8], [10, "su", 1.0], [10, "ka", 1.3]],
        inputs=[num_names, prefix, temperature],
        label="Try an example",
    )

    submit_btn.click(fn=generate_names, inputs=[num_names, prefix, temperature], outputs=output)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
