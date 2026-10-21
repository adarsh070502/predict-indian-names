import torch
import torch.nn.functional as F

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params

class Embedding:
    def __init__(self, num_embeddings, embeddings_size):
        self.weight = torch.randn((num_embeddings, embeddings_size))

    def __call__(self, ix):
        self.out = self.weight[ix]
        # print(f"Embedding layer: {self.out.shape}")
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        batch_size, block_size, n_emb = x.shape

        x = x.view(batch_size, block_size//self.n, n_emb*self.n)

        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        # print(f"Flatten layer: {self.out.shape}")
        return self.out

    def parameters(self):
        return []


class Linear:
    def __init__(self, fanin, fanout, bias=True):
        self.weight = torch.randn((fanin, fanout)) / (fanin**0.5)
        self.bias = torch.randn(fanout) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        # print(f"Linear layer: {self.out.shape}")
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])


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
        if self.training:
            dim = None
            if x.dim() == 2:
                dim = 0
            elif x.dim() == 3:
                dim = (0,1)

            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        with torch.no_grad():
            if self.training:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        # print(f"Batch Norm layer: {self.out.shape}")
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        # print(f"TanH layer: {self.out.shape}")
        return self.out
    def parameters(self):
        return []


@torch.no_grad()
def generate(n=20):
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))



ckpt = torch.load('wavenet.pt', map_location='cpu')
itos = ckpt['itos']
block_size = ckpt['block_size']
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

for key, t in ckpt['state'].items():
    i, name = key.split('.')
    setattr(model.layers[int(i)], name, t)

for layer in model.layers:
    layer.training = False
    
generate()