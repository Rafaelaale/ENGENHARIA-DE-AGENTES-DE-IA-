import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
MODELOS = ROOT / "modelos"
RESULTADOS = ROOT / "resultados"

MODELOS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EXPERIMENTO 02 - TRANSFORMER")
print("=" * 60)
print("Dispositivo:", device)

# 1 - DATASET E TOKENIZADOR

texto = (
    DADOS / "wikipedia_pt_limpo_5000.txt"
).read_text(encoding="utf-8")

tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-large-portuguese-cased"
)

ids = tokenizer.encode(
    texto,
    add_special_tokens=False
)

data = torch.tensor(ids, dtype=torch.long)

n_train = int(len(data) * 0.8)

train_data = data[:n_train]
val_data = data[n_train:]

print("\nTokens:", len(data))
print("Treino:", len(train_data))
print("Validacao:", len(val_data))

# 2 - CONFIGURACAO

config = {
    "vocab_size": tokenizer.vocab_size,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "block_size": 32,
    "dropout": 0.1,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "max_iters": 200
}

vocab_size = config["vocab_size"]
d_model = config["d_model"]
block_size = config["block_size"]

# 3 - BLOCO TRANSFORMER

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(config["dropout"])
        )

    def forward(self, x):

        T = x.shape[1]

        mask = torch.triu(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        z = self.ln1(x)

        attention, _ = self.attention(
            z, z, z,
            attn_mask=mask,
            need_weights=False
        )

        x = x + attention
        x = x + self.ffn(self.ln2(x))

        return x

# 4 - MODELO

class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.position_embedding = nn.Embedding(
            block_size, d_model
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(config["n_layers"])
            ]
        )

        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        # Inicializacao corrigida
        self.apply(self._init_weights)

        # Compartilhamento dos pesos
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        positions = torch.arange(
            T, device=idx.device
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

        return logits, loss

model = TransformerLanguageModel().to(device)

print("\nParametros treinaveis:", sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
))

# 5 - BATCHES

def get_batch(split):

    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (config["batch_size"],)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# 6 - AVALIACAO

@torch.no_grad()
def avaliar():

    model.eval()

    resultado = {}

    for split in ("train", "val"):

        losses = []

        for _ in range(5):

            x, y = get_batch(split)

            _, loss = model(x, y)

            losses.append(loss.item())

        resultado[split] = sum(losses) / len(losses)

    model.train()

    return resultado

# 7 - TREINAMENTO

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"]
)

historico = []
best_val_loss = float("inf")

print("\nINICIANDO TREINAMENTO")

for iteration in range(1, config["max_iters"] + 1):

    model.train()

    x, y = get_batch("train")

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    if iteration == 1 or iteration % 20 == 0:

        metricas = avaliar()

        train_loss = metricas["train"]
        val_loss = metricas["val"]

        ppl = math.exp(val_loss)

        historico.append({
            "iteracao": iteration,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": ppl
        })

        print(
            f"Iter {iteration:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PPL: {ppl:.2f}",
            flush=True
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "model": model.state_dict(),
                "config": config,
                "tokenizer": "neuralmind/bert-large-portuguese-cased",
                "iter": iteration,
                "val_loss": val_loss
            }, MODELOS / "experimento02_best_model.pt")

# 8 - GRAFICO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = [x["iteracao"] for x in historico]
train_losses = [x["train_loss"] for x in historico]
val_losses = [x["val_loss"] for x in historico]

plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="Treino")
plt.plot(steps, val_losses, label="Validacao")

plt.xlabel("Iteracoes")
plt.ylabel("Loss")
plt.title("Experimento 02 - Transformer")
plt.legend()
plt.grid(True, alpha=0.3)

grafico = RESULTADOS / "experimento02_curvas.png"

plt.savefig(grafico, dpi=150)
plt.close()

# 9 - RELATORIO

relatorio = {
    "experimento": 2,
    "config": config,
    "tokens": len(data),
    "best_val_loss": best_val_loss,
    "historico": historico,
    "grafico": grafico.name
}

arquivo_relatorio = (
    RESULTADOS / "experimento02_resultados.json"
)

arquivo_relatorio.write_text(
    json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("EXPERIMENTO 02 CONCLUIDO")
print("=" * 60)
print("Melhor loss validacao:", best_val_loss)
print("Grafico:", grafico)
print("Relatorio:", arquivo_relatorio)
print("Checkpoint:", MODELOS / "experimento02_best_model.pt")

