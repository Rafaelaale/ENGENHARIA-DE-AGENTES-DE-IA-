import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
RESULTADOS = ROOT / "resultados"
MODELOS = ROOT / "modelos"

RESULTADOS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

torch.manual_seed(42)
random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AULA 09 - TREINAMENTO DO TRANSFORMER")
print("=" * 60)
print("Dispositivo:", device)

# 1 - CARREGAMENTO DO DATASET

arquivo = DADOS / "wikipedia_pt_limpo_5000.txt"

if not arquivo.exists():
    raise FileNotFoundError(f"Dataset ausente: {arquivo}")

text = arquivo.read_text(encoding="utf-8")

print("\n1 - DATASET")
print("Caracteres:", len(text))

# 2 - TOKENIZADOR BERT PORTUGUES

nome_tokenizer = "neuralmind/bert-large-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(nome_tokenizer)

text_id = tokenizer.encode(
    text,
    add_special_tokens=False
)

print("\n2 - TOKENIZACAO")
print("Tokens:", len(text_id))

data = torch.tensor(text_id, dtype=torch.long)

n_train = int(0.8 * len(data))

train_data = data[:n_train]
val_data = data[n_train:]

print("Treino:", len(train_data))
print("Validacao:", len(val_data))

# 3 - HIPERPARAMETROS

batch_size = 8
block_size = min(32, len(val_data) - 2)
learning_rate = 3e-4
max_iters = 80
eval_interval = 20
save_interval = 40
eval_iters = 5

d_model = 64
n_heads = 4
n_layers = 2
dropout = 0.1

vocab_size = tokenizer.vocab_size

if block_size < 2:
    raise ValueError("Dataset insuficiente para treinamento.")

config = {
    "vocab_size": vocab_size,
    "d_model": d_model,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "block_size": block_size,
    "dropout": dropout,
    "batch_size": batch_size,
    "learning_rate": learning_rate
}

print("\n3 - CONFIGURACAO")
print(json.dumps(config, indent=2))

# 4 - BLOCO TRANSFORMER

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        B, T, C = x.shape

        mask = torch.triu(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        normalized = self.ln1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False
        )

        x = x + attention_output
        x = x + self.ffn(self.ln2(x))

        return x

# 5 - MODELO AUTORREGRESSIVO

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
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):

        B, T = idx.shape

        positions = torch.arange(
            T, device=idx.device
        )

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

        return logits, loss, None

model = TransformerLanguageModel().to(device)

parametros = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n4 - MODELO CRIADO")
print("Parametros treinaveis:", parametros)

# 6 - BATCHES

def get_batch(split, batch_size=batch_size, block_size=block_size):

    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
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

# 7 - AVALIACAO

@torch.no_grad()
def estimate_loss(model, eval_iters=eval_iters):

    model.eval()
    out = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out

@torch.no_grad()
def calculate_perplexity(model, split, n_batches=5):

    model.eval()
    losses = []

    for _ in range(n_batches):
        X, Y = get_batch(split)
        _, loss, _ = model(X, Y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    model.train()

    return math.exp(avg_loss) if avg_loss < 700 else float("inf")

# 8 - OTIMIZADOR E SCHEDULER

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=40,
    T_mult=2
)

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

# 9 - TREINAMENTO

print("\n5 - INICIANDO TREINAMENTO")

train_losses = []
val_losses = []
val_ppls = []
steps = []

best_val_loss = float("inf")

model.train()

def salvar_checkpoint(nome, iteracao, train_loss, val_loss, ppl):

    torch.save({
        "iter": iteracao,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_perplexity": ppl,
        "tokenizer": nome_tokenizer
    }, MODELOS / nome)

for iteration in range(max_iters):

    xb, yb = get_batch("train")

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        enabled=use_amp
    ):
        logits, loss, _ = model(xb, yb)

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    scaler.step(optimizer)
    scaler.update()

    scheduler.step(iteration + 1)

    if (
        iteration % eval_interval == 0
        or iteration == max_iters - 1
    ):

        losses = estimate_loss(model)

        train_loss = losses["train"]
        val_loss = losses["val"]

        val_ppl = calculate_perplexity(
            model, "val"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        steps.append(iteration + 1)

        print(
            f"Iter {iteration + 1:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PPL: {val_ppl:.2f}",
            flush=True
        )

        if math.isfinite(val_loss) and val_loss < best_val_loss:

            best_val_loss = val_loss

            salvar_checkpoint(
                "aula09_best_model.pt",
                iteration + 1,
                train_loss,
                val_loss,
                val_ppl
            )

            print("Novo melhor modelo salvo!", flush=True)

    if (
        (iteration + 1) % save_interval == 0
        or iteration == max_iters - 1
    ):

        salvar_checkpoint(
            f"aula09_checkpoint_{iteration + 1}.pt",
            iteration + 1,
            train_loss,
            val_loss,
            val_ppl
        )

        print(
            f"Checkpoint {iteration + 1} salvo.",
            flush=True
        )

# 10 - GRAFICOS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(steps, train_losses, label="Train Loss")
ax1.plot(steps, val_losses, label="Val Loss")

ax1.set_xlabel("Iteracoes")
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()
ax2.plot(steps, val_ppls, label="Perplexidade", linestyle="--")
ax2.set_ylabel("Perplexidade")

fig.suptitle("Treinamento do Transformer")
fig.tight_layout()

grafico = RESULTADOS / "aula09_curvas_treinamento.png"

fig.savefig(grafico, dpi=150)
plt.close(fig)

# 11 - RELATORIO

relatorio = {
    "aula": 9,
    "tema": "Treinamento do Transformer",
    "device": str(device),
    "tokens": len(text_id),
    "parametros": parametros,
    "config": config,
    "iteracoes": max_iters,
    "steps": steps,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_perplexities": val_ppls,
    "best_val_loss": best_val_loss,
    "grafico": grafico.name,
    "treinamento_realizado": True
}

relatorio_path = RESULTADOS / "aula09_treinamento.json"

relatorio_path.write_text(
    json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("AULA 09 - TREINAMENTO CONCLUIDO")
print("=" * 60)
print("Melhor validation loss:", best_val_loss)
print("Grafico:", grafico)
print("Relatorio:", relatorio_path)
print("Modelos:", MODELOS)
