import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELOS = ROOT / "modelos"
RESULTADOS = ROOT / "resultados"
DADOS = ROOT / "dados"

RESULTADOS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AULA 10 - AVALIACAO E VALIDACAO DO TRANSFORMER")
print("=" * 60)
print("Dispositivo:", device)

# 1 - CARREGAR CHECKPOINT DA AULA 09

checkpoint_path = MODELOS / "aula09_best_model.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint nao encontrado: {checkpoint_path}"
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

config = checkpoint["config"]

vocab_size = config["vocab_size"]
d_model = config["d_model"]
n_heads = config["n_heads"]
n_layers = config["n_layers"]
block_size = config["block_size"]
dropout = config["dropout"]

print("\n1 - CHECKPOINT CARREGADO")
print("Iteracao:", checkpoint["iter"])
print("Validation loss:", checkpoint["val_loss"])

# 2 - TOKENIZADOR

tokenizer_name = checkpoint.get(
    "tokenizer",
    "neuralmind/bert-large-portuguese-cased"
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

if len(tokenizer) > vocab_size:
    raise ValueError(
        "O vocabulario do tokenizer excede o vocabulario do modelo."
    )

print("\n2 - TOKENIZADOR")
print(tokenizer_name)

# 3 - ARQUITETURA COMPATIVEL COM AULA 09

class TransformerBlock(nn.Module):

    def __init__(self):
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

        T = x.shape[1]

        mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device
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
            *[TransformerBlock() for _ in range(n_layers)]
        )

        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > block_size:
            raise ValueError("Sequencia excedeu block_size.")

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

        return logits, loss, None


model = TransformerLanguageModel()

model.load_state_dict(
    checkpoint["model"],
    strict=True
)

model = model.to(device)
model.eval()

print("\n3 - MODELO RESTAURADO")
print("Camadas:", n_layers)
print("Cabecas de atencao:", n_heads)
print("Block size:", block_size)

# 4 - GERACAO AUTORREGRESSIVA

@torch.no_grad()
def generate(
    model,
    prompt,
    temperature=1.0,
    max_new_tokens=20,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
):

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError("Prompt sem tokens.")

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        context = x[:, -block_size:]

        logits, _, _ = model(context)

        next_logits = logits[:, -1, :].clone()

        # Penalidade de repeticao

        for token_id in set(x[0].tolist()):

            value = next_logits[0, token_id]

            if value < 0:
                next_logits[0, token_id] *= repetition_penalty
            else:
                next_logits[0, token_id] /= repetition_penalty

        temperature = max(
            0.1,
            min(2.0, float(temperature))
        )

        next_logits = next_logits / temperature

        # Top-K

        k = min(top_k, next_logits.shape[-1])

        top_values, _ = torch.topk(
            next_logits, k
        )

        threshold = top_values[:, -1:]

        next_logits = next_logits.masked_fill(
            next_logits < threshold,
            float("-inf")
        )

        # Top-P

        sorted_logits, sorted_indices = torch.sort(
            next_logits,
            descending=True
        )

        sorted_probs = F.softmax(
            sorted_logits,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        remove = cumulative_probs - sorted_probs >= top_p

        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf")
        )

        probs = F.softmax(
            sorted_logits,
            dim=-1
        )

        sampled = torch.multinomial(
            probs,
            num_samples=1
        )

        next_token = sorted_indices.gather(
            -1, sampled
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )

        if tokenizer.sep_token_id is not None:
            if next_token.item() == tokenizer.sep_token_id:
                break

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=True
    )


# 5 - TESTE DE PROMPTS E TEMPERATURAS

test_prompts = [
    "Astronomia é",
    "O gato subiu no telhado",
    "Ciência de Dados",
    "Em 2001",
    "O Brasil"
]

temperatures = [0.8, 1.0, 1.2]

geracoes = []

print("\n4 - GERACAO DE TEXTOS")

for prompt in test_prompts:

    print("\nPrompt:", prompt)

    for temp in temperatures:

        generated = generate(
            model,
            prompt,
            temperature=temp,
            max_new_tokens=20,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1
        )

        print(f"\nTemperatura {temp}")
        print(generated, flush=True)

        geracoes.append({
            "prompt": prompt,
            "temperatura": temp,
            "texto": generated
        })

# 6 - VISUALIZACAO DA ATENCAO

print("\n5 - ANALISANDO ATENCAO")

sample_text = (
    "O gato subiu no telhado ontem "
    "à noite sozinho porque estava assustado."
)

sample_ids = tokenizer.encode(
    sample_text,
    add_special_tokens=False
)[:block_size]

x_sample = torch.tensor(
    [sample_ids],
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    T = x_sample.shape[1]

    positions = torch.arange(
        T, device=device
    )

    x = (
        model.token_embedding(x_sample)
        + model.position_embedding(positions)
    )

    for block in list(model.blocks)[:-1]:
        x = block(x)

    last_block = model.blocks[-1]

    normalized = last_block.ln1(x)

    mask = torch.triu(
        torch.ones(
            T, T,
            dtype=torch.bool,
            device=device
        ),
        diagonal=1
    )

    _, attention_weights = last_block.attention(
        normalized,
        normalized,
        normalized,
        attn_mask=mask,
        need_weights=True,
        average_attn_weights=False
    )

attention_weights = attention_weights[0].cpu().numpy()

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

axes = axes.flatten()

for head in range(n_heads):

    im = axes[head].imshow(
        attention_weights[head],
        cmap="Blues",
        aspect="auto"
    )

    axes[head].set_title(
        f"Cabeca de Atencao {head + 1}"
    )

    axes[head].set_xlabel("Token observado")
    axes[head].set_ylabel("Token consultando")

    plt.colorbar(
        im,
        ax=axes[head],
        fraction=0.046
    )

plt.suptitle("Padroes de Atencao - Aula 10")
plt.tight_layout()

attention_path = (
    RESULTADOS / "aula10_padroes_atencao.png"
)

plt.savefig(
    attention_path,
    dpi=150
)

plt.close()

print("Grafico salvo:", attention_path)

# 7 - CALCULAR PERPLEXIDADE FINAL

print("\n6 - PERPLEXIDADE FINAL")

text = (
    DADOS / "wikipedia_pt_limpo_5000.txt"
).read_text(encoding="utf-8")

text_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

data = torch.tensor(
    text_ids,
    dtype=torch.long
)

n_train = int(0.8 * len(data))

train_data = data[:n_train]
val_data = data[n_train:]

@torch.no_grad()
def calculate_perplexity(source, n_batches=10):

    model.eval()
    losses = []

    for _ in range(n_batches):

        ix = torch.randint(
            len(source) - block_size,
            (8,)
        )

        x = torch.stack([
            source[i:i + block_size]
            for i in ix
        ]).to(device)

        y = torch.stack([
            source[i + 1:i + block_size + 1]
            for i in ix
        ]).to(device)

        _, loss, _ = model(x, y)

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    return math.exp(avg_loss) if avg_loss < 700 else float("inf")


train_ppl = calculate_perplexity(train_data)
val_ppl = calculate_perplexity(val_data)

print(f"Perplexidade Treino: {train_ppl:.4e}")
print(f"Perplexidade Validacao: {val_ppl:.4e}")

# 8 - SALVAR CHECKPOINT FINAL

final_checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": config,
    "tokenizer": tokenizer_name,
    "vocab_size": vocab_size,
    "training_history": {
        "best_val_loss": checkpoint["val_loss"],
        "final_train_ppl": train_ppl,
        "final_val_ppl": val_ppl,
        "total_iterations": checkpoint["iter"]
    },
    "hyperparameters": {
        "batch_size": config["batch_size"],
        "block_size": block_size,
        "learning_rate": config["learning_rate"]
    }
}

final_path = MODELOS / "aula10_transformer_final.pt"

torch.save(
    final_checkpoint,
    final_path
)

# 9 - RELATORIO JSON

relatorio = {
    "aula": 10,
    "tema": "Avaliacao e Validacao",
    "checkpoint_origem": checkpoint_path.name,
    "prompts": geracoes,
    "perplexidade_treino": train_ppl,
    "perplexidade_validacao": val_ppl,
    "grafico_atencao": attention_path.name,
    "checkpoint_final": final_path.name,
    "validacao_executada": True
}

relatorio_path = (
    RESULTADOS / "aula10_avaliacao_transformer.json"
)

relatorio_path.write_text(
    json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("AULA 10 - AVALIACAO CONCLUIDA")
print("=" * 60)
print("Checkpoint:", final_path)
print("Grafico:", attention_path)
print("Relatorio:", relatorio_path)

