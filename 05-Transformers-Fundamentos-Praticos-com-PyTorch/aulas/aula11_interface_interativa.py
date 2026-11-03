import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "modelos" / "aula10_transformer_final.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    def __init__(self, config):
        super().__init__()

        self.config = config

        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        block_size = config["block_size"]
        dropout = config["dropout"]

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.position_embedding = nn.Embedding(
            block_size, d_model
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model, n_heads, dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):

        B, T = idx.shape

        block_size = self.config["block_size"]

        if T > block_size:
            raise ValueError("Contexto maior que block_size")

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

        return logits


def carregar_modelo(model_path=MODEL_PATH):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint["config"]

    model = TransformerLanguageModel(config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint["tokenizer"]
    )

    if len(tokenizer) > config["vocab_size"]:
        raise ValueError("Tokenizer incompatível com o modelo")

    return model, tokenizer


@torch.no_grad()
def gerar_texto(
    model,
    tokenizer,
    prompt,
    max_tokens=50,
    temperatura=0.8,
    top_k=50,
    top_p=0.9
):

    ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not ids:
        raise ValueError("Prompt sem tokens validos")

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    block_size = model.config["block_size"]

    for _ in range(max_tokens):

        contexto = x[:, -block_size:]

        logits = model(contexto)

        logits = logits[:, -1, :].clone()
        logits = logits / temperatura

        k = min(top_k, logits.shape[-1])

        valores, _ = torch.topk(logits, k)

        logits = logits.masked_fill(
            logits < valores[:, -1:],
            float("-inf")
        )

        sorted_logits, sorted_indices = torch.sort(
            logits,
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

        if (
            tokenizer.sep_token_id is not None
            and next_token.item() == tokenizer.sep_token_id
        ):
            break

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=True
    )


def interface_interativa():

    print("=" * 60)
    print("AULA 11 - INTERFACE INTERATIVA TRANSFORMER")
    print("=" * 60)

    print("Dispositivo:", device)
    print("Carregando modelo...")

    try:
        model, tokenizer = carregar_modelo()

    except Exception as erro:
        print("ERRO AO CARREGAR MODELO:")
        print(erro)
        return

    print("\nMODELO CARREGADO COM SUCESSO!")
    print("Digite 'sair' para encerrar.")

    while True:

        try:
            prompt = input("\nDigite seu prompt: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando interface.")
            break

        if prompt.lower() == "sair":
            print("Interface encerrada.")
            break

        if not prompt:
            print("Digite um texto valido.")
            continue

        entrada_tokens = input(
            "Maximo de tokens [50]: "
        ).strip()

        try:
            max_tokens = int(entrada_tokens) if entrada_tokens else 50

            if not 1 <= max_tokens <= 200:
                max_tokens = 50

        except ValueError:
            max_tokens = 50

        entrada_temperatura = input(
            "Temperatura 0.1 a 2.0 [0.8]: "
        ).strip()

        try:
            temperatura = (
                float(entrada_temperatura.replace(",", "."))
                if entrada_temperatura else 0.8
            )

            if not 0.1 <= temperatura <= 2.0:
                temperatura = 0.8

        except ValueError:
            temperatura = 0.8

        print("\nGerando texto...\n")

        try:
            resultado = gerar_texto(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperatura=temperatura
            )

            print("-" * 60)
            print(resultado)
            print("-" * 60)

        except Exception as erro:
            print("Erro na geracao:", erro)


if __name__ == "__main__":
    interface_interativa()
