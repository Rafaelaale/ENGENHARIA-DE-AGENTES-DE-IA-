from pathlib import Path
import sys
import re
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

pasta = Path(__file__).resolve().parent
origem = Path(sys.argv[1])

dados = torch.load(
    origem,
    map_location="cpu",
    weights_only=False
)

X_original = dados["X"]
y_original = dados["y"]
comprimentos = dados["comprimentos"]
vocabulario = dados["vocabulario"]
pesos_embedding = dados["embedding_weights"]

PAD = vocabulario["<PAD>"]
UNK = vocabulario["<UNK>"]

EMBED_DIM = 100
HIDDEN_SIZE = 256
OUTPUT_SIZE = 2
EPOCAS = 30
BATCH_SIZE = 3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"TESTE {numero} FALHOU: {descricao}"
    resultados.append(
        f"{numero}. {descricao}: APROVADO"
    )
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 70)
print("LABORATORIO - CLASSIFICANDO SEQUENCIAS")
print("=" * 70)
print("Dispositivo:", device)

# Recuperar sequencias originais.
sequencias = [
    X_original[:int(comprimentos[i]), i].clone()
    for i in range(X_original.shape[1])
]

def montar_batch(indices):
    indices = sorted(
        indices,
        key=lambda i: len(sequencias[i]),
        reverse=True
    )

    textos = [sequencias[i] for i in indices]

    tamanhos = torch.tensor(
        [len(texto) for texto in textos],
        dtype=torch.long
    )

    X = pad_sequence(
        textos,
        batch_first=False,
        padding_value=PAD
    ).to(device)

    y = y_original[indices].to(device)

    return X, tamanhos, y

class ClassificadorSentimentos(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        embed_vectors,
        ind_unk,
        ind_pad,
        hidden_size,
        output_size
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=ind_pad
        )

        with torch.no_grad():
            self.embedding.weight.copy_(embed_vectors)
            self.embedding.weight[ind_unk].zero_()
            self.embedding.weight[ind_pad].zero_()

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False
        )

        self.linear = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, X, tamanhos):
        embed = self.embedding(X)

        hidden_inicial = torch.zeros(
            1,
            X.size(1),
            self.hidden_size,
            device=X.device,
            dtype=embed.dtype
        )

        packed_input = pack_padded_sequence(
            embed,
            tamanhos.cpu(),
            batch_first=False,
            enforce_sorted=True
        )

        _, hidden_final = self.rnn(
            packed_input,
            hidden_inicial
        )

        # Selecionar a unica camada recorrente.
        # Nao usar squeeze() sem dimensao, pois B pode ser 1.
        representacao = hidden_final[-1]

        logits = self.linear(representacao)

        return logits

modelo = ClassificadorSentimentos(
    vocab_size=len(vocabulario),
    embed_dim=EMBED_DIM,
    embed_vectors=pesos_embedding,
    ind_unk=UNK,
    ind_pad=PAD,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE
).to(device)

testar(
    1,
    "Modelo possui Embedding, GRU e Linear",
    isinstance(modelo.embedding, nn.Embedding)
    and isinstance(modelo.rnn, nn.GRU)
    and isinstance(modelo.linear, nn.Linear)
)

X_teste, tamanhos_teste, y_teste = montar_batch([0, 1, 2])

with torch.no_grad():
    logits_teste = modelo(X_teste, tamanhos_teste)

testar(
    2,
    "Saida possui dimensao B x 2",
    logits_teste.shape == (3, 2)
)

testar(
    3,
    "Rotulos possuem formato correto",
    y_teste.shape == (3,)
    and y_teste.dtype == torch.long
)

# CrossEntropyLoss recebe logits, nao probabilidades.
criterio = nn.CrossEntropyLoss()

loss_inicial = criterio(logits_teste, y_teste)

testar(
    4,
    "CrossEntropyLoss calcula perda valida",
    torch.isfinite(loss_inicial).item()
)

probabilidades = F.softmax(
    logits_teste,
    dim=1
)

testar(
    5,
    "Softmax produz probabilidades que somam 1",
    torch.allclose(
        probabilidades.sum(dim=1),
        torch.ones(3, device=device),
        atol=1e-6
    )
)

# Testar batch com uma unica avaliacao.
X_um, tamanhos_um, _ = montar_batch([0])

with torch.no_grad():
    logits_um = modelo(X_um, tamanhos_um)

testar(
    6,
    "Batch de tamanho 1 preserva dimensao 1 x 2",
    logits_um.shape == (1, 2)
)

# Treinamento didatico com seis frases.
# Nao e uma avaliacao real de generalizacao.
otimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=0.003
)

historico = []

print("\nTREINAMENTO DIDATICO")
print("Amostras:", len(sequencias))
print("Epocas:", EPOCAS)

for epoca in range(1, EPOCAS + 1):
    modelo.train()

    ordem = list(range(len(sequencias)))
    random.shuffle(ordem)

    perda_total = 0.0
    acertos = 0
    total = 0

    for inicio in range(0, len(ordem), BATCH_SIZE):
        indices = ordem[inicio:inicio + BATCH_SIZE]

        X, tamanhos, y = montar_batch(indices)

        otimizador.zero_grad()

        logits = modelo(X, tamanhos)
        loss = criterio(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(),
            max_norm=1.0
        )

        otimizador.step()

        quantidade = len(indices)

        perda_total += loss.item() * quantidade
        acertos += (
            (logits.argmax(dim=1) == y)
            .sum()
            .item()
        )
        total += quantidade

    perda_media = perda_total / total
    acuracia = acertos / total

    historico.append(
        (epoca, perda_media, acuracia)
    )

    if epoca == 1 or epoca % 5 == 0:
        print(
            f"Epoca {epoca:02d}/{EPOCAS} | "
            f"Loss: {perda_media:.4f} | "
            f"Acuracia treino: {acuracia:.2%}"
        )

testar(
    7,
    "Treinamento executou todas as epocas",
    len(historico) == EPOCAS
)

testar(
    8,
    "Perdas permaneceram finitas",
    all(
        float("-inf") < perda < float("inf")
        for _, perda, _ in historico
    )
)

# Avaliar somente o ajuste aos exemplos de treino.
modelo.eval()

with torch.no_grad():
    X_todos, tamanhos_todos, y_todos = montar_batch(
        list(range(len(sequencias)))
    )

    logits_finais = modelo(
        X_todos,
        tamanhos_todos
    )

    previsoes_finais = logits_finais.argmax(dim=1)

    acuracia_memorizacao = (
        (previsoes_finais == y_todos)
        .float()
        .mean()
        .item()
    )

print(
    "\nAcuracia nas mesmas 6 frases de treino:",
    f"{acuracia_memorizacao:.2%}"
)

# Funcao de predicao para frases novas.
def tokenizar(texto):
    return re.findall(
        r"\w+|[^\w\s]",
        texto.lower(),
        flags=re.UNICODE
    )

def predict(frases):
    if isinstance(frases, str):
        frases = [frases]

    if not frases:
        raise ValueError("Lista de frases vazia.")

    entradas = []

    for frase in frases:
        tokens = tokenizar(frase)

        if not tokens:
            tokens = ["<UNK>"]

        indices = [
            vocabulario.get(token, UNK)
            for token in tokens
        ]

        entradas.append(
            torch.tensor(indices, dtype=torch.long)
        )

    ordem = sorted(
        range(len(entradas)),
        key=lambda i: len(entradas[i]),
        reverse=True
    )

    ordenadas = [entradas[i] for i in ordem]

    tamanhos = torch.tensor(
        [len(seq) for seq in ordenadas],
        dtype=torch.long
    )

    X = pad_sequence(
        ordenadas,
        batch_first=False,
        padding_value=PAD
    ).to(device)

    modelo.eval()

    with torch.no_grad():
        logits = modelo(X, tamanhos)
        probs = F.softmax(logits, dim=1).cpu()

    respostas = [None] * len(frases)

    for posicao, indice_original in enumerate(ordem):
        respostas[indice_original] = {
            "frase": frases[indice_original],
            "negativo": float(probs[posicao, 0]),
            "positivo": float(probs[posicao, 1])
        }

    return respostas

frases_exemplo = [
    "This movie was amazing!",
    "Terrible movie.",
    "A wonderful film.",
    "The film was disappointing."
]

predicoes = predict(frases_exemplo)

testar(
    9,
    "Predicao aceita multiplas frases",
    len(predicoes) == len(frases_exemplo)
)

testar(
    10,
    "Probabilidades de cada frase somam 1",
    all(
        abs(
            item["negativo"] + item["positivo"] - 1.0
        ) < 1e-5
        for item in predicoes
    )
)

print("\nPREDICOES DIDATICAS")
for item in predicoes:
    print("Frase:", item["frase"])
    print(
        f"  Negativo: {item['negativo']:.2%} | "
        f"Positivo: {item['positivo']:.2%}"
    )

# Salvar novo checkpoint.
checkpoint = pasta / "modelo_sentimentos_didatico_treinado.pt"

torch.save(
    {
        "model_state_dict": modelo.state_dict(),
        "vocabulario": vocabulario,
        "embed_dim": EMBED_DIM,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "epocas": EPOCAS,
        "origem": "6 frases didaticas",
        "pre_treinado_glove": False,
        "treinado": True
    },
    checkpoint
)

testar(
    11,
    "Checkpoint treinado salvo",
    checkpoint.is_file()
)

relatorio = pasta / "relatorio_classificando_sequencias.txt"

relatorio.write_text(
    "AULA - CLASSIFICANDO SEQUENCIAS\n\n"
    + "\n".join(resultados)
    + "\n\n"
    + f"Amostras didaticas: {len(sequencias)}\n"
    + f"Epocas: {EPOCAS}\n"
    + f"Embedding: {EMBED_DIM}\n"
    + f"Hidden: {HIDDEN_SIZE}\n"
    + f"Saida: {OUTPUT_SIZE}\n"
    + f"Acuracia nas frases de treino: "
      f"{acuracia_memorizacao:.2%}\n"
    + "\nTreinamento apenas demonstrativo.\n"
    + "Nao foi utilizado IMDb oficial nem GloVe.\n"
    + "Nao ha teste independente de generalizacao.\n"
    + "\n11/11 TESTES APROVADOS\n",
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("RESULTADO FINAL: 11/11 TESTES APROVADOS")
print("Checkpoint:", checkpoint)
print("Relatorio:", relatorio)