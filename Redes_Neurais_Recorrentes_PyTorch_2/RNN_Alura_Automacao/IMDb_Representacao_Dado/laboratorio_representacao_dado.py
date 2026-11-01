from pathlib import Path
from collections import Counter
import json
import sys

import torch
from torch import nn

torch.manual_seed(42)
torch.set_num_threads(2)

arquivo_dados = Path(sys.argv[1])
pasta = Path(__file__).resolve().parent

dados = json.loads(arquivo_dados.read_text(encoding="utf-8"))
amostras = dados["amostras"]

resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"TESTE {numero} FALHOU: {descricao}"
    resultados.append(f"{numero}. {descricao}: APROVADO")
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 67)
print("LABORATORIO - REPRESENTACAO DO DADO")
print("=" * 67)

# Construir vocabulario somente a partir das amostras
# disponiveis neste laboratorio.
frequencias = Counter(
    token
    for amostra in amostras
    for token in amostra["text"]
)

MAX_SIZE = 25_000
EMBED_DIM = 100

UNK = "<UNK>"
PAD = "<PAD>"

vocabulario = {
    UNK: 0,
    PAD: 1
}

for token, _ in frequencias.most_common(MAX_SIZE):
    if token not in vocabulario:
        vocabulario[token] = len(vocabulario)

ind_unk = vocabulario[UNK]
ind_pad = vocabulario[PAD]
vocab_size = len(vocabulario)

print("Amostras:", len(amostras))
print("Vocabulario:", vocab_size)
print("Limite configurado:", MAX_SIZE)
print("Embedding:", EMBED_DIM)

testar(
    1,
    "Vocabulario com tokens especiais",
    vocabulario[UNK] == 0
    and vocabulario[PAD] == 1
)

testar(
    2,
    "Limite de vocabulario respeitado",
    vocab_size <= MAX_SIZE + 2
)

print("\nTOKENS MAIS FREQUENTES")
print(frequencias.most_common(10))

testar(
    3,
    "Frequencias dos tokens calculadas",
    len(frequencias) > 0
    and sum(frequencias.values()) > 0
)

# Palavra fora do vocabulario recebe UNK.
def converter_tokens(tokens):
    return [
        vocabulario.get(token, ind_unk)
        for token in tokens
    ]

desconhecido = converter_tokens(["palavra_inexistente_123"])

testar(
    4,
    "Palavras desconhecidas recebem indice UNK",
    desconhecido == [ind_unk]
)

# Rotulos binarios.
rotulos = {"neg": 0, "pos": 1}

y = torch.tensor(
    [rotulos[a["label"]] for a in amostras],
    dtype=torch.long
)

testar(
    5,
    "Rotulos convertidos para torch.long",
    y.dtype == torch.long
    and set(y.tolist()) == {0, 1}
)

# Representacao dos textos por indices.
sequencias = [
    converter_tokens(a["text"])
    for a in amostras
]

comprimentos = torch.tensor(
    [len(s) for s in sequencias],
    dtype=torch.long
)

testar(
    6,
    "Textos convertidos em sequencias de indices",
    all(len(s) > 0 for s in sequencias)
    and len(sequencias) == len(amostras)
)

# Montar tensor retangular S x B com padding.
B = len(sequencias)
S = max(len(s) for s in sequencias)

X = torch.full(
    (S, B),
    fill_value=ind_pad,
    dtype=torch.long
)

for b, seq in enumerate(sequencias):
    X[:len(seq), b] = torch.tensor(
        seq,
        dtype=torch.long
    )

testar(
    7,
    "Batch com formato S x B e padding",
    X.shape == (S, B)
    and X.dtype == torch.long
)

# Criar camada embedding.
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=EMBED_DIM,
    padding_idx=ind_pad
)

# Zerar representacoes especiais.
with torch.no_grad():
    embedding.weight[ind_unk].zero_()
    embedding.weight[ind_pad].zero_()

vetores = embedding(X)

print("\nDIMENSOES")
print("Indices:", tuple(X.shape))
print("Embedding:", tuple(vetores.shape))
print("Rotulos:", tuple(y.shape))
print("Comprimentos:", comprimentos.tolist())

testar(
    8,
    "Embedding transforma S x B em S x B x 100",
    vetores.shape == (S, B, EMBED_DIM)
)

testar(
    9,
    "Vetores UNK e PAD inicializados com zeros",
    torch.count_nonzero(embedding.weight[ind_unk]).item() == 0
    and torch.count_nonzero(embedding.weight[ind_pad]).item() == 0
)

# Comparar representacao one-hot com busca por indice.
indice_exemplo = vocabulario[amostras[0]["text"][0]]

one_hot = torch.nn.functional.one_hot(
    torch.tensor(indice_exemplo),
    num_classes=vocab_size
).float()

vetor_por_multiplicacao = (
    one_hot @ embedding.weight.detach()
)

vetor_por_indice = embedding(
    torch.tensor(indice_exemplo)
).detach()

testar(
    10,
    "Embedding equivale a selecionar vetor por one-hot",
    torch.allclose(
        vetor_por_multiplicacao,
        vetor_por_indice
    )
)

# Salvar resultados para a proxima aula.
torch.save(
    {
        "X": X,
        "y": y,
        "comprimentos": comprimentos,
        "embedding_weights": embedding.weight.detach(),
        "vocabulario": vocabulario,
        "rotulos": rotulos,
        "embedding_dim": EMBED_DIM,
        "origem": "6 avaliacoes didaticas",
        "pre_treinado": False
    },
    pasta / "dados_representados.pt"
)

relatorio = pasta / "relatorio_representacao_dado.txt"

relatorio.write_text(
    "AULA - REPRESENTACAO DO DADO\n\n"
    + "\n".join(resultados)
    + "\n\n"
    + f"Amostras: {B}\n"
    + f"Vocabulario: {vocab_size}\n"
    + f"Embedding: {EMBED_DIM}\n"
    + f"Tensor X: {tuple(X.shape)}\n"
    + f"Tensor embedding: {tuple(vetores.shape)}\n"
    + f"Rotulos: {tuple(y.shape)}\n"
    + "\nOrigem: exemplos didaticos.\n"
    + "Embedding inicializado aleatoriamente, nao GloVe.\n"
    + "Nenhum classificador foi treinado.\n"
    + "\n10/10 TESTES APROVADOS\n",
    encoding="utf-8"
)

print("\n" + "=" * 67)
print("RESULTADO FINAL: 10/10 TESTES APROVADOS")
print("Dados:", pasta / "dados_representados.pt")
print("Relatorio:", relatorio)