from pathlib import Path
import re
import json

import torch

torch.manual_seed(42)

pasta = Path(__file__).resolve().parent
resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"FALHOU: {descricao}"
    resultados.append(f"{numero}. {descricao}: APROVADO")
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 65)
print("LABORATORIO - DESCRICAO DO PROBLEMA IMDb")
print("=" * 65)
print("PyTorch:", torch.__version__)

# Avaliacoes didaticas, nao extraidas do IMDb.
amostras = [
    {
        "text": "This movie was amazing! Great acting and story.",
        "label": "pos"
    },
    {
        "text": "Terrible movie. I hated the ending.",
        "label": "neg"
    },
    {
        "text": "A wonderful film with excellent performances.",
        "label": "pos"
    },
    {
        "text": "Boring plot and very bad dialogue.",
        "label": "neg"
    },
    {
        "text": "I really loved this fantastic movie!",
        "label": "pos"
    },
    {
        "text": "The film was disappointing and too long.",
        "label": "neg"
    }
]

# Equivalente didatico da configuracao Field.
def tokenizar(texto):
    return re.findall(
        r"\w+|[^\w\s]",
        texto.lower(),
        flags=re.UNICODE
    )

def processar(amostra):
    tokens = tokenizar(amostra["text"])

    return {
        "text": tokens,
        "label": amostra["label"],
        "length": len(tokens)
    }

dados = [processar(a) for a in amostras]

testar(
    1,
    "Carregamento das avaliacoes de demonstracao",
    len(dados) == 6
)

print("\nEXEMPLO DE AVALIACAO")
print("Texto original:", amostras[0]["text"])
print("Tokens:", dados[0]["text"])
print("Rotulo:", dados[0]["label"])
print("Comprimento:", dados[0]["length"])

testar(
    2,
    "Tokenizacao em minusculas",
    dados[0]["text"][0] == "this"
    and "!" in dados[0]["text"]
)

testar(
    3,
    "Separacao entre texto e rotulo",
    all(
        isinstance(a["text"], list)
        and a["label"] in ("pos", "neg")
        for a in dados
    )
)

testar(
    4,
    "Comprimento de cada sequencia",
    all(
        a["length"] == len(a["text"])
        for a in dados
    )
)

# Converter rotulos para inteiros long.
rotulos = {
    "neg": 0,
    "pos": 1
}

y = torch.tensor(
    [rotulos[a["label"]] for a in dados],
    dtype=torch.long
)

testar(
    5,
    "Rotulos convertidos para torch.long",
    y.dtype == torch.long
    and y.shape == (6,)
)

# Comprimentos das sequencias.
comprimentos = torch.tensor(
    [a["length"] for a in dados],
    dtype=torch.long
)

testar(
    6,
    "Comprimentos armazenados como tensores",
    comprimentos.shape == (6,)
    and torch.all(comprimentos > 0).item()
)

# Vocabulario simples para preparar proxima aula.
vocabulario = {
    "<PAD>": 0,
    "<UNK>": 1
}

for amostra in dados:
    for token in amostra["text"]:
        if token not in vocabulario:
            vocabulario[token] = len(vocabulario)

sequencias = [
    [vocabulario[token] for token in a["text"]]
    for a in dados
]

testar(
    7,
    "Vocabulario numerico criado",
    len(vocabulario) > 2
    and all(
        len(seq) > 0 for seq in sequencias
    )
)

# Demonstracao de batch_first.
maior = max(len(seq) for seq in sequencias)

batch = torch.zeros(
    len(sequencias),
    maior,
    dtype=torch.long
)

for i, seq in enumerate(sequencias):
    batch[i, :len(seq)] = torch.tensor(seq)

batch_first_false = batch.transpose(0, 1)

testar(
    8,
    "batch_first altera organizacao B x S e S x B",
    batch.shape == (6, maior)
    and batch_first_false.shape == (maior, 6)
)

# Procurar dataset IMDb previamente extraido.
raiz_projeto = pasta.parents[1]

possiveis_locais = [
    raiz_projeto / "aclImdb",
    raiz_projeto / "data" / "aclImdb",
    raiz_projeto / "data" / "imdb" / "aclImdb"
]

dataset_local = next(
    (
        caminho for caminho in possiveis_locais
        if (caminho / "train" / "pos").is_dir()
        and (caminho / "train" / "neg").is_dir()
        and (caminho / "test" / "pos").is_dir()
        and (caminho / "test" / "neg").is_dir()
    ),
    None
)

print("\nDATASET OFICIAL IMDb")
if dataset_local is None:
    print("Nao encontrado nos caminhos padrao do projeto.")
    print("Este laboratorio usa 6 exemplos didaticos.")
else:
    print("Encontrado:", dataset_local)

# Salvar dados e configuracao.
saida_json = pasta / "amostras_processadas.json"

saida_json.write_text(
    json.dumps(
        {
            "origem": "Exemplos didaticos, nao IMDb",
            "amostras": dados,
            "vocabulario": vocabulario
        },
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

relatorio = pasta / "relatorio_imdb_descricao.txt"

relatorio.write_text(
    "AULA - DESCRICAO DO PROBLEMA IMDb\n\n"
    + "\n".join(resultados)
    + "\n\n"
    + f"PyTorch: {torch.__version__}\n"
    + f"Amostras didaticas: {len(dados)}\n"
    + f"Vocabulario: {len(vocabulario)} tokens\n"
    + f"Maior sequencia: {maior} tokens\n"
    + f"IMDb local: {dataset_local or 'nao encontrado'}\n"
    + "\n8/8 TESTES APROVADOS\n"
    + "Nenhum modelo foi treinado.\n",
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("RESULTADO FINAL: 8/8 TESTES APROVADOS")
print("Vocabulario:", len(vocabulario))
print("Batch B x S:", tuple(batch.shape))
print("Batch S x B:", tuple(batch_first_false.shape))
print("Relatorio:", relatorio)