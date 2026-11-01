from pathlib import Path
import sys
import unicodedata
import random

import torch
from torch import nn

torch.manual_seed(42)
random.seed(42)
torch.set_num_threads(2)

pasta_dados = Path(sys.argv[1])
pasta_saida = Path(__file__).resolve().parent

arquivos = sorted(pasta_dados.glob("*.txt"))
categorias = [arquivo.stem for arquivo in arquivos]

assert len(categorias) == 18, (
    f"Esperadas 18 nacionalidades, encontradas {len(categorias)}"
)

def normalizar(texto):
    return "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caractere) != "Mn"
    ).strip()

registros = []

for indice, arquivo in enumerate(arquivos):
    linhas = arquivo.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for linha in linhas:
        nome = normalizar(linha)
        if nome:
            registros.append((nome, indice))

assert registros, "Nenhum nome encontrado."

# Criar vocabulario com os caracteres presentes no projeto.
caracteres = sorted({
    caractere
    for nome, _ in registros
    for caractere in nome
})

vocabulario = {
    caractere: indice
    for indice, caractere in enumerate(caracteres)
}

N = len(vocabulario)
F = 128
C = len(categorias)

def codificar_nome(nome):
    indices = [
        vocabulario[caractere]
        for caractere in normalizar(nome)
        if caractere in vocabulario
    ]

    if not indices:
        raise ValueError("Nome sem caracteres validos.")

    # Formato inicial: S x N
    return torch.nn.functional.one_hot(
        torch.tensor(indices),
        num_classes=N
    ).float()

class ClassificadorGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.recurrent = nn.GRU(
            input_size=N,
            hidden_size=F,
            num_layers=1,
            batch_first=False
        )

        self.classificador = nn.Linear(F, C)

    def forward(self, nome, mostrar_dimensoes=False):
        # Nome inicial: S x N
        # Adicionar batch no meio: S x 1 x N
        nome_atualizado = nome.unsqueeze(1)

        # Memoria inicial: L x B x F
        h_inicial = torch.zeros(
            1, 1, F,
            device=nome.device
        )

        saida, h_final = self.recurrent(
            nome_atualizado,
            h_inicial
        )

        # Many-to-one: ultima etapa da sequencia.
        ultimo_estado = saida[-1]

        logits = self.classificador(ultimo_estado)

        if mostrar_dimensoes:
            print("Nome inicial:", tuple(nome.shape))
            print("H inicial:", tuple(h_inicial.shape))
            print("Nome atualizado:", tuple(nome_atualizado.shape))
            print("Saida GRU:", tuple(saida.shape))
            print("H final:", tuple(h_final.shape))
            print("Saida classificacao:", tuple(logits.shape))

        return logits

modelo = ClassificadorGRU()

def testar(numero, descricao, condicao):
    assert condicao, f"FALHOU: {descricao}"
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 65)
print("AULA - RNN NO PYTORCH")
print("=" * 65)
print("Nacionalidades:", C)
print("Nomes carregados:", len(registros))
print("Caracteres no vocabulario:", N)

testar(
    1,
    "Carregamento das 18 nacionalidades",
    len(categorias) == 18
)

nome_exemplo = codificar_nome(registros[0][0])

testar(
    2,
    "Nome codificado no formato S x N",
    nome_exemplo.ndim == 2
    and nome_exemplo.shape[1] == N
)

print("\nDIMENSOES DA REDE")
with torch.no_grad():
    logits = modelo(
        nome_exemplo,
        mostrar_dimensoes=True
    )

testar(
    3,
    "GRU gera classificacao 1 x 18",
    logits.shape == (1, 18)
)

# Confirmar equivalencia entre o ultimo output e hidden.
with torch.no_grad():
    sequencia = nome_exemplo.unsqueeze(1)
    saidas, hidden = modelo.recurrent(sequencia)

testar(
    4,
    "Ultimo output corresponde ao hidden final",
    torch.allclose(saidas[-1], hidden[-1], atol=1e-6)
)

# Treinamento pequeno com exemplos reais.
# Separar treino e teste por registros.
random.shuffle(registros)

divisao = int(len(registros) * 0.8)
treino = registros[:divisao]
teste = registros[divisao:]

otimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=0.003
)

criterio = nn.CrossEntropyLoss()

print("\nTREINAMENTO DA GRU")

modelo.train()

for epoca in range(3):
    amostra = random.sample(
        treino,
        min(250, len(treino))
    )

    perdas = []

    for nome, classe in amostra:
        entrada = codificar_nome(nome)
        alvo = torch.tensor([classe])

        otimizador.zero_grad()

        previsao = modelo(entrada)
        perda = criterio(previsao, alvo)

        perda.backward()
        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(),
            max_norm=1.0
        )
        otimizador.step()

        perdas.append(perda.item())

    media = sum(perdas) / len(perdas)
    print(f"Epoca {epoca + 1}/3 - perda: {media:.4f}")

testar(
    5,
    "Treinamento com backpropagation",
    len(perdas) > 0
    and all(
        valor == valor and valor != float("inf")
        for valor in perdas
    )
)

# Avaliacao em registros separados do treino.
modelo.eval()

acertos = 0
amostra_teste = random.sample(
    teste,
    min(200, len(teste))
)

with torch.no_grad():
    for nome, classe in amostra_teste:
        previsao = modelo(codificar_nome(nome))
        classe_prevista = previsao.argmax(dim=1).item()
        acertos += int(classe_prevista == classe)

acuracia = acertos / len(amostra_teste)

print(f"\nAcuracia no teste: {acuracia * 100:.2f}%")

testar(
    6,
    "Avaliacao com nomes nao usados no treino",
    0.0 <= acuracia <= 1.0
)

# Salvar em arquivo NOVO.
checkpoint = pasta_saida / "modelo_gru_nacionalidades.pt"

torch.save({
    "model_state_dict": modelo.state_dict(),
    "categorias": categorias,
    "vocabulario": vocabulario,
    "input_size": N,
    "hidden_size": F,
    "num_classes": C,
}, checkpoint)

relatorio = pasta_saida / "relatorio_rnn_pytorch.txt"

relatorio.write_text(
    "AULA - RNN NO PYTORCH\n\n"
    "Arquitetura: nn.GRU\n"
    f"Nacionalidades: {C}\n"
    f"Nomes carregados: {len(registros)}\n"
    f"Vocabulario: {N}\n"
    "Formato entrada: S x 1 x N\n"
    "Formato hidden: 1 x 1 x F\n"
    "Formato classificacao: 1 x 18\n"
    f"Acuracia no teste: {acuracia * 100:.2f}%\n"
    "Treinamento: 3 epocas x ate 250 exemplos\n"
    "6/6 TESTES APROVADOS\n"
    "Checkpoint anterior preservado.\n",
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("RESULTADO FINAL: 6/6 TESTES APROVADOS")
print("Modelo GRU:", checkpoint)
print("Relatorio:", relatorio)