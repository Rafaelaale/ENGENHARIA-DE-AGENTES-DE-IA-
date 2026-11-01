from pathlib import Path
import sys
import random
import unicodedata
import math

import torch
from torch import nn

torch.manual_seed(42)
random.seed(42)
torch.set_num_threads(2)

pasta_dados = Path(sys.argv[1])
pasta_saida = Path(__file__).resolve().parent

resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"FALHOU: {descricao}"
    resultados.append(f"{numero}. {descricao}: APROVADO")
    print(f"[APROVADO] {numero}. {descricao}")

def parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

print("=" * 68)
print("LABORATORIO - RNN PROFUNDA E BIDIRECIONAL")
print("=" * 68)

S = 7
B = 4
N = 10
F = 16
L = 3
D = 2

entrada = torch.randn(S, B, N)

# 1. GRU de uma camada, unidirecional
gru_simples = nn.GRU(N, F, num_layers=1)
saida_simples, h_simples = gru_simples(entrada)

testar(
    1,
    "GRU simples: output S x B x F e hidden 1 x B x F",
    saida_simples.shape == (S, B, F)
    and h_simples.shape == (1, B, F)
)

# 2. GRU profunda
gru_profunda = nn.GRU(N, F, num_layers=L)
saida_profunda, h_profundo = gru_profunda(entrada)

testar(
    2,
    "GRU profunda: hidden L x B x F",
    saida_profunda.shape == (S, B, F)
    and h_profundo.shape == (L, B, F)
)

# 3. GRU bidirecional
gru_bidirecional = nn.GRU(
    N, F,
    num_layers=1,
    bidirectional=True
)

saida_bi, h_bi = gru_bidirecional(entrada)

testar(
    3,
    "GRU bidirecional: output S x B x 2F",
    saida_bi.shape == (S, B, 2 * F)
    and h_bi.shape == (2, B, F)
)

# 4. GRU profunda e bidirecional
gru_completa = nn.GRU(
    N, F,
    num_layers=L,
    bidirectional=True
)

h0 = torch.zeros(L * D, B, F)
saida_completa, h_completo = gru_completa(
    entrada, h0
)

testar(
    4,
    "GRU profunda bidirecional: hidden L*D x B x F",
    saida_completa.shape == (S, B, 2 * F)
    and h_completo.shape == (L * D, B, F)
)

# 5. Separar camadas e direcoes com reshape
h_organizado = h_completo.reshape(L, D, B, F)

ultima_camada_frente = h_organizado[-1, 0]
ultima_camada_tras = h_organizado[-1, 1]

testar(
    5,
    "Reshape separa camadas e direcoes",
    h_organizado.shape == (L, D, B, F)
    and ultima_camada_frente.shape == (B, F)
    and ultima_camada_tras.shape == (B, F)
)

# 6. Ultimos estados das duas direcoes
representacao_final = torch.cat(
    [
        ultima_camada_frente,
        ultima_camada_tras
    ],
    dim=1
)

testar(
    6,
    "Concatenacao dos estados finais produz B x 2F",
    representacao_final.shape == (B, 2 * F)
)

# 7. Verificar correspondencia com output
# Direcao forward termina no ultimo passo.
# Direcao backward termina no primeiro passo.
forward_final = saida_completa[-1, :, :F]
backward_final = saida_completa[0, :, F:]

testar(
    7,
    "Estados finais correspondem as pontas corretas",
    torch.allclose(
        forward_final,
        ultima_camada_frente,
        atol=1e-6
    )
    and torch.allclose(
        backward_final,
        ultima_camada_tras,
        atol=1e-6
    )
)

# 8. LSTM profunda bidirecional
lstm = nn.LSTM(
    N, F,
    num_layers=L,
    bidirectional=True
)

saida_lstm, (h_lstm, c_lstm) = lstm(entrada)

testar(
    8,
    "LSTM profunda bidirecional possui h e c",
    saida_lstm.shape == (S, B, 2 * F)
    and h_lstm.shape == (L * D, B, F)
    and c_lstm.shape == (L * D, B, F)
)

# 9. Comparar quantidade de parametros
p_simples = parametros(gru_simples)
p_profunda = parametros(gru_profunda)
p_bidirecional = parametros(gru_bidirecional)
p_completa = parametros(gru_completa)

print("\nCOMPARACAO DE PARAMETROS")
print("GRU simples:", p_simples)
print("GRU profunda:", p_profunda)
print("GRU bidirecional:", p_bidirecional)
print("GRU profunda bidirecional:", p_completa)

testar(
    9,
    "Arquiteturas profundas e bidirecionais aumentam parametros",
    p_profunda > p_simples
    and p_bidirecional > p_simples
    and p_completa > p_profunda
    and p_completa > p_bidirecional
)

# 10. Classificador real de nomes
arquivos = sorted(pasta_dados.glob("*.txt"))
categorias = [arquivo.stem for arquivo in arquivos]

assert len(categorias) == 18

def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    ).strip()

registros = []

for indice, arquivo in enumerate(arquivos):
    for linha in arquivo.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        nome = normalizar(linha)
        if nome:
            registros.append((nome, indice))

caracteres = sorted({
    c for nome, _ in registros for c in nome
})

vocabulario = {
    c: i for i, c in enumerate(caracteres)
}

def codificar(nome):
    indices = [
        vocabulario[c]
        for c in normalizar(nome)
        if c in vocabulario
    ]

    return torch.nn.functional.one_hot(
        torch.tensor(indices),
        num_classes=len(vocabulario)
    ).float()

class ClassificadorProfundoBidirecional(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=len(vocabulario),
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        self.saida = nn.Linear(128, 18)

    def forward(self, nome):
        entrada = nome.unsqueeze(1)
        _, hidden = self.gru(entrada)

        hidden = hidden.reshape(2, 2, 1, 64)
        representacao = torch.cat(
            [hidden[-1, 0], hidden[-1, 1]],
            dim=1
        )

        return self.saida(representacao)

modelo = ClassificadorProfundoBidirecional()

nome_exemplo = codificar(registros[0][0])

with torch.no_grad():
    classificacao = modelo(nome_exemplo)

testar(
    10,
    "Classificador profundo bidirecional retorna 18 classes",
    classificacao.shape == (1, 18)
)

# Salvar checkpoint estrutural, sem treinamento.
checkpoint = pasta_saida / "modelo_gru_profunda_bidirecional_inicial.pt"

torch.save({
    "model_state_dict": modelo.state_dict(),
    "categorias": categorias,
    "vocabulario": vocabulario,
    "hidden_size": 64,
    "num_layers": 2,
    "bidirectional": True,
    "treinado": False
}, checkpoint)

relatorio = pasta_saida / "relatorio_rnn_profunda_bidirecional.txt"

relatorio.write_text(
    "LABORATORIO - RNN PROFUNDA BIDIRECIONAL\n\n"
    + "\n".join(resultados)
    + "\n\nPARAMETROS\n"
    + f"GRU simples: {p_simples}\n"
    + f"GRU profunda: {p_profunda}\n"
    + f"GRU bidirecional: {p_bidirecional}\n"
    + f"GRU profunda bidirecional: {p_completa}\n"
    + f"\nNomes carregados: {len(registros)}\n"
    + f"Nacionalidades: {len(categorias)}\n"
    + "\n10/10 TESTES APROVADOS\n"
    + "Modelo profundo bidirecional criado, nao treinado.\n"
    + "Checkpoints anteriores preservados.\n",
    encoding="utf-8"
)

print("\n" + "=" * 68)
print("RESULTADO FINAL: 10/10 TESTES APROVADOS")
print("Nomes carregados:", len(registros))
print("Modelo inicial:", checkpoint)
print("Relatorio:", relatorio)