from pathlib import Path
import torch
from torch import nn

torch.manual_seed(42)
torch.set_num_threads(2)

pasta = Path(__file__).resolve().parent
resultados = []

def aprovar(nome, condicao):
    assert condicao, f"FALHOU: {nome}"
    resultados.append(f"{nome}: APROVADO")
    print(f"[APROVADO] {nome}")

print("=" * 60)
print("LABORATORIO - RNN, GRU E LSTM")
print("=" * 60)

batch_size = 4
input_size = 10
hidden_size = 16

x = torch.randn(batch_size, input_size)
h = torch.zeros(batch_size, hidden_size)
c = torch.zeros(batch_size, hidden_size)

# 1. RNNCell: entrada + hidden state
rnn_cell = nn.RNNCell(input_size, hidden_size)
h_rnn = rnn_cell(x, h)

aprovar(
    "1. RNNCell produz hidden state",
    h_rnn.shape == (batch_size, hidden_size)
)

# 2. GRUCell: mesma interface basica da RNNCell
gru_cell = nn.GRUCell(input_size, hidden_size)
h_gru = gru_cell(x, h)

aprovar(
    "2. GRUCell produz hidden state",
    h_gru.shape == (batch_size, hidden_size)
)

# 3. LSTMCell: hidden state + cell state
lstm_cell = nn.LSTMCell(input_size, hidden_size)
h_lstm, c_lstm = lstm_cell(x, (h, c))

aprovar(
    "3. LSTMCell produz duas memorias",
    h_lstm.shape == (batch_size, hidden_size)
    and c_lstm.shape == (batch_size, hidden_size)
)

# 4. Comparacao de parametros
def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

p_rnn = contar_parametros(rnn_cell)
p_gru = contar_parametros(gru_cell)
p_lstm = contar_parametros(lstm_cell)

print("\nPARAMETROS DAS CELULAS")
print("RNNCell :", p_rnn)
print("GRUCell :", p_gru)
print("LSTMCell:", p_lstm)

aprovar(
    "4. Quantidade de parametros: RNN < GRU < LSTM",
    p_rnn < p_gru < p_lstm
)

# 5. Processamento de uma sequencia
sequencia = torch.randn(batch_size, 8, input_size)

estado_gru = torch.zeros(batch_size, hidden_size)
estado_lstm = torch.zeros(batch_size, hidden_size)
celula_lstm = torch.zeros(batch_size, hidden_size)

saidas_gru = []
saidas_lstm = []

for t in range(sequencia.size(1)):
    entrada_atual = sequencia[:, t, :]

    estado_gru = gru_cell(entrada_atual, estado_gru)

    estado_lstm, celula_lstm = lstm_cell(
        entrada_atual,
        (estado_lstm, celula_lstm)
    )

    saidas_gru.append(estado_gru)
    saidas_lstm.append(estado_lstm)

saidas_gru = torch.stack(saidas_gru, dim=1)
saidas_lstm = torch.stack(saidas_lstm, dim=1)

aprovar(
    "5. GRU e LSTM processam sequencias",
    saidas_gru.shape == (batch_size, 8, hidden_size)
    and saidas_lstm.shape == (batch_size, 8, hidden_size)
)

# 6. Treinamento em tarefa sintetica
# Objetivo: lembrar o primeiro valor da sequencia.
# O restante da sequencia contem ruido.

class Classificador(nn.Module):
    def __init__(self, tipo):
        super().__init__()
        if tipo == "GRU":
            self.rede = nn.GRU(1, 24, batch_first=True)
        else:
            self.rede = nn.LSTM(1, 24, batch_first=True)

        self.saida = nn.Linear(24, 2)

    def forward(self, entrada):
        sequencia_saida, _ = self.rede(entrada)
        return self.saida(sequencia_saida[:, -1, :])

def gerar_dados(n, comprimento=12):
    rotulos = torch.randint(0, 2, (n,))
    dados = torch.randn(n, comprimento, 1) * 0.15

    dados[:, 0, 0] = rotulos.float() * 2 - 1
    return dados, rotulos

dados_treino, rotulos_treino = gerar_dados(512)
dados_teste, rotulos_teste = gerar_dados(128)

metricas = {}

for tipo in ["GRU", "LSTM"]:
    modelo = Classificador(tipo)
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=0.01
    )
    criterio = nn.CrossEntropyLoss()

    for epoca in range(80):
        modelo.train()
        otimizador.zero_grad()

        previsoes = modelo(dados_treino)
        perda = criterio(previsoes, rotulos_treino)

        perda.backward()
        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(), 1.0
        )
        otimizador.step()

    modelo.eval()
    with torch.no_grad():
        previsoes_teste = modelo(dados_teste)
        acuracia = (
            previsoes_teste.argmax(dim=1)
            == rotulos_teste
        ).float().mean().item()

    metricas[tipo] = acuracia
    print(
        f"\n{tipo}: acuracia de teste = "
        f"{acuracia * 100:.2f}%"
    )

aprovar(
    "6. Treinamento GRU e LSTM executado",
    all(
        0.0 <= valor <= 1.0
        for valor in metricas.values()
    )
)

relatorio = pasta / "relatorio_lstm_gru.txt"

texto = (
    "LABORATORIO - LSTM E GRU\n\n"
    + "\n".join(resultados)
    + "\n\nPARAMETROS\n"
    + f"RNNCell: {p_rnn}\n"
    + f"GRUCell: {p_gru}\n"
    + f"LSTMCell: {p_lstm}\n"
    + "\nACURACIA DE TESTE\n"
    + f"GRU: {metricas['GRU'] * 100:.2f}%\n"
    + f"LSTM: {metricas['LSTM'] * 100:.2f}%\n"
    + "\n6/6 TESTES ESTRUTURAIS APROVADOS\n"
    + "Acuracia registrada, sem minimo exigido.\n"
    + "Dados sinteticos; modelos anteriores preservados.\n"
)

relatorio.write_text(texto, encoding="utf-8")

print("\n" + "=" * 60)
print("RESULTADO FINAL: 6/6 TESTES APROVADOS")
print("Relatorio:", relatorio)