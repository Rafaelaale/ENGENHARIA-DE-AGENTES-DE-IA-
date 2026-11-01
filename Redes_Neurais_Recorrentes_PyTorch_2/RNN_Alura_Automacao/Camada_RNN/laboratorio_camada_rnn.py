from pathlib import Path
import torch
from torch import nn

torch.manual_seed(42)
torch.set_num_threads(2)

pasta = Path(__file__).resolve().parent
resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"TESTE {numero} FALHOU: {descricao}"
    resultados.append(f"{numero}. {descricao}: APROVADO")
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 65)
print("LABORATORIO - CAMADA RNN")
print("=" * 65)

# Quatro cidades, sete dias, uma temperatura por dia.
B = 4
S = 7
N = 1
F = 16

temperaturas = torch.tensor([
    [29., 30., 31., 29., 28., 30., 32.],
    [23., 24., 25., 26., 24., 23., 22.],
    [31., 32., 33., 34., 32., 31., 30.],
    [19., 20., 21., 20., 18., 19., 20.]
]).unsqueeze(-1)

# TESTE 1: batch_first=True
assert temperaturas.shape == (B, S, N)
testar(
    1,
    "Entrada organizada em B x S x N",
    temperaturas.shape == (4, 7, 1)
)

# TESTE 2: GRU processa sequencia inteira
gru = nn.GRU(
    input_size=N,
    hidden_size=F,
    batch_first=True
)

output, hidden = gru(temperaturas)

print("\nGRU COM BATCH_FIRST=True")
print("Entrada:", tuple(temperaturas.shape))
print("Output:", tuple(output.shape))
print("Hidden:", tuple(hidden.shape))

testar(
    2,
    "GRU retorna output e hidden",
    output.shape == (B, S, F)
    and hidden.shape == (1, B, F)
)

# TESTE 3: Ultimo output corresponde ao hidden
ultimo_output = output[:, -1, :]
ultimo_hidden = hidden[-1]

testar(
    3,
    "Ultimo output igual ao hidden final",
    torch.allclose(ultimo_output, ultimo_hidden, atol=1e-6)
)

# TESTE 4: batch_first=False
temperaturas_sbn = temperaturas.transpose(0, 1)

gru_sbn = nn.GRU(
    input_size=N,
    hidden_size=F,
    batch_first=False
)

output_sbn, hidden_sbn = gru_sbn(temperaturas_sbn)

print("\nGRU COM BATCH_FIRST=False")
print("Entrada:", tuple(temperaturas_sbn.shape))
print("Output:", tuple(output_sbn.shape))
print("Hidden:", tuple(hidden_sbn.shape))

testar(
    4,
    "Formato S x B x N e saida S x B x F",
    output_sbn.shape == (S, B, F)
    and hidden_sbn.shape == (1, B, F)
)

# TESTE 5: Comparar as duas organizacoes com os mesmos pesos
gru_sbn.load_state_dict(gru.state_dict())

output_sbn, hidden_sbn = gru_sbn(temperaturas_sbn)

testar(
    5,
    "batch_first altera dimensoes, nao calculos",
    torch.allclose(
        output,
        output_sbn.transpose(0, 1),
        atol=1e-6
    )
    and torch.allclose(hidden, hidden_sbn, atol=1e-6)
)

# TESTE 6: GRUCell manual versus GRU completa
gru_cell = nn.GRUCell(N, F)

# Transferir os mesmos pesos da camada GRU para a GRUCell.
with torch.no_grad():
    gru_cell.weight_ih.copy_(gru.weight_ih_l0)
    gru_cell.weight_hh.copy_(gru.weight_hh_l0)
    gru_cell.bias_ih.copy_(gru.bias_ih_l0)
    gru_cell.bias_hh.copy_(gru.bias_hh_l0)

estado = torch.zeros(B, F)
saidas_manuais = []

for t in range(S):
    estado = gru_cell(temperaturas[:, t, :], estado)
    saidas_manuais.append(estado)

output_manual = torch.stack(saidas_manuais, dim=1)

testar(
    6,
    "GRUCell com loop equivale a GRU",
    torch.allclose(output_manual, output, atol=1e-5)
)

# TESTE 7: LSTM possui hidden e cell state
lstm = nn.LSTM(
    input_size=N,
    hidden_size=F,
    batch_first=True
)

output_lstm, (hidden_lstm, cell_lstm) = lstm(temperaturas)

testar(
    7,
    "LSTM retorna output, hidden e cell state",
    output_lstm.shape == (B, S, F)
    and hidden_lstm.shape == (1, B, F)
    and cell_lstm.shape == (1, B, F)
)

# TESTE 8: Multiplas camadas
gru_multicamadas = nn.GRU(
    input_size=N,
    hidden_size=F,
    num_layers=3,
    batch_first=True
)

output_multi, hidden_multi = gru_multicamadas(temperaturas)

testar(
    8,
    "Hidden acompanha numero de camadas",
    output_multi.shape == (B, S, F)
    and hidden_multi.shape == (3, B, F)
)

print("\nRESUMO DAS DIMENSOES")
print("Temperaturas:", tuple(temperaturas.shape))
print("GRU output:", tuple(output.shape))
print("GRU hidden:", tuple(hidden.shape))
print("LSTM output:", tuple(output_lstm.shape))
print("LSTM hidden:", tuple(hidden_lstm.shape))
print("LSTM cell:", tuple(cell_lstm.shape))
print("GRU 3 camadas hidden:", tuple(hidden_multi.shape))

relatorio = pasta / "relatorio_camada_rnn.txt"

texto = (
    "LABORATORIO - CAMADA RNN\n\n"
    + "\n".join(resultados)
    + "\n\nDIMENSOES\n"
    + f"Entrada: {tuple(temperaturas.shape)}\n"
    + f"GRU output: {tuple(output.shape)}\n"
    + f"GRU hidden: {tuple(hidden.shape)}\n"
    + f"LSTM cell: {tuple(cell_lstm.shape)}\n"
    + f"GRU 3 camadas hidden: {tuple(hidden_multi.shape)}\n"
    + "\n8/8 TESTES APROVADOS\n"
    + "Experimentos estruturais, sem treinamento.\n"
    + "Checkpoint anterior preservado.\n"
)

relatorio.write_text(texto, encoding="utf-8")

print("\n" + "=" * 65)
print("RESULTADO FINAL: 8/8 TESTES APROVADOS")
print("Relatorio:", relatorio)