import torch
from torch import nn
from pathlib import Path

torch.manual_seed(42)
pasta = Path(__file__).resolve().parent
resultados = []

def registrar(nome):
    resultados.append(f"{nome}: APROVADO")
    print(f"\n[APROVADO] {nome}")

print("=" * 60)
print("LABORATORIO - GATES EM REDES RECORRENTES")
print("=" * 60)

# EXPERIMENTO 1
# A sigmoid transforma numeros em valores entre 0 e 1.

valores = torch.tensor([-3.0, 0.0, 3.0])
gate = torch.sigmoid(valores)

print("\n1. CRIANDO UM GATE")
print("Valores originais:", valores.tolist())
print("Valores do gate:", gate.tolist())

assert torch.all(gate >= 0)
assert torch.all(gate <= 1)
registrar("Gate com valores entre 0 e 1")

# EXPERIMENTO 2
# Cada dimensao da memoria recebe um controle independente.

memoria = torch.tensor([10.0, 20.0, 30.0])
gate_manual = torch.tensor([0.9, 0.6, 0.2])

memoria_filtrada = memoria * gate_manual
esperado = torch.tensor([9.0, 12.0, 6.0])

print("\n2. CONTROLANDO O HIDDEN STATE")
print("Memoria original:", memoria.tolist())
print("Gate:", gate_manual.tolist())
print("Memoria filtrada:", memoria_filtrada.tolist())

assert torch.allclose(memoria_filtrada, esperado)
registrar("Multiplicacao elemento a elemento")

# EXPERIMENTO 3
# O gate pode ser produzido por uma camada treinavel.

camada_gate = nn.Linear(3, 3)
entrada = torch.tensor([[1.0, 2.0, 3.0]])

gate_aprendido = torch.sigmoid(camada_gate(entrada))
saida = entrada * gate_aprendido

print("\n3. GATE COM PESOS TREINAVEIS")
print("Entrada:", entrada.tolist())
print("Gate gerado:", gate_aprendido.tolist())
print("Saida filtrada:", saida.tolist())

assert gate_aprendido.shape == entrada.shape
assert torch.all((gate_aprendido >= 0) & (gate_aprendido <= 1))
registrar("Gate produzido por nn.Linear e sigmoid")

# EXPERIMENTO 4
# Treinamos um gate para aproximar uma filtragem desejada.
# Este exemplo didatico nao e uma LSTM nem uma GRU.

modelo = nn.Sequential(
    nn.Linear(3, 3),
    nn.Sigmoid()
)

otimizador = torch.optim.Adam(modelo.parameters(), lr=0.05)
criterio = nn.MSELoss()

entrada_treino = torch.tensor([[1.0, 1.0, 1.0]])
gate_desejado = torch.tensor([[0.9, 0.6, 0.2]])

with torch.no_grad():
    perda_inicial = criterio(
        modelo(entrada_treino), gate_desejado
    ).item()

for epoca in range(300):
    otimizador.zero_grad()
    previsao = modelo(entrada_treino)
    perda = criterio(previsao, gate_desejado)
    perda.backward()
    otimizador.step()

with torch.no_grad():
    gate_final = modelo(entrada_treino)
    perda_final = criterio(
        gate_final, gate_desejado
    ).item()

print("\n4. TREINANDO O GATE")
print("Gate desejado:", gate_desejado.tolist())
print("Gate aprendido:", gate_final.tolist())
print("Perda inicial:", perda_inicial)
print("Perda final:", perda_final)

assert perda_final < perda_inicial
assert perda_final < 0.001
registrar("Gate aprendeu os valores desejados")

# RELATORIO

texto = (
    "LABORATORIO - GATES\n\n"
    + "\n".join(resultados)
    + f"\n\nPerda inicial: {perda_inicial:.8f}"
    + f"\nPerda final: {perda_final:.8f}"
    + "\n\n4/4 TESTES APROVADOS\n"
    + "\nObservacao: laboratorio didatico, sem treinar LSTM/GRU.\n"
)

relatorio = pasta / "relatorio_gates.txt"
relatorio.write_text(texto, encoding="utf-8")

print("\n" + "=" * 60)
print("RESULTADO FINAL: 4/4 TESTES APROVADOS")
print("Relatorio:", relatorio)