import torch
from torch import nn

torch.manual_seed(42)

# Uma sequencia de 5 elementos, cada um com 10 caracteristicas.
entrada = torch.randn(1, 5, 10)

rnn = nn.RNN(
    input_size=10,
    hidden_size=16,
    batch_first=True
)

classificador = nn.Linear(16, 3)

with torch.no_grad():
    saidas, estado_final = rnn(entrada)

    # MANY-TO-ONE:
    # Uma classificacao para a sequencia inteira.
    many_to_one = classificador(saidas[:, -1, :])

    # MANY-TO-MANY:
    # Uma classificacao para cada elemento.
    many_to_many = classificador(saidas)

print("ENTRADA:", entrada.shape)
print("ESTADOS OCULTOS:", saidas.shape)
print("MANY-TO-ONE:", many_to_one.shape)
print("MANY-TO-MANY:", many_to_many.shape)

assert many_to_one.shape == (1, 3)
assert many_to_many.shape == (1, 5, 3)

print("\nEXERCICIO EXECUTADO COM SUCESSO")