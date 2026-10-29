import torch
from torch import nn

torch.manual_seed(42)

# Camada linear com 3 entradas e 1 saída
perceptron = nn.Linear(in_features=3, out_features=1)
print(perceptron)

# Mostrar os nomes e valores dos parâmetros
for nome, tensor in perceptron.named_parameters():
    print(nome, tensor.data)

# Entrada: x0, x1 e x2
x = torch.tensor([0.0, -1.0, 2.0])

# Forward: aplicar a camada linear à entrada
y = perceptron(x)

print("Entrada:", x)
print("Saída do perceptron:", y.item())