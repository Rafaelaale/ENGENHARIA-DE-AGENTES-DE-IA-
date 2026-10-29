import torch
from torch import nn

# XOR: 2 entradas, 6 neurônios escondidos e 1 saída
rede_xor = nn.Sequential(
    nn.Linear(2, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
    nn.Sigmoid()
)

# Três classes: 2 entradas, 6 neurônios escondidos e 3 saídas
rede_tres_classes = nn.Sequential(
    nn.Linear(2, 6),
    nn.ReLU(),
    nn.Linear(6, 3)
)

entrada = torch.tensor([[1.0, -1.0]])

probabilidade_xor = rede_xor(entrada)
saidas_classes = rede_tres_classes(entrada)
probabilidades_classes = torch.softmax(saidas_classes, dim=1)

print("Probabilidade de saída do XOR:", probabilidade_xor)
print("Saídas para três classes:", saidas_classes)
print("Probabilidades para três classes:", probabilidades_classes)
print("Soma das probabilidades:", probabilidades_classes.sum(dim=1))
