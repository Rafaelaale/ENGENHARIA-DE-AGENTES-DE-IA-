import torch
from torch import nn

softmax = nn.Softmax(dim=0)
saidas = torch.tensor([2.0, -1.0, 0.5])
probabilidades = softmax(saidas)

camada = nn.Linear(in_features=2, out_features=3)

print("Saídas:", saidas)
print("Probabilidades:", probabilidades)
print("Soma das probabilidades:", probabilidades.sum())
print("Camada linear:", camada)
