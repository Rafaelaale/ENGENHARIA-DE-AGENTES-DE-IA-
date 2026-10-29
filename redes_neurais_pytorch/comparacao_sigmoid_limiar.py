import torch
from torch import nn

sigmoide = nn.Sigmoid()

retornos = torch.tensor([-2.56, -1.0, 0.0, 0.5, 1.0, 2.56])
probabilidades = sigmoide(retornos)
decisoes = (probabilidades >= 0.5).int()

print("Retorno linear | Sigmoid | Decisão pelo limiar 0,5")
for retorno, probabilidade, decisao in zip(retornos, probabilidades, decisoes):
    print(f"{retorno.item():>14.2f} | {probabilidade.item():>7.4f} | classe {decisao.item()}")
