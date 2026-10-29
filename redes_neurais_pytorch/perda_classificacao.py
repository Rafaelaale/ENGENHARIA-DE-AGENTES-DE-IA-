import torch

probabilidades = torch.tensor([0.5000, 0.8581, 0.0148])
rotulos = torch.tensor([1.0, 1.0, 0.0])

perda = torch.nn.functional.binary_cross_entropy(probabilidades, rotulos)
print("Perda:", perda.item())
