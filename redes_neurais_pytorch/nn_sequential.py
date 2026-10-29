import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Distribuição make_moons")
plt.show()

entrada = torch.from_numpy(X).float()

net = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

pred = net(entrada)

print(net)
print("Formato da entrada:", entrada.shape)
print("Formato da saída:", pred.shape)
print("Total de parâmetros:", sum(p.numel() for p in net.parameters()))
