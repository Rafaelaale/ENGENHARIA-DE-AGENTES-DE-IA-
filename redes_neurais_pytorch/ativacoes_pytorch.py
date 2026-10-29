import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import make_classification

# Recria a distribuição usada na aula
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=30,
)

# Perceptron da demonstração: pesos 5 e 1; viés -0.4
perceptron = nn.Linear(2, 1)
with torch.no_grad():
    perceptron.weight.copy_(torch.tensor([[5.0, 1.0]]))
    perceptron.bias.copy_(torch.tensor([-0.4]))

sigmoide = nn.Sigmoid()
tangente_hiperbolica = nn.Tanh()
relu = nn.ReLU()
indices = [17, 21, 43, 66]
marcadores = ["^", "v", ">", "<"]
cores = ["red", "green", "blue", "gray"]

print("Pesos:", perceptron.weight.detach())
print("Viés:", perceptron.bias.detach())
print("\nComparação para os pontos selecionados:")

with torch.no_grad():
    for indice in indices:
        x = torch.tensor(X[indice], dtype=torch.float32)
        ret = perceptron(x)
        act_sigmoide = sigmoide(ret)
        act_limiar = int(ret.item() >= 0)

        print(
            f"X[{indice}] = {x.tolist()} | "
            f"retorno: {ret.item():.2f} | "
            f"limiar: {act_limiar} | "
            f"Sigmoid: {act_sigmoide.item():.4f} | "
            f"Tanh: {tangente_hiperbolica(ret).item():.4f} | "
            f"ReLU: {relu(ret).item():.4f}"
        )

# Plota a distribuição, os pontos escolhidos e a fronteira do perceptron
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.65)

for k, indice in enumerate(indices):
    plt.plot(
        X[indice, 0],
        X[indice, 1],
        marker=marcadores[k],
        color=cores[k],
        markersize=12,
        linestyle="none",
        label=f"X[{indice}]",
    )

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1 = [x1_min, x1_max]
x2 = [-(5.0 * valor - 0.4) for valor in x1]
plt.plot(x1, x2, "k--", label="Fronteira do perceptron")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Ativações no PyTorch: pontos selecionados")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    r"C:\Users\rafal\Desktop\redes_neurais_pytorch\ativacoes_pytorch.png",
    bbox_inches="tight",
)
plt.show()
