import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Gera 100 amostras com duas características
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=30
)

# Pesos e bias da fronteira de decisão
w1, w2, bias = -3, 5, 3

# Classifica os pontos conforme a fronteira linear
y_pred = (w1 * X[:, 0] + w2 * X[:, 1] + bias >= 0).astype(int)
acuracia = np.mean(y_pred == y) * 100

# Desenha as amostras, destacando X[10]
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.scatter(
    X[10, 0], X[10, 1],
    marker="^", s=180, color="red", label="X[10]"
)

# Desenha a fronteira: w1*x1 + w2*x2 + bias = 0
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
x2 = -(w1 * x1 + bias) / w2
plt.plot(x1, x2, "k--", label="Fronteira de decisão")

plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.title(f"Classificação linear — acurácia: {acuracia:.0f}%")
plt.legend()
plt.grid(True)

plt.savefig(
    "C:/Users/rafal/Desktop/redes_neurais_pytorch/exercicio_classificacao_linear.png",
    bbox_inches="tight"
)
plt.show()

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")
print(f"Classe de X[10]: {y[10]}")
print(f"Acurácia: {acuracia:.0f}%")
