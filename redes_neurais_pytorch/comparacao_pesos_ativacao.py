import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=30,
)

viés = -0.9131562
configuracoes = [
    ([5.0, 1.0], "Pesos [5, 1]"),
    ([1.0, 5.0], "Pesos [1, 5]"),
    ([2.0, 2.0], "Pesos [2, 2]"),
]

x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)

fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for eixo, (pesos, titulo) in zip(eixos, configuracoes):
    w1, w2 = pesos
    retorno = X @ np.array(pesos) + viés
    probabilidade = 1 / (1 + np.exp(-retorno))

    eixo.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr",
                 edgecolors="k", alpha=0.65)
    eixo.plot(x1, -(w1 * x1 + viés) / w2, color="black",
              linewidth=2, label="fronteira: retorno = 0")
    eixo.set_title(titulo)
    eixo.set_xlabel("Característica 1")
    eixo.grid(alpha=0.2)
    eixo.legend(fontsize=8)

    print(f"\n{titulo}; viés = {viés:.4f}")
    for indice in [17, 21, 43, 66]:
        print(
            f"X[{indice}]: retorno = {retorno[indice]:.4f} | "
            f"Sigmoid = {probabilidade[indice]:.4f}"
        )

eixos[0].set_ylabel("Característica 2")
fig.suptitle("Efeito dos pesos na fronteira de decisão")
fig.tight_layout()

saida = r"C:\Users\rafal\Desktop\redes_neurais_pytorch\comparacao_pesos_ativacao.png"
fig.savefig(saida, dpi=150, bbox_inches="tight")
print(f"\nGráfico salvo em: {saida}")
plt.show()
