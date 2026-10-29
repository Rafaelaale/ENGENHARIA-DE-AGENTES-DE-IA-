import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2, 0.4], [1, 0.6], [3, -0.4]])
configuracoes = [([-1, 4], 0.4), ([1, 4], 0.4), ([-1, 2], 0.4)]
x = np.linspace(-2, 4, 200)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))

for ax, (pesos, bias) in zip(axs, configuracoes):
    w = np.array(pesos)
    escores = X @ w + bias
    y = (-w[0] * x - bias) / w[1]

    print(f"\nPesos={pesos}, viés={bias}")
    for i, escore in enumerate(escores, start=1):
        print(f"Ponto {i}: escore={escore:.1f}, classe={int(escore >= 0)}")

    ax.plot(x, y)
    ax.scatter(X[:, 0], X[:, 1],
               c=["blue" if escore >= 0 else "red" for escore in escores])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"w={pesos}, b={bias}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid()

plt.tight_layout()
plt.savefig("efeito_dos_pesos.png")
print("\nGráfico salvo: efeito_dos_pesos.png")
