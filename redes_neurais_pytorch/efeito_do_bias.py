import numpy as np
import matplotlib.pyplot as plt

w = np.array([-1, 4])
X = np.array([[2, 0.4], [1, 0.6], [3, -0.4]])
bs = [-2, 0.4, 2]
x = np.linspace(-2, 4, 200)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))

for ax, b in zip(axs, bs):
    scores = X @ w + b
    y = (-w[0] * x - b) / w[1]

    print(f"\nb = {b}: fronteira y = ({-w[0]}*x1 - {b})/{w[1]}")
    for i, score in enumerate(scores, start=1):
        print(f"  Ponto {i}: escore={score:.1f}, classe={int(score >= 0)}")

    ax.plot(x, y)
    ax.scatter(X[:, 0], X[:, 1],
               c=["blue" if score >= 0 else "red" for score in scores])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"b = {b}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid()

plt.tight_layout()
plt.savefig("efeito_do_bias.png")
print("\nGráfico salvo: efeito_do_bias.png")
