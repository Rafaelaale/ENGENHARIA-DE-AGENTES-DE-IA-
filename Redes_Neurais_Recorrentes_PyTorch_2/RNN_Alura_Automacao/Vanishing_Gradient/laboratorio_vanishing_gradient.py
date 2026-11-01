import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)

pasta = Path(__file__).resolve().parent

passos = list(range(1, 151))

def calcular_gradientes(peso, quantidade_passos):
    valores = []

    for n in quantidade_passos:
        x = torch.tensor(1.0, requires_grad=True)
        h = x

        for _ in range(n):
            h = peso * h

        h.backward()
        valores.append(abs(x.grad.item()))

    return valores

# Pesos menores que 1 provocam decaimento.
gradientes_desaparecendo = calcular_gradientes(0.8, passos)

# Pesos maiores que 1 provocam crescimento.
gradientes_explodindo = calcular_gradientes(1.2, passos)

assert gradientes_desaparecendo[-1] < gradientes_desaparecendo[0]
assert gradientes_explodindo[-1] > gradientes_explodindo[0]

print("VANISHING GRADIENT")
print("Gradiente inicial:", gradientes_desaparecendo[0])
print("Gradiente final:", gradientes_desaparecendo[-1])

print("\nEXPLODING GRADIENT")
print("Gradiente inicial:", gradientes_explodindo[0])
print("Gradiente final:", gradientes_explodindo[-1])

# Demonstracao de gradient clipping.
parametro = torch.nn.Parameter(torch.tensor(1.0))
perda = parametro * (1.2 ** 100)
perda.backward()

gradiente_antes = abs(parametro.grad.item())

torch.nn.utils.clip_grad_norm_(
    [parametro],
    max_norm=1.0
)

gradiente_depois = abs(parametro.grad.item())

assert gradiente_depois <= 1.00001

print("\nGRADIENT CLIPPING")
print("Antes:", gradiente_antes)
print("Depois:", gradiente_depois)

plt.figure(figsize=(11, 6))

plt.semilogy(
    passos,
    gradientes_desaparecendo,
    label="Vanishing (peso 0.8)"
)

plt.semilogy(
    passos,
    gradientes_explodindo,
    label="Exploding (peso 1.2)"
)

plt.xlabel("Passos temporais")
plt.ylabel("Magnitude do gradiente - escala logaritmica")
plt.title("Vanishing Gradient e Exploding Gradient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

grafico = pasta / "grafico_gradientes.png"
plt.savefig(grafico, dpi=160)
plt.close()

relatorio = pasta / "relatorio_gradientes.txt"

texto = f"""
LABORATORIO - VANISHING GRADIENT

Passos analisados: {len(passos)}

VANISHING:
Gradiente inicial: {gradientes_desaparecendo[0]:.8e}
Gradiente final: {gradientes_desaparecendo[-1]:.8e}

EXPLODING:
Gradiente inicial: {gradientes_explodindo[0]:.8e}
Gradiente final: {gradientes_explodindo[-1]:.8e}

GRADIENT CLIPPING:
Antes: {gradiente_antes:.8e}
Depois: {gradiente_depois:.8e}

RESULTADOS:
Vanishing gradient: APROVADO
Exploding gradient: APROVADO
Gradient clipping: APROVADO

OBSERVACAO:
Experimento escalar didatico.
Nao representa treinamento completo de uma RNN.
"""

relatorio.write_text(texto, encoding="utf-8")

print("\n3/3 TESTES APROVADOS")
print("Grafico:", grafico)
print("Relatorio:", relatorio)