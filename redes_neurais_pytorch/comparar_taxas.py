import torch

X = torch.tensor([
    [2.0,  0.4],
    [1.0,  0.6],
    [3.0, -0.4]
])
y = torch.tensor([1.0, 1.0, 0.0])

taxas = [0.01, 0.1, 1.0]

for taxa in taxas:
    w = torch.tensor([-1.0, 4.0], requires_grad=True)
    b = torch.tensor(0.4, requires_grad=True)

    perda_fn = torch.nn.BCEWithLogitsLoss()
    otimizador = torch.optim.SGD([w, b], lr=taxa)

    perda_inicial = perda_fn(X @ w + b, y).item()

    for epoca in range(100):
        otimizador.zero_grad()
        perda = perda_fn(X @ w + b, y)
        perda.backward()
        otimizador.step()

    with torch.no_grad():
        perda_final = perda_fn(X @ w + b, y).item()

    print(f"Taxa de aprendizado: {taxa}")
    print(f"Perda inicial: {perda_inicial:.4f}")
    print(f"Perda após 100 épocas: {perda_final:.4f}")
    print("-" * 30)
