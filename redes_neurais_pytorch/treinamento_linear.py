import torch

X = torch.tensor([[2.0, 0.4], [1.0, 0.6], [3.0, -0.4]])
rotulos = torch.tensor([1.0, 1.0, 0.0])

w = torch.tensor([-1.0, 4.0], requires_grad=True)
b = torch.tensor(0.4, requires_grad=True)

otimizador = torch.optim.SGD([w, b], lr=0.1)
funcao_perda = torch.nn.BCEWithLogitsLoss()

with torch.no_grad():
    perda_inicial = funcao_perda(X @ w + b, rotulos).item()

for epoca in range(100):
    otimizador.zero_grad()
    escores = X @ w + b
    perda = funcao_perda(escores, rotulos)
    perda.backward()
    otimizador.step()

with torch.no_grad():
    perda_final = funcao_perda(X @ w + b, rotulos).item()
    probabilidades = torch.sigmoid(X @ w + b)
    classes = (probabilidades >= 0.5).int()

print(f"Perda inicial: {perda_inicial:.4f}")
print(f"Perda final:   {perda_final:.4f}")
print("Pesos treinados:", w)
print("Viés treinado:", b.item())
print("Probabilidades:", probabilidades)
print("Classes previstas:", classes)
