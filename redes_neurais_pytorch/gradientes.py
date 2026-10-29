import torch

X = torch.tensor([[2.0, 0.4], [1.0, 0.6], [3.0, -0.4]])
rotulos = torch.tensor([1.0, 1.0, 0.0])
w = torch.tensor([-1.0, 4.0], requires_grad=True)
b = torch.tensor(0.4, requires_grad=True)

probabilidades = torch.sigmoid(X @ w + b)
perda = torch.nn.functional.binary_cross_entropy(probabilidades, rotulos)
perda.backward()

print("Perda:", perda.item())
print("Gradiente dos pesos:", w.grad)
print("Gradiente do viés:", b.grad)
