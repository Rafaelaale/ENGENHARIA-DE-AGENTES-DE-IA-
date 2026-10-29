import torch

X = torch.tensor([[2.0, 0.4], [1.0, 0.6], [3.0, -0.4]])
w = torch.tensor([-1.0, 4.0])
b = 0.4

escores = X @ w + b
classes = (escores >= 0).int()

print("Escores:", escores)
print("Classes:", classes)
