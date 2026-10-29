import torch

# 1. Criando tensores a partir de listas
lista = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(lista, dtype=torch.float32)
print("Tensor:", tensor)
print("Tipo:", tensor.dtype)
print("Formato:", tensor.shape)

# Conversão explícita para tipos comuns
print("Float:", torch.FloatTensor(lista).dtype)
print("Long:", torch.tensor(lista, dtype=torch.long).dtype)

# 2. Conversão de NumPy para tensor (preserva o tipo)
import numpy as np

array = np.random.rand(3, 4)
tensor_numpy = torch.from_numpy(array)
print("\nNumPy:", array.dtype)
print("Tensor convertido:", tensor_numpy.dtype)

# 3. Tensores inicializados com valores
uns = torch.ones(2, 3)
zeros = torch.zeros(4, 5)
aleatorio = torch.randn(3, 3)  # distribuição normal
print("\nUns:\n", uns)
print("Zeros:\n", zeros)
print("Aleatório:\n", aleatorio)

# Conversão de tensor para NumPy
array_convertido = aleatorio.numpy()
print("\nTipo convertido:", type(array_convertido))

# 4. Indexação, fatias e alteração de valores
aleatorio[0, 2] = -10
print("\nTensor modificado:\n", aleatorio)
print("Linhas 0 e 1:\n", aleatorio[0:2])
print("Coluna 2:\n", aleatorio[:, 2])
print("Elemento único:", aleatorio[0, 2])
print("Dimensões do elemento:", aleatorio[0, 2].dim())

# 5. Operações elemento a elemento
a = torch.randn(2, 3)
b = torch.ones(2, 3)
print("\nSoma elemento a elemento:\n", a + b)
print("Multiplicação elemento a elemento:\n", a * b)

# Produto de matrizes: dimensões internas devem ser compatíveis
print("Produto de matrizes:\n", a @ b.T)

# 6. Formato e redimensionamento
tensor_3d = torch.randn(2, 2, 3)
print("\nFormato original:", tensor_3d.shape)
print("Achatar:", tensor_3d.view(-1).shape)
print("Formato 4 x 3:", tensor_3d.view(4, 3).shape)
print("Manter primeira dimensão e achatar o restante:",
      tensor_3d.view(tensor_3d.size(0), -1).shape)

# 7. Escolha do dispositivo e envio do tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_device = torch.randn(3).to(device)
print("\nDispositivo disponível:", device)
print("Dispositivo do tensor:", tensor_device.device)
