import torch

# Dados: cada linha é um ponto [x1, x2]; rótulos são as classes corretas.
X = torch.tensor([
    [2.0,  0.4],
    [1.0,  0.6],
    [3.0, -0.4]
])
y = torch.tensor([1.0, 1.0, 0.0])

# Classificador linear: escore = w · x + b
w = torch.tensor([-1.0, 4.0], requires_grad=True)
b = torch.tensor(0.4, requires_grad=True)

# 1. Escore linear, probabilidades e classes antes do treinamento
with torch.no_grad():
    escores_iniciais = X @ w + b
    probabilidades_iniciais = torch.sigmoid(escores_iniciais)
    classes_iniciais = (probabilidades_iniciais >= 0.5).int()

print("ANTES DO TREINAMENTO")
print("Escores:", escores_iniciais)
print("Probabilidades:", probabilidades_iniciais)
print("Classes previstas:", classes_iniciais)

# 2. Perda e gradientes iniciais
funcao_perda = torch.nn.BCEWithLogitsLoss()
perda_inicial = funcao_perda(X @ w + b, y)
perda_inicial.backward()

print("\nPRIMEIRO CÁLCULO DOS GRADIENTES")
print("Perda:", perda_inicial.item())
print("Gradiente dos pesos:", w.grad)
print("Gradiente do viés:", b.grad)

# 3. Treinamento: atualizar pesos e viés para reduzir a perda
otimizador = torch.optim.SGD([w, b], lr=0.1)
perdas = []

for epoca in range(100):
    otimizador.zero_grad()
    escores = X @ w + b
    perda = funcao_perda(escores, y)
    perda.backward()
    otimizador.step()
    perdas.append(perda.item())

# 4. Resultado final
with torch.no_grad():
    escores_finais = X @ w + b
    probabilidades_finais = torch.sigmoid(escores_finais)
    classes_finais = (probabilidades_finais >= 0.5).int()
    perda_final = funcao_perda(escores_finais, y).item()

print("\nDEPOIS DO TREINAMENTO")
print(f"Perda inicial: {perdas[0]:.4f}")
print(f"Perda final:   {perda_final:.4f}")
print("Pesos treinados:", w)
print("Viés treinado:", b.item())
print("Escores:", escores_finais)
print("Probabilidades:", probabilidades_finais)
print("Classes previstas:", classes_finais)
print("Classes corretas: ", y.int())

# Fronteira de decisão: w1*x1 + w2*x2 + b = 0
print("\nFRONTEIRA DE DECISÃO")
print(f"{w[0].item():.4f}*x1 + {w[1].item():.4f}*x2 + {b.item():.4f} = 0")
