"""
CURSO 03 - CNN COM PYTORCH
AULA 01 - MOTIVACAO

Laboratorio baseado nos conceitos da transcricao
da aula Motivacao da Alura.

Complementos praticos:
- kernels
- mapas de caracteristicas
- arquitetura Conv2d
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

PASTA = Path("resultados")
PASTA.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def secao(titulo):
    print("\n" + "=" * 65)
    print(titulo)
    print("=" * 65)


# ---------------------------------------------------
# 1. IMAGEM = MATRIZ DE PIXELS
# ---------------------------------------------------

secao("1. IMAGEM COMO MATRIZ DE PIXELS")

imagem = np.zeros((256, 256), dtype=np.float32)

# Desenhamos uma imagem artificial:
# fundo preto, quadrado cinza e area branca.
imagem[40:210, 40:210] = 0.5
imagem[85:170, 85:170] = 1.0

print("Dimensoes:", imagem.shape)
print("Numero de pixels:", imagem.size)
print("Pixel preto:", imagem[0, 0])
print("Pixel cinza:", imagem[50, 50])
print("Pixel branco:", imagem[100, 100])

plt.figure(figsize=(6, 5))
plt.imshow(imagem, cmap="gray", vmin=0, vmax=1)
plt.title("Imagem como matriz 256 x 256")
plt.colorbar(label="Intensidade do pixel")
plt.tight_layout()
plt.savefig(PASTA / "01_matriz_pixels.png")
plt.close()


# ---------------------------------------------------
# 2. ZOOM EM UMA REGIAO
# ---------------------------------------------------

secao("2. ZOOM NOS PIXELS")

regiao = imagem[78:98, 78:98]

plt.figure(figsize=(7, 6))
plt.imshow(regiao, cmap="gray", vmin=0, vmax=1)

for linha in range(regiao.shape[0]):
    for coluna in range(regiao.shape[1]):
        plt.text(
            coluna,
            linha,
            f"{regiao[linha, coluna]:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="red"
        )

plt.title("Cada pixel possui um valor numerico")
plt.tight_layout()
plt.savefig(PASTA / "02_zoom_pixels.png")
plt.close()


# ---------------------------------------------------
# 3. LINEARIZACAO DA IMAGEM PARA MLP
# ---------------------------------------------------

secao("3. IMAGEM 2D PARA VETOR 1D")

vetor = imagem.flatten()

print("Formato original:", imagem.shape)
print("Formato linearizado:", vetor.shape)
print("Primeiros 10 valores:", vetor[:10])

# A MLP recebe um vetor de caracteristicas.
# A imagem 256x256 vira 65536 entradas.


# ---------------------------------------------------
# 4. QUANTIDADE DE PESOS EM UMA MLP
# ---------------------------------------------------

secao("4. CUSTO DE UMA MLP")

n = 256
entradas = n * n
neuronios = n * n

pesos_mlp = entradas * neuronios
bias_mlp = neuronios
parametros_mlp = pesos_mlp + bias_mlp

memoria_gb = (pesos_mlp * 4) / (1024 ** 3)

print("Entradas:", entradas)
print("Neuronios na camada seguinte:", neuronios)
print("Pesos:", f"{pesos_mlp:,}")
print("Bias:", f"{bias_mlp:,}")
print("Total de parametros:", f"{parametros_mlp:,}")
print("Memoria dos pesos em GiB:", round(memoria_gb, 2))

# Nao vamos instanciar essa camada gigante.
# Ela consumiria muita memoria.


# ---------------------------------------------------
# 5. KERNEL DE CONVOLUCAO
# ---------------------------------------------------

secao("5. KERNEL 3 x 3")

kernel_bordas = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=torch.float32
)

print("Kernel:")
print(kernel_bordas)

# Conv2d espera:
# [lote, canais, altura, largura]

tensor_imagem = torch.from_numpy(imagem)
entrada = tensor_imagem.unsqueeze(0).unsqueeze(0)

filtro = kernel_bordas.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    saida_bordas = F.conv2d(
        entrada,
        filtro,
        padding=1
    )

print("Entrada:", tuple(entrada.shape))
print("Saida:", tuple(saida_bordas.shape))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap="gray")
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    saida_bordas[0, 0].numpy(),
    cmap="gray"
)
plt.title("Bordas detectadas")
plt.axis("off")

plt.tight_layout()
plt.savefig(PASTA / "03_convolucao_bordas.png")
plt.close()


# ---------------------------------------------------
# 6. CNN: COMPARTILHAMENTO DE PESOS
# ---------------------------------------------------

secao("6. MLP VERSUS CNN")

# Camada convolucional:
# 1 canal de entrada
# 8 filtros de saida
# kernel 3x3

cnn_camada = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)

parametros_cnn = sum(
    p.numel() for p in cnn_camada.parameters()
)

print("Parametros da MLP:", f"{parametros_mlp:,}")
print("Parametros da Conv2d:", parametros_cnn)
print(
    "Razao MLP / Conv2d:",
    round(parametros_mlp / parametros_cnn)
)

# A comparacao ilustra a economia de parametros,
# embora as camadas tenham capacidades diferentes.


# ---------------------------------------------------
# 7. MAPAS DE CARACTERISTICAS
# ---------------------------------------------------

secao("7. MAPAS DE CARACTERISTICAS")

with torch.no_grad():
    mapas = cnn_camada(entrada)

print("Formato dos mapas:", tuple(mapas.shape))

fig, eixos = plt.subplots(2, 4, figsize=(12, 6))

for i, eixo in enumerate(eixos.flat):
    eixo.imshow(
        mapas[0, i].numpy(),
        cmap="viridis"
    )
    eixo.set_title(f"Filtro {i + 1}")
    eixo.axis("off")

plt.suptitle("Oito mapas de caracteristicas")
plt.tight_layout()
plt.savefig(PASTA / "04_mapas_caracteristicas.png")
plt.close()

# Os filtros acima ainda nao foram treinados.
# Portanto, suas respostas nao representam
# caracteristicas aprendidas de um conjunto de dados.


# ---------------------------------------------------
# 8. ARQUITETURA CNN COMPLETA
# ---------------------------------------------------

secao("8. ARQUITETURA CNN COM PYTORCH")


class MinhaCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.classificador = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reduz cada mapa espacial a um valor.
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, 1)
        return self.classificador(x)


modelo = MinhaCNN()

with torch.no_grad():
    previsao = modelo(entrada)

total = sum(p.numel() for p in modelo.parameters())

print(modelo)
print("Total de parametros:", total)
print("Formato da previsao:", tuple(previsao.shape))
print("Saida antes do treinamento:", previsao)

# Esta CNN nao foi treinada.
# Seus resultados ainda nao sao classificacoes confiaveis.


# ---------------------------------------------------
# 9. DESAFIO DE ESTUDO
# ---------------------------------------------------

secao("9. DESAFIOS")

print("""
1. O que significa um pixel de intensidade 0?
2. Por que linearizamos a imagem para usar uma MLP?
3. Por que uma camada totalmente conectada pode
   exigir tantos pesos?
4. O que um kernel 3x3 faz?
5. Por que a CNN usa menos parametros?
6. O que sao mapas de caracteristicas?
7. Qual a diferenca entre um filtro aleatorio
   e um filtro treinado?
8. Por que usamos MaxPool2d?
""")

print("\nLaboratorio finalizado.")
print("Graficos salvos em:", PASTA.resolve())
