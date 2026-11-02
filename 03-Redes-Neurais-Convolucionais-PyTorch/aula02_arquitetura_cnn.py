"""
ALURA - REDES NEURAIS CONVOLUCIONAIS
AULA 02 - ARQUITETURA DA CNN

Experimentos:
1. Imagens RGB como volumes 3D
2. Diferenca entre MLP e CNN
3. Convolucao e campo receptivo
4. Pooling e subamostragem
5. Arquitetura CNN completa
6. Aprendizado hierarquico
7. Mapas de caracteristicas
8. Inferencia e classificacao
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

RESULTADOS = Path("resultados")
RESULTADOS.mkdir(exist_ok=True)


def titulo(texto):
    print("\n" + "=" * 65)
    print(texto)
    print("=" * 65)


# ===================================================
# 1. IMAGEM RGB: ALTURA, LARGURA E CANAIS
# ===================================================

titulo("1. IMAGEM RGB COMO VOLUME 3D")

# Imagem artificial RGB de 128 x 128.
imagem = np.zeros((128, 128, 3), dtype=np.float32)

# Quadrado vermelho
imagem[10:60, 10:60, 0] = 1.0

# Quadrado verde
imagem[35:95, 45:105, 1] = 1.0

# Quadrado azul
imagem[70:120, 15:75, 2] = 1.0

print("Formato NumPy:", imagem.shape)
print("Total de valores:", imagem.size)
print("Canais: vermelho, verde e azul")

fig, eixos = plt.subplots(1, 4, figsize=(14, 4))

eixos[0].imshow(imagem)
eixos[0].set_title("Imagem RGB")

nomes = ["Vermelho", "Verde", "Azul"]

for i in range(3):
    eixos[i + 1].imshow(
        imagem[:, :, i],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    eixos[i + 1].set_title(nomes[i])

for eixo in eixos:
    eixo.axis("off")

plt.tight_layout()
plt.savefig(RESULTADOS / "05_canais_rgb.png")
plt.close()


# ===================================================
# 2. FORMATO DO TENSOR NO PYTORCH
# ===================================================

titulo("2. FORMATO DOS TENSORES")

# NumPy: altura, largura, canais
# PyTorch Conv2d: lote, canais, altura, largura

tensor = torch.from_numpy(imagem)
entrada = tensor.permute(2, 0, 1).unsqueeze(0)

print("NumPy (H, W, C):", imagem.shape)
print("PyTorch (N, C, H, W):", tuple(entrada.shape))

# Para uma imagem RGB 256x256:
valores_rgb = 256 * 256 * 3
print("Valores em RGB 256x256:", valores_rgb)


# ===================================================
# 3. CONVOLUCAO: TRANSFORMACAO
# ===================================================

titulo("3. CONVOLUCAO E MAPAS DE CARACTERISTICAS")

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    padding=1
)

with torch.no_grad():
    mapas1 = F.relu(conv1(entrada))

print("Entrada:", tuple(entrada.shape))
print("Saida Conv1:", tuple(mapas1.shape))

print(
    "Parametros Conv1:",
    sum(p.numel() for p in conv1.parameters())
)

# Cada filtro recebe os 3 canais RGB.
# Sao 6 filtros, cada um com 3x3x3 pesos.
# Mais 1 bias por filtro:
# 6 * (3 * 3 * 3 + 1) = 168 parametros.


# ===================================================
# 4. POOLING: SUBAMOSTRAGEM
# ===================================================

titulo("4. POOLING E REDUCAO ESPACIAL")

pool = nn.MaxPool2d(kernel_size=2, stride=2)

with torch.no_grad():
    mapas_pool = pool(mapas1)

print("Antes do pooling:", tuple(mapas1.shape))
print("Depois do pooling:", tuple(mapas_pool.shape))

# Pooling reduz altura e largura.
# Nao reduz automaticamente o numero de canais.
# MaxPool2d nao possui pesos treinaveis.


# ===================================================
# 5. CNN COMPLETA
# ===================================================

titulo("5. CONSTRUINDO UMA CNN COMPLETA")


class ArquiteturaCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 18, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # Reduz cada mapa a 1 valor.
        self.media_global = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(18, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, mostrar=False):

        if mostrar:
            print("Entrada: ", tuple(x.shape))

        x = F.relu(self.conv1(x))

        if mostrar:
            print("Conv1:   ", tuple(x.shape))

        x = self.pool(x)

        if mostrar:
            print("Pool1:   ", tuple(x.shape))

        x = F.relu(self.conv2(x))

        if mostrar:
            print("Conv2:   ", tuple(x.shape))

        x = self.pool(x)

        if mostrar:
            print("Pool2:   ", tuple(x.shape))

        x = F.relu(self.conv3(x))

        if mostrar:
            print("Conv3:   ", tuple(x.shape))

        x = self.pool(x)

        if mostrar:
            print("Pool3:   ", tuple(x.shape))

        x = self.media_global(x)

        if mostrar:
            print("Media:   ", tuple(x.shape))

        x = torch.flatten(x, start_dim=1)

        if mostrar:
            print("Flatten: ", tuple(x.shape))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if mostrar:
            print("Saida:   ", tuple(x.shape))

        return x


modelo = ArquiteturaCNN()

print(modelo)

print(
    "Total de parametros:",
    sum(p.numel() for p in modelo.parameters())
)

with torch.no_grad():
    saida = modelo(entrada, mostrar=True)

print("Logits:", saida)


# ===================================================
# 6. APRENDIZADO HIERARQUICO
# ===================================================

titulo("6. APRENDIZADO HIERARQUICO")

# Capturamos as respostas das tres convolucoes.
# Os filtros ainda sao aleatorios: nao houve treino.

ativacoes = {}


def salvar_ativacao(nome):

    def hook(modulo, entradas, saida):
        ativacoes[nome] = F.relu(saida.detach())

    return hook


h1 = modelo.conv1.register_forward_hook(
    salvar_ativacao("baixo_nivel")
)

h2 = modelo.conv2.register_forward_hook(
    salvar_ativacao("medio_nivel")
)

h3 = modelo.conv3.register_forward_hook(
    salvar_ativacao("alto_nivel")
)

with torch.no_grad():
    modelo(entrada)

h1.remove()
h2.remove()
h3.remove()

for nome, mapa in ativacoes.items():
    print(nome, tuple(mapa.shape))


# ===================================================
# 7. VISUALIZACAO DOS MAPAS
# ===================================================

titulo("7. VISUALIZANDO MAPAS DE CARACTERISTICAS")

fig, eixos = plt.subplots(3, 6, figsize=(15, 8))

camadas = [
    ("baixo_nivel", "Conv1"),
    ("medio_nivel", "Conv2"),
    ("alto_nivel", "Conv3")
]

for linha, (nome, descricao) in enumerate(camadas):

    mapas = ativacoes[nome][0].numpy()

    for coluna in range(6):
        eixo = eixos[linha, coluna]

        eixo.imshow(
            mapas[coluna],
            cmap="viridis"
        )

        eixo.set_title(
            f"{descricao} - mapa {coluna + 1}",
            fontsize=9
        )

        eixo.axis("off")

plt.suptitle(
    "Mapas de caracteristicas - rede nao treinada"
)

plt.tight_layout()
plt.savefig(
    RESULTADOS / "06_hierarquia_caracteristicas.png"
)
plt.close()


# ===================================================
# 8. CLASSIFICACAO
# ===================================================

titulo("8. INFERENCIA COM A CNN")

with torch.no_grad():
    logits = modelo(entrada)
    probabilidades = torch.softmax(logits, dim=1)

print("Logits:", logits)
print("Probabilidades:", probabilidades)
print(
    "Classe prevista (sem treinamento):",
    probabilidades.argmax(dim=1).item()
)

# IMPORTANTE:
# A rede nao foi treinada.
# Esta classe nao tem significado real.
# O objetivo e demonstrar a arquitetura.


# ===================================================
# 9. RESUMO
# ===================================================

titulo("9. CONCLUSOES")

print("""
A CNN processa imagens como tensores.

Conv2d:
- Aprende transformacoes locais.
- Produz mapas de caracteristicas.
- Compartilha pesos espacialmente.

MaxPool2d:
- Reduz altura e largura.
- Preserva a quantidade de canais.
- Nao possui parametros treinaveis.

Camadas profundas:
- Podem aprender caracteristicas mais abstratas.
- Dependem de treinamento para isso acontecer.

Fully Connected:
- Combina as caracteristicas finais.
- Produz a saida para classificacao ou regressao.

ATENCAO:
Este laboratorio demonstra a arquitetura.
Ainda nao treinamos a rede.
""")

print("Laboratorio concluido.")
print("Graficos em:", RESULTADOS.resolve())
