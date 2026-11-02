"""
ALURA - CNN COM PYTORCH
AULA 04 - APLICACOES: CLASSIFICACAO

Objetivos:
- Carregar o MNIST
- Entender imagens e rotulos
- Converter imagens em tensores
- Visualizar 10 exemplos
- Construir uma CNN com 10 saidas

A CNN sera criada, mas ainda nao treinada.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


torch.manual_seed(42)

PASTA_DADOS = Path("dados")
PASTA_RESULTADOS = Path("resultados")

PASTA_DADOS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)


def secao(nome):
    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)


# -------------------------------------------------
# 1. CARREGANDO O MNIST
# -------------------------------------------------

secao("1. CARREGANDO O DATASET MNIST")

transformacao = transforms.ToTensor()

mnist = datasets.MNIST(
    root=str(PASTA_DADOS),
    train=False,
    transform=transformacao,
    download=True
)

print("Dataset:", type(mnist))
print("Quantidade de imagens:", len(mnist))
print("Classes:", mnist.classes)


# -------------------------------------------------
# 2. DADO E ROTULO
# -------------------------------------------------

secao("2. IMAGEM E ROTULO")

dado, rotulo = mnist[0]

print("Tipo da imagem:", type(dado))
print("Tipo do rotulo:", type(rotulo))
print("Formato da imagem:", tuple(dado.shape))
print("Rotulo:", rotulo)

print("Menor valor do tensor:", dado.min().item())
print("Maior valor do tensor:", dado.max().item())

# Formato PyTorch:
# (canais, altura, largura)
#
# MNIST:
# (1, 28, 28)
#
# 1 canal porque a imagem e em escala de cinza.


# -------------------------------------------------
# 3. VISUALIZANDO OS DEZ PRIMEIROS DIGITOS
# -------------------------------------------------

secao("3. VISUALIZANDO DEZ IMAGENS")

fig, eixos = plt.subplots(
    1, 10,
    figsize=(15, 4)
)

for i in range(10):

    imagem, classe = mnist[i]

    eixos[i].imshow(
        imagem[0],
        cmap="gray"
    )

    eixos[i].set_title(str(classe))
    eixos[i].axis("off")

plt.suptitle("MNIST - Dez primeiros exemplos")
plt.tight_layout()

arquivo = (
    PASTA_RESULTADOS /
    "08_mnist_dez_primeiros.png"
)

plt.savefig(arquivo)
plt.close()

print("Grafico salvo:", arquivo)


# -------------------------------------------------
# 4. COMO UMA CNN RECEBE A IMAGEM
# -------------------------------------------------

secao("4. ADICIONANDO A DIMENSAO DE LOTE")

# Um exemplo individual:
# (1, 28, 28)
#
# Uma CNN espera:
# (lote, canais, altura, largura)

entrada = dado.unsqueeze(0)

print("Imagem individual:", tuple(dado.shape))
print("Imagem com lote:", tuple(entrada.shape))


# -------------------------------------------------
# 5. CNN PARA CLASSIFICAR 10 DIGITOS
# -------------------------------------------------

secao("5. ARQUITETURA CNN PARA MNIST")


class CNN_MNIST(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        # Entrada 28x28
        # Pool1 -> 14x14
        # Pool2 -> 7x7
        # 16 mapas de 7x7

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64
        )

        # Uma saida para cada digito de 0 a 9
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)


modelo = CNN_MNIST()

print(modelo)

total = sum(
    parametro.numel()
    for parametro in modelo.parameters()
)

print("Total de parametros:", total)
print("Neuronios da ultima camada:", 10)


# -------------------------------------------------
# 6. EXECUTANDO A REDE SEM TREINAMENTO
# -------------------------------------------------

secao("6. SAIDA DA CNN")

modelo.eval()

with torch.no_grad():

    logits = modelo(entrada)

    probabilidades = torch.softmax(
        logits,
        dim=1
    )

print("Formato da saida:", tuple(logits.shape))
print("Logits:", logits)

print("\nProbabilidades por classe:")

for classe, probabilidade in enumerate(
    probabilidades[0]
):
    print(
        f"Digito {classe}: "
        f"{probabilidade.item():.2%}"
    )

classe_prevista = probabilidades.argmax(
    dim=1
).item()

print("\nRotulo verdadeiro:", rotulo)
print("Previsao sem treinamento:", classe_prevista)

print(
    "\nATENCAO: a CNN ainda nao foi treinada."
)
print(
    "A previsao nao demonstra reconhecimento real."
)


# -------------------------------------------------
# 7. CONCLUSAO
# -------------------------------------------------

secao("7. CONCLUSAO")

print("""
CLASSIFICACAO DE IMAGENS

Entrada:
- Imagem de um digito
- Tensor com formato (1, 28, 28)

Rotulo:
- Inteiro de 0 a 9

CNN:
- Extrai mapas de caracteristicas
- Reduz dimensoes com pooling
- Combina caracteristicas nas camadas finais

Saida:
- 10 valores, um para cada classe

PROXIMO PASSO:
Treinar a CNN para reconhecer os digitos.
""")

print("AULA 04 EXECUTADA COM SUCESSO!")
