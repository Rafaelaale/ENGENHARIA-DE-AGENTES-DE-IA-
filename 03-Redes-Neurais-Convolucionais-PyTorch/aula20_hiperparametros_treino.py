# ==========================================================
# ALURA - AULA 20: HIPERPARAMETROS DE TREINO
# LeNet-5 / CIFAR-10
#
# Nesta aula NAO executamos treinamento.
# Nao usamos backward() nem optimizer.step().
# ==========================================================

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Reutilizar a CNN construida na Aula 19.
from aula19_cnn_do_zero import (
    construir_lenet_rgb,
    CLASSES
)


# ==========================================================
# 1 - CONFIGURACAO DO PROJETO
# ==========================================================

BASE = Path(__file__).resolve().parent

RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

torch.manual_seed(20)


def salvar(fig, nome):

    destino = RESULTADOS / nome

    fig.tight_layout()

    fig.savefig(
        destino,
        dpi=140
    )

    plt.close(fig)

    print(
        "Imagem salva:",
        destino
    )


print("=" * 67)
print("AULA 20 - HIPERPARAMETROS DE TREINO | ALURA")
print("=" * 67)


# ==========================================================
# 2 - DEFINIR OS HIPERPARAMETROS
# ==========================================================

# Detectar se existe GPU CUDA disponivel.
#
# No ambiente CPU, a rede tambem funciona.
#
# A transcricao menciona 10e-3:
# 10e-3 = 0.01
# 1e-3  = 0.001
#
# Sao duas taxas diferentes.

args = {

    "device": torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),

    "batch_size": 32,

    "learning_rate": 10e-3,

    "weight_decay": 5e-4,

    "epoch_num": 100

}


print("\n1 - HIPERPARAMETROS DEFINIDOS")

for nome, valor in args.items():

    print(
        f"{nome:17s}: {valor}"
    )


print(
    "GPU CUDA disponivel:",
    torch.cuda.is_available()
)

print(
    "Nota: 10e-3 = 0.01; 1e-3 = 0.001."
)


# ==========================================================
# 3 - ENVIAR A REDE PARA O DEVICE
# ==========================================================

# Importamos a arquitetura da Aula 19.
# Nao precisamos escrever a LeNet novamente.

net = construir_lenet_rgb()

net = net.to(
    args["device"]
)


print("\n2 - REDE NEURAL NO DISPOSITIVO")

print(
    "Modelo:",
    type(net).__name__
)

print(
    "Primeiro parametro em:",
    next(net.parameters()).device
)

print(
    "Total parametros:",
    sum(
        p.numel()
        for p in net.parameters()
    )
)


assert (
    next(net.parameters()).device.type
    == args["device"].type
)


# ==========================================================
# 4 - CARREGAR CIFAR-10 REAL
# ==========================================================

# O dataset ja foi localizado em:
#
# dados/cifar-10-batches-py
#
# Nesta aula exigimos o CIFAR-10 real.
# Nao usamos FakeData.

pasta_dados = BASE / "dados"

transformacao = transforms.ToTensor()


try:

    treino = datasets.CIFAR10(

        root=str(pasta_dados),

        train=True,

        download=False,

        transform=transformacao

    )

    teste = datasets.CIFAR10(

        root=str(pasta_dados),

        train=False,

        download=False,

        transform=transformacao

    )

except (
    RuntimeError,
    OSError,
    FileNotFoundError
) as erro:

    raise SystemExit(

        "CIFAR-10 real nao foi carregado em ./dados. "
        "Verifique o dataset antes de treinar. "
        "Detalhes: " + str(erro)

    ) from erro


assert (
    len(treino) == 50000
    and len(teste) == 10000
), "Tamanho inesperado do CIFAR-10 real."


# ==========================================================
# 5 - CONFIGURAR O DATALOADER
# ==========================================================

# O DataLoader separa o conjunto de dados
# em lotes de 32 imagens.
#
# shuffle=True embaralha as amostras
# do treinamento.
#
# num_workers=0 evita problemas com
# subprocessos no Windows.

carregador_treino = DataLoader(

    treino,

    batch_size=args["batch_size"],

    shuffle=True,

    num_workers=0

)


carregador_teste = DataLoader(

    teste,

    batch_size=args["batch_size"],

    shuffle=False,

    num_workers=0

)


# Obter somente um batch para verificacao.

imagens, rotulos = next(
    iter(carregador_treino)
)


print("\n3 - CIFAR-10 REAL E DATALOADER")

print(
    "Treinamento:",
    len(treino),
    "imagens;",
    len(carregador_treino),
    "lotes por epoca"
)

print(
    "Teste:",
    len(teste),
    "imagens;",
    len(carregador_teste),
    "lotes"
)

print(
    "Um lote:",
    tuple(imagens.shape)
)

print(
    "Rotulos:",
    tuple(rotulos.shape)
)


assert tuple(imagens.shape) == (
    32, 3, 32, 32
)

assert tuple(rotulos.shape) == (
    32,
)


print(
    "VALIDACAO CIFAR-10 E DATALOADER: OK"
)


# ==========================================================
# 6 - DEFINIR A FUNCAO DE PERDA
# ==========================================================

# Classificacao com 10 categorias:
#
# CrossEntropyLoss recebe:
# 1. Logits produzidos pela CNN.
# 2. Rotulos verdadeiros.
#
# Nao devemos aplicar Softmax
# antes de CrossEntropyLoss.

criterion = nn.CrossEntropyLoss()

criterion = criterion.to(
    args["device"]
)


# ==========================================================
# 7 - CONFIGURAR O OTIMIZADOR ADAM
# ==========================================================

# O Adam sera responsavel por atualizar
# os pesos durante o treinamento.
#
# Aqui apenas o configuramos.

optimizer = optim.Adam(

    net.parameters(),

    lr=args["learning_rate"],

    weight_decay=args["weight_decay"]

)


print("\n4 - FUNCAO DE PERDA E OTIMIZADOR")

print(
    "Funcao de perda:",
    criterion
)

print(
    "Otimizador:",
    optimizer.__class__.__name__
)

print(
    "Learning rate:",
    optimizer.param_groups[0]["lr"]
)

print(
    "Weight decay:",
    optimizer.param_groups[0]["weight_decay"]
)


assert math.isclose(

    optimizer.param_groups[0]["lr"],

    args["learning_rate"]

)


assert math.isclose(

    optimizer.param_groups[0]["weight_decay"],

    args["weight_decay"]

)


assert len(optimizer.state) == 0


# ==========================================================
# 8 - VERIFICAR A LOSS SEM TREINAR
# ==========================================================

# Vamos executar apenas um forward.
#
# Isso serve para conferir que:
#
# - Os dados chegam a rede.
# - A rede produz 10 logits.
# - Os rotulos estao corretos.
# - A CrossEntropyLoss funciona.
#
# Nenhum peso sera atualizado.

net.eval()


# Dados e rede devem estar
# no mesmo dispositivo.

imagens_device = imagens.to(
    args["device"]
)

rotulos_device = rotulos.to(
    args["device"]
)


with torch.no_grad():

    logits = net(
        imagens_device
    )

    loss_inicial = criterion(
        logits,
        rotulos_device
    )

    # Softmax apenas para visualizar.
    probabilidades = torch.softmax(
        logits,
        dim=1
    )


assert tuple(logits.shape) == (
    32, 10
)


assert (
    loss_inicial.ndim == 0
    and torch.isfinite(loss_inicial)
)


torch.testing.assert_close(

    probabilidades.sum(dim=1),

    torch.ones(
        32,
        device=args["device"]
    ),

    rtol=0,

    atol=1e-6

)


# Nenhum passo do Adam foi realizado.

assert len(optimizer.state) == 0


print(
    "\n5 - PASSAGEM DE VERIFICACAO (SEM TREINAR)"
)

print(
    "Logits:",
    tuple(logits.shape)
)

print(
    "Rotulos:",
    tuple(rotulos.shape)
)

print(
    "CrossEntropyLoss inicial:",
    round(
        float(loss_inicial.item()),
        5
    )
)

print(
    "Soma softmax do exemplo 1:",
    round(
        float(
            probabilidades[0].sum().item()
        ),
        6
    )
)

print(
    "Estado do Adam vazio:",
    len(optimizer.state) == 0
)

print(
    "VALIDACAO DA LOSS COM LOGITS: OK"
)


# ==========================================================
# 9 - GRAFICO DOS HIPERPARAMETROS
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10.5, 5.2)
)

ax.axis("off")


linhas = [

    (
        "Hardware",
        str(args["device"])
    ),

    (
        "Mini-batch",
        str(args["batch_size"]) + " imagens"
    ),

    (
        "Loss",
        "CrossEntropyLoss (logits + rotulos)"
    ),

    (
        "Otimizador",
        "Adam"
    ),

    (
        "Taxa de aprendizado",
        str(args["learning_rate"])
    ),

    (
        "Weight decay",
        str(args["weight_decay"])
    ),

    (
        "Epocas configuradas",
        str(args["epoch_num"]) + " (nao executadas)"
    ),

    (
        "Dataset",
        "CIFAR-10 REAL: 50.000 treino / 10.000 teste"
    )

]


for indice, (nome, valor) in enumerate(linhas):

    y = 0.92 - indice * 0.115

    ax.text(

        0.04, y, nome,

        fontsize=11,

        weight="bold",

        va="center",

        transform=ax.transAxes

    )

    ax.text(

        0.39, y, valor,

        fontsize=11,

        va="center",

        transform=ax.transAxes

    )

    ax.plot(

        [0.03, 0.98],

        [y - 0.047, y - 0.047],

        lw=0.55,

        alpha=0.45,

        transform=ax.transAxes

    )


ax.set_title(
    "Aula 20 - Hiperparametros de treino",
    fontsize=14
)


salvar(
    fig,
    "80_hiperparametros_treino.png"
)


# ==========================================================
# 10 - GRAFICO DA LOSS INICIAL
# ==========================================================

# A previsao da rede ainda nao e
# confiavel porque nao houve treinamento.

probs_np = (
    probabilidades[0]
    .cpu()
    .numpy()
)

rotulo_0 = int(
    rotulos[0].item()
)


fig, ax = plt.subplots(
    figsize=(11, 5)
)


barras = ax.bar(
    np.arange(10),
    probs_np
)

# Destacar a classe verdadeira.

barras[rotulo_0].set_edgecolor(
    "black"
)

barras[rotulo_0].set_linewidth(
    2.5
)


ax.set_xticks(

    np.arange(10),

    CLASSES,

    rotation=30,

    ha="right"

)


ax.set_ylim(

    0,

    max(
        0.25,
        float(probs_np.max()) * 1.25
    )

)


ax.set_ylabel(
    "Probabilidade (apenas para visualizacao)"
)

ax.set_title(

    "Softmax antes do treino: "
    "classe verdadeira = " + CLASSES[rotulo_0]

)


ax.text(

    0.01,
    0.96,

    "CrossEntropyLoss recebeu LOGITS, sem Softmax",

    transform=ax.transAxes,

    va="top",

    fontsize=10

)


salvar(
    fig,
    "81_loss_inicial_sem_treino.png"
)


# ==========================================================
# 11 - GRAFICO DO FLUXO DE TREINAMENTO
# ==========================================================

# Somente visualizacao do que teremos
# na proxima aula.
#
# backward() e Adam.step() NAO
# serao executados neste script.

fig, ax = plt.subplots(
    figsize=(12, 3.9)
)

ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")


rotulos_fluxo = [

    (
        "Batch CIFAR-10",
        "PRONTO"
    ),

    (
        "LeNet-5 logits",
        "VERIFICADO"
    ),

    (
        "CrossEntropy",
        "VERIFICADA"
    ),

    (
        "backward()",
        "PROXIMA AULA"
    ),

    (
        "Adam.step()",
        "PROXIMA AULA"
    )

]


for i, (nome, status) in enumerate(
    rotulos_fluxo
):

    x0 = 0.15 + i * 2.42

    ax.text(

        x0 + 1.03,

        2.35,

        nome,

        ha="center",

        va="center",

        fontsize=10,

        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#e8edf5",
            "edgecolor": "#68788a"
        }

    )

    ax.text(

        x0 + 1.03,

        1.55,

        status,

        ha="center",

        fontsize=9

    )

    if i < 4:

        ax.annotate(

            "",

            xy=(x0 + 2.4, 2.35),

            xytext=(x0 + 2.08, 2.35),

            arrowprops={
                "arrowstyle": "->",
                "lw": 1.5
            }

        )


ax.set_title(

    "Fluxo do treinamento: "
    "preparado, mas ainda nao iniciado",

    fontsize=14

)


salvar(
    fig,
    "82_fluxo_treinamento_preparado.png"
)


# ==========================================================
# 12 - VALIDACAO FINAL
# ==========================================================

print("\n6 - CONFERENCIA FINAL")

print(
    "Device configurado: OK"
)

print(
    "Modelo e dados no mesmo device: OK"
)

print(
    "Batch size configurado: OK"
)

print(
    "CrossEntropyLoss configurada: OK"
)

print(
    "Adam / LR / weight_decay configurados: OK"
)

print(
    "100 epocas configuradas, 0 executadas."
)

print(
    "Nenhum backward() ou optimizer.step() foi executado."
)


print(
    "\nAULA 20 EXECUTADA COM SUCESSO!"
)
