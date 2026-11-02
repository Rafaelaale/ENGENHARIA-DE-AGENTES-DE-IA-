# ==========================================================
# ALURA - AULA 19: CNN DO ZERO EM PYTORCH
# LeNet-5 adaptada para RGB / CIFAR-10
#
# Nesta aula:
# - Construir a arquitetura
# - Preparar os dados
# - Criar DataLoader
# - Validar as dimensoes
# - Fazer um forward sem treinamento
# ==========================================================

from collections import OrderedDict
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

torch.manual_seed(19)

CLASSES = (
    "aviao",
    "automovel",
    "passaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhao"
)


def salvar(fig, nome):

    destino = RESULTADOS / nome

    fig.tight_layout()

    fig.savefig(
        destino,
        dpi=135
    )

    plt.close(fig)

    print("Imagem salva:", destino)


# ==========================================================
# 2 - CARREGAR O CIFAR-10
# ==========================================================

def carregar_dados(baixar=False, raiz=None):

    conversao = transforms.ToTensor()

    if raiz:

        pastas = [
            Path(raiz).expanduser()
        ]

    else:

        pastas = [
            BASE / "dados",
            BASE / "data",
            BASE
        ]

    # Primeiro, procurar dados reais
    # ja armazenados no computador.

    for pasta in pastas:

        try:

            treino = datasets.CIFAR10(
                root=str(pasta),
                train=True,
                download=False,
                transform=conversao
            )

            teste = datasets.CIFAR10(
                root=str(pasta),
                train=False,
                download=False,
                transform=conversao
            )

            print(
                "CIFAR-10 REAL ENCONTRADO EM:",
                pasta
            )

            return treino, teste, True

        except (
            RuntimeError,
            FileNotFoundError,
            OSError
        ):

            pass

    # Download somente quando solicitado
    # com o argumento --baixar.

    if baixar:

        pasta = pastas[0]

        print(
            "Tentando baixar CIFAR-10 em:",
            pasta
        )

        try:

            treino = datasets.CIFAR10(
                root=str(pasta),
                train=True,
                download=True,
                transform=conversao
            )

            teste = datasets.CIFAR10(
                root=str(pasta),
                train=False,
                download=True,
                transform=conversao
            )

            print(
                "CIFAR-10 REAL BAIXADO COM SUCESSO."
            )

            return treino, teste, True

        except Exception as erro:

            print(
                "Falha no download:",
                type(erro).__name__,
                str(erro)[:250]
            )

    # Fallback offline:
    # imagens e rotulos artificiais.
    #
    # Serve para verificar o funcionamento
    # da rede, NAO para treinar um
    # classificador de objetos real.

    treino = datasets.FakeData(
        size=256,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=conversao,
        random_offset=19
    )

    teste = datasets.FakeData(
        size=64,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=conversao,
        random_offset=1019
    )

    print(
        "ATENCAO: usando FakeData."
    )

    print(
        "As imagens e os rotulos sao "
        "SINTETICOS, nao sao CIFAR-10."
    )

    return treino, teste, False


# ==========================================================
# 3 - CONSTRUIR A LENET-5 RGB
# ==========================================================

def construir_lenet_rgb():

    # Arquitetura didatica da transcricao:
    #
    # Entrada (b, 3, 32, 32)
    #
    # Bloco 1:
    # Conv 3 -> 6, kernel 5
    # BatchNorm(6)
    # Tanh
    # AvgPool(2)
    # Saida: (b, 6, 14, 14)
    #
    # Bloco 2:
    # Conv 6 -> 16, kernel 5
    # BatchNorm(16)
    # Tanh
    # AvgPool(2)
    # Saida: (b, 16, 5, 5)
    #
    # Bloco 3:
    # Conv 16 -> 120, kernel 5
    # BatchNorm(120)
    # Tanh
    # Saida: (b, 120, 1, 1)
    #
    # Flatten: (b, 120)
    # Linear: 120 -> 84
    # Tanh
    # Linear: 84 -> 10

    modelo = nn.Sequential(

        OrderedDict([

            (
                "bloco1",

                nn.Sequential(

                    nn.Conv2d(
                        in_channels=3,
                        out_channels=6,
                        kernel_size=5,
                        stride=1,
                        padding=0
                    ),

                    nn.BatchNorm2d(6),

                    nn.Tanh(),

                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
            ),

            (
                "bloco2",

                nn.Sequential(

                    nn.Conv2d(
                        in_channels=6,
                        out_channels=16,
                        kernel_size=5,
                        stride=1,
                        padding=0
                    ),

                    nn.BatchNorm2d(16),

                    nn.Tanh(),

                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
            ),

            (
                "bloco3",

                nn.Sequential(

                    nn.Conv2d(
                        in_channels=16,
                        out_channels=120,
                        kernel_size=5,
                        stride=1,
                        padding=0
                    ),

                    nn.BatchNorm2d(120),

                    nn.Tanh()
                )
            ),

            (
                "flatten",
                nn.Flatten()
            ),

            (
                "fc1",
                nn.Linear(120, 84)
            ),

            (
                "tanh_fc",
                nn.Tanh()
            ),

            (
                "fc2",
                nn.Linear(84, 10)
            )

        ])
    )

    return modelo


# ==========================================================
# 4 - INICIAR A PRATICA
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baixar",
        action="store_true",
        help="Baixar o CIFAR-10 real"
    )

    parser.add_argument(
        "--dados",
        type=str,
        default=None,
        help="Pasta raiz do CIFAR-10"
    )

    args = parser.parse_args()

    print("=" * 65)
    print(
        "AULA 19 - CNN DO ZERO EM PYTORCH"
    )
    print("=" * 65)


    # ==================================================
    # 5 - INSTANCIAR A REDE
    # ==================================================

    print(
        "\n1 - ARQUITETURA LENET-5 ADAPTADA"
    )

    modelo = construir_lenet_rgb()

    print(modelo)

    parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    print(
        "Parametros totais:",
        parametros
    )

    print(
        "Os pesos foram apenas inicializados."
    )


    # ==================================================
    # 6 - CARREGAR DADOS
    # ==================================================

    print(
        "\n2 - DATASET CIFAR-10"
    )

    treino, teste, dados_reais = carregar_dados(
        baixar=args.baixar,
        raiz=args.dados
    )

    print(
        "Origem:",
        "CIFAR-10 REAL"
        if dados_reais
        else "FakeData SINTETICO"
    )

    print(
        "Quantidade de treino:",
        len(treino)
    )

    print(
        "Quantidade de teste:",
        len(teste)
    )

    print(
        "Classes:",
        CLASSES
    )

    imagem0, rotulo0 = treino[0]

    print(
        "Primeira imagem (C,H,W):",
        tuple(imagem0.shape)
    )

    print(
        "Primeiro rotulo:",
        rotulo0,
        CLASSES[int(rotulo0)]
    )

    assert tuple(imagem0.shape) == (
        3, 32, 32
    )


    # ==================================================
    # 7 - VISUALIZAR 10 IMAGENS
    # ==================================================

    fig, axs = plt.subplots(
        2, 5,
        figsize=(12, 5.7)
    )

    for i, ax in enumerate(axs.flat):

        imagem, rotulo = treino[i]

        imagem_rgb = (
            imagem
            .permute(1, 2, 0)
            .numpy()
        )

        ax.imshow(imagem_rgb)

        ax.set_title(
            f"{i}: {CLASSES[int(rotulo)]}",
            fontsize=9
        )

        ax.axis("off")

    if dados_reais:

        fig.suptitle(
            "Imagens reais do CIFAR-10"
        )

    else:

        fig.suptitle(
            "FakeData: imagens sinteticas "
            "(NAO sao CIFAR-10)"
        )

    salvar(
        fig,
        "76_lenet5_imagens_dataset.png"
    )


    # ==================================================
    # 8 - CRIAR DATALOADER
    # ==================================================

    # O DataLoader organiza as imagens
    # em pequenos lotes.
    #
    # Batch_size = 32
    #
    # num_workers = 0 facilita a execucao
    # no Windows sem criar subprocessos.

    carregador_treino = DataLoader(
        treino,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    carregador_teste = DataLoader(
        teste,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    imagens, rotulos = next(
        iter(carregador_treino)
    )

    imagens_teste, rotulos_teste = next(
        iter(carregador_teste)
    )

    print(
        "\n3 - DATALOADER"
    )

    print(
        "Lote de treinamento:",
        tuple(imagens.shape)
    )

    print(
        "Rotulos:",
        tuple(rotulos.shape)
    )

    print(
        "Lote de teste:",
        tuple(imagens_teste.shape)
    )

    assert tuple(imagens.shape) == (
        32, 3, 32, 32
    )

    assert tuple(rotulos.shape) == (
        32,
    )

    print(
        "VALIDACAO DO DATALOADER: OK"
    )


    # ==================================================
    # 9 - FORWARD SEM TREINAMENTO
    # ==================================================

    # Em eval(), BatchNorm usa as
    # estatisticas acumuladas.
    #
    # Como a rede ainda nao foi treinada,
    # essas estatisticas estao inicialmente
    # em media 0 e variancia 1.
    #
    # Nao usaremos otimizador nem
    # calculo de gradientes.

    modelo.eval()

    formatos = [
        (
            "Entrada",
            tuple(imagens.shape)
        )
    ]

    passos = {}

    with torch.no_grad():

        x = imagens

        for nome, camada in modelo.named_children():

            x = camada(x)

            passos[nome] = x

            formatos.append(
                (
                    nome,
                    tuple(x.shape)
                )
            )

        logits = x

        probabilidades = torch.softmax(
            logits,
            dim=1
        )


    # ==================================================
    # 10 - VALIDAR DIMENSOES
    # ==================================================

    esperado = {

        "bloco1": (32, 6, 14, 14),

        "bloco2": (32, 16, 5, 5),

        "bloco3": (32, 120, 1, 1),

        "flatten": (32, 120),

        "fc1": (32, 84),

        "tanh_fc": (32, 84),

        "fc2": (32, 10)

    }

    print(
        "\n4 - DIMENSOES DE CADA CAMADA"
    )

    for nome, formato in formatos:

        print(
            f"{nome:12s}: {formato}"
        )

        if nome in esperado:

            assert formato == esperado[nome], (
                f"Dimensao incorreta: {nome}"
            )

    assert tuple(probabilidades.shape) == (
        32, 10
    )

    torch.testing.assert_close(
        probabilidades.sum(dim=1),
        torch.ones(32),
        atol=1e-6,
        rtol=0
    )

    print(
        "VALIDACAO DE TODAS AS DIMENSOES: OK"
    )

    print(
        "VALIDACAO DAS PROBABILIDADES: OK"
    )


    # ==================================================
    # 11 - VISUALIZAR ARQUITETURA
    # ==================================================

    nomes = {

        "Entrada": "Imagem RGB",

        "bloco1": (
            "Conv(3->6,5) + BN + Tanh + AvgPool(2)"
        ),

        "bloco2": (
            "Conv(6->16,5) + BN + Tanh + AvgPool(2)"
        ),

        "bloco3": (
            "Conv(16->120,5) + BN + Tanh"
        ),

        "flatten": "Flatten",

        "fc1": "Linear(120,84)",

        "tanh_fc": "Tanh",

        "fc2": "Linear(84,10) = logits"

    }

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    ax.axis("off")

    for i, (nome, formato) in enumerate(formatos):

        posicao = 0.92 - i * 0.105

        ax.text(
            0.05,
            posicao,
            nomes[nome],
            transform=ax.transAxes,
            fontsize=10,
            va="center"
        )

        ax.text(
            0.99,
            posicao,
            str(formato),
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="center",
            family="monospace"
        )

    ax.set_title(
        "LeNet-5 RGB: arquitetura e dimensoes"
    )

    salvar(
        fig,
        "77_lenet5_arquitetura_dimensoes.png"
    )


    # ==================================================
    # 12 - MAPAS DE CARACTERISTICAS
    # ==================================================

    # Os filtros possuem pesos aleatorios.
    # Portanto estes mapas ainda nao
    # representam caracteristicas aprendidas.

    mapas1 = passos["bloco1"][0]
    mapas2 = passos["bloco2"][0]

    fig, axs = plt.subplots(
        2, 6,
        figsize=(14, 5.8)
    )

    for i in range(6):

        axs[0, i].imshow(
            mapas1[i].numpy(),
            cmap="viridis"
        )

        axs[0, i].set_title(
            f"Bloco 1: canal {i + 1}",
            fontsize=9
        )

        axs[1, i].imshow(
            mapas2[i].numpy(),
            cmap="viridis"
        )

        axs[1, i].set_title(
            f"Bloco 2: canal {i + 1}",
            fontsize=9
        )

        axs[0, i].axis("off")
        axs[1, i].axis("off")

    fig.suptitle(
        "Mapas de ativacao - filtros NAO treinados"
    )

    salvar(
        fig,
        "78_lenet5_mapas_sem_treino.png"
    )


    # ==================================================
    # 13 - LOGITS E PROBABILIDADES
    # ==================================================

    # A camada nn.Linear(84,10)
    # produz 10 logits.
    #
    # Logits NAO sao probabilidades.
    #
    # Aplicamos Softmax separadamente
    # para obter probabilidades que
    # somam 1.
    #
    # Como nao houve treinamento,
    # esses valores nao sao confiaveis.

    exemplo_logits = (
        logits[0].numpy()
    )

    exemplo_probas = (
        probabilidades[0].numpy()
    )

    fig, axs = plt.subplots(
        1, 2,
        figsize=(14, 4.7)
    )

    axs[0].bar(
        range(10),
        exemplo_logits
    )

    axs[0].set_xticks(
        range(10),
        CLASSES,
        rotation=45,
        ha="right"
    )

    axs[0].set_title(
        "Saida direta da rede: logits"
    )

    axs[0].set_ylabel(
        "Valores sem normalizacao"
    )

    axs[1].bar(
        range(10),
        exemplo_probas
    )

    axs[1].set_xticks(
        range(10),
        CLASSES,
        rotation=45,
        ha="right"
    )

    axs[1].set_title(
        "Softmax: probabilidades"
    )

    axs[1].set_ylabel(
        "Probabilidade"
    )

    axs[1].set_ylim(
        0,
        max(
            0.2,
            float(exemplo_probas.max()) * 1.2
        )
    )

    fig.suptitle(
        "Rede sem treinamento: "
        "previsoes ainda nao confiaveis"
    )

    salvar(
        fig,
        "79_lenet5_logits_e_softmax.png"
    )


    # ==================================================
    # 14 - VALIDACAO DA ARQUITETURA
    # ==================================================

    assert (
        modelo[2][0].out_channels
        == 120
    )

    assert (
        modelo[4].in_features
        == 120
    )

    assert (
        modelo[4].out_features
        == 84
    )

    assert (
        modelo[6].out_features
        == 10
    )


    print(
        "\n5 - SAIDA DO MODELO"
    )

    print(
        "Logits do primeiro exemplo:",
        np.round(
            exemplo_logits,
            4
        )
    )

    print(
        "Softmax do primeiro exemplo:",
        np.round(
            exemplo_probas,
            4
        )
    )

    print(
        "Soma das probabilidades:",
        round(
            float(exemplo_probas.sum()),
            6
        )
    )

    print(
        "\nOBSERVACAO:"
    )

    print(
        "A CNN foi construida, mas "
        "seus pesos nao foram treinados."
    )

    print(
        "A classificacao ainda nao possui "
        "valor preditivo confiavel."
    )

    print(
        "\nAULA 19 EXECUTADA COM SUCESSO!"
    )

    print(
        "TREINAMENTO RESERVADO "
        "PARA A PROXIMA AULA DA ALURA."
    )


if __name__ == "__main__":

    main()
