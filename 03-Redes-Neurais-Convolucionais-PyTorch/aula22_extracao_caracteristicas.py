# =====================================================
# ALURA - AULA 22: EXTRACAO DE CARACTERISTICAS
#
# VGG16-BN pre-treinada no ImageNet
# CIFAR-10 real
# Extracao de 4096 caracteristicas
# Classificador SVM linear
#
# Os pesos da VGG16 NAO serao treinados.
# =====================================================

from pathlib import Path
from urllib.parse import urlsplit
import argparse
import time
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.models import VGG16_BN_Weights

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


# =====================================================
# 1 - CONFIGURACAO
# =====================================================

BASE = Path(__file__).resolve().parent

PASTA_DADOS = BASE / "dados"

RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

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

PESOS = VGG16_BN_Weights.IMAGENET1K_V1


def salvar(fig, arquivo):

    destino = RESULTADOS / arquivo

    fig.tight_layout()

    fig.savefig(
        destino,
        dpi=135
    )

    plt.close(fig)

    print(
        "Imagem salva:",
        destino,
        flush=True
    )


# =====================================================
# 2 - SELECIONAR AMOSTRAS EQUILIBRADAS
# =====================================================

def amostra_equilibrada(
    dataset,
    quantidade_por_classe,
    semente
):

    gerador = np.random.default_rng(
        semente
    )

    rotulos = np.asarray(
        dataset.targets
    )

    indices = []

    for classe in range(10):

        candidatos = np.flatnonzero(
            rotulos == classe
        )

        if len(candidatos) < quantidade_por_classe:

            raise ValueError(
                "Quantidade excessiva para "
                f"a classe {classe}."
            )

        escolhidos = gerador.choice(
            candidatos,
            size=quantidade_por_classe,
            replace=False
        )

        indices.extend(
            escolhidos.tolist()
        )

    gerador.shuffle(indices)

    return Subset(
        dataset,
        indices
    )


# =====================================================
# 3 - EXTRAIR CARACTERISTICAS
# =====================================================

def extrair(
    modelo,
    loader,
    dispositivo,
    nome_conjunto
):

    modelo.eval()

    blocos = []
    rotulos = []

    processadas = 0

    total = len(
        loader.dataset
    )

    intervalo = max(
        5,
        total // 10
    )

    ultimo_aviso = 0

    inicio = time.perf_counter()

    # Apenas forward.
    # Nenhum gradiente sera calculado.

    with torch.inference_mode():

        for imagens, y in loader:

            imagens = imagens.to(
                dispositivo
            )

            features = modelo(
                imagens
            )

            formato = tuple(
                features.shape
            )

            assert (
                features.ndim == 2
                and features.size(1) == 4096
            ), (
                "Dimensao incorreta: "
                + str(formato)
            )

            blocos.append(
                features.cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )

            rotulos.append(
                y.numpy()
            )

            processadas += len(y)

            if (
                processadas - ultimo_aviso >= intervalo
                or processadas == total
            ):

                tempo = (
                    time.perf_counter()
                    - inicio
                )

                print(
                    f"{nome_conjunto}: "
                    f"{processadas}/{total} imagens "
                    f"| {tempo:.1f}s",
                    flush=True
                )

                ultimo_aviso = processadas

    X = np.concatenate(
        blocos,
        axis=0
    )

    y = np.concatenate(
        rotulos,
        axis=0
    ).astype(
        np.int64
    )

    assert X.shape == (
        total,
        4096
    )

    assert y.shape == (
        total,
    )

    return X, y


# =====================================================
# 4 - ARGUMENTOS
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baixar-pesos",
        action="store_true",
        help="Baixar pesos oficiais da VGG16-BN"
    )

    parser.add_argument(
        "--treino-por-classe",
        type=int,
        default=2
    )

    parser.add_argument(
        "--teste-por-classe",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    args = parser.parse_args()

    if not 1 <= args.treino_por_classe <= 5000:

        parser.error(
            "Treino por classe deve estar "
            "entre 1 e 5000."
        )

    if not 1 <= args.teste_por_classe <= 1000:

        parser.error(
            "Teste por classe deve estar "
            "entre 1 e 1000."
        )

    if not 1 <= args.batch_size <= 8:

        parser.error(
            "Batch size deve estar "
            "entre 1 e 8."
        )

    print("=" * 65)

    print(
        "AULA 22 - EXTRACAO DE CARACTERISTICAS"
    )

    print("=" * 65)


    # =================================================
    # 5 - VERIFICAR OS PESOS PRE-TREINADOS
    # =================================================

    nome_arquivo = Path(
        urlsplit(PESOS.url).path
    ).name

    caminho_pesos = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / nome_arquivo
    )

    print(
        "\n1 - PESOS PRE-TREINADOS"
    )

    print(
        "Cache:",
        caminho_pesos
    )

    if (
        not caminho_pesos.exists()
        and not args.baixar_pesos
    ):

        print(
            "Pesos pre-treinados nao encontrados."
        )

        print(
            "Execute novamente com "
            "--baixar-pesos."
        )

        sys.exit(2)


    # =================================================
    # 6 - DISPOSITIVO
    # =================================================

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if dispositivo.type == "cpu":

        torch.set_num_threads(
            min(
                4,
                torch.get_num_threads()
            )
        )

    print(
        "\n2 - DISPOSITIVO"
    )

    print(
        "Hardware:",
        dispositivo
    )

    print(
        "Imagens de treino:",
        args.treino_por_classe * 10
    )

    print(
        "Imagens de teste:",
        args.teste_por_classe * 10
    )

    print(
        "Batch size:",
        args.batch_size
    )


    # =================================================
    # 7 - PRE-PROCESSAMENTO IMAGENET
    # =================================================

    # O torchvision fornece as
    # transformacoes oficiais associadas
    # aos pesos escolhidos.
    #
    # Resize 256
    # CenterCrop 224
    # ToTensor
    # Normalize:
    # media: 0.485, 0.456, 0.406
    # desvio: 0.229, 0.224, 0.225

    pre_processamento = (
        PESOS.transforms()
    )

    print(
        "\n3 - PRE-PROCESSAMENTO IMAGENET"
    )

    print(
        pre_processamento
    )


    # =================================================
    # 8 - CARREGAR O CIFAR-10 REAL
    # =================================================

    treino_total = datasets.CIFAR10(

        root=str(PASTA_DADOS),

        train=True,

        download=False,

        transform=pre_processamento

    )

    teste_total = datasets.CIFAR10(

        root=str(PASTA_DADOS),

        train=False,

        download=False,

        transform=pre_processamento

    )

    assert len(treino_total) == 50000

    assert len(teste_total) == 10000

    treino = amostra_equilibrada(

        treino_total,

        args.treino_por_classe,

        semente=22

    )

    teste = amostra_equilibrada(

        teste_total,

        args.teste_por_classe,

        semente=23

    )

    loader_treino = DataLoader(

        treino,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0

    )

    loader_teste = DataLoader(

        teste,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0

    )

    print(
        "\n4 - DATASET CIFAR-10 REAL"
    )

    print(
        "Dataset original de treino:",
        len(treino_total)
    )

    print(
        "Dataset original de teste:",
        len(teste_total)
    )

    print(
        "Amostra de treino:",
        len(treino)
    )

    print(
        "Amostra de teste:",
        len(teste)
    )


    # =================================================
    # 9 - VISUALIZAR NORMALIZACAO
    # =================================================

    original = datasets.CIFAR10(

        root=str(PASTA_DADOS),

        train=True,

        download=False,

        transform=None

    )

    indice = treino.indices[0]

    imagem_pil, rotulo = original[
        indice
    ]

    tensor = pre_processamento(
        imagem_pil
    )

    assert tuple(tensor.shape) == (
        3, 224, 224
    )

    media = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    desvio = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    restaurada = (
        tensor * desvio + media
    ).clamp(0, 1)

    fig, axs = plt.subplots(
        1, 3,
        figsize=(11, 4)
    )

    axs[0].imshow(
        imagem_pil
    )

    axs[0].set_title(
        "Original: 32x32"
    )

    axs[1].imshow(

        tensor.permute(
            1, 2, 0
        ).numpy().clip(0, 1)

    )

    axs[1].set_title(
        "Normalizado: visualizacao truncada"
    )

    axs[2].imshow(

        restaurada.permute(
            1, 2, 0
        ).numpy()

    )

    axs[2].set_title(
        "Desnormalizado: 224x224"
    )

    for ax in axs:

        ax.axis("off")

    salvar(
        fig,
        "86_vgg16_preprocessamento_imagenet.png"
    )


    # =================================================
    # 10 - CARREGAR A VGG16-BN PRE-TREINADA
    # =================================================

    print(
        "\n5 - REDE VGG16-BN PRE-TREINADA"
    )

    if not caminho_pesos.exists():

        print(
            "Baixando pesos oficiais "
            "do ImageNet...",
            flush=True
        )

    try:

        net = models.vgg16_bn(
            weights=PESOS
        )

    except Exception as erro:

        raise SystemExit(
            "Nao foi possivel carregar "
            "os pesos pre-treinados: "
            + str(erro)
        ) from erro

    print(
        "Modelo VGG16-BN carregado."
    )


    # =================================================
    # 11 - INSPECIONAR CHILDREN()
    # =================================================

    print(
        "\n6 - CAMADAS PRINCIPAIS"
    )

    filhos = list(
        net.named_children()
    )

    for nome, modulo in filhos:

        print(
            nome,
            type(modulo).__name__
        )

    assert [
        nome for nome, _ in filhos
    ] == [
        "features",
        "avgpool",
        "classifier"
    ]

    print(
        "Total de filhos:",
        len(filhos)
    )

    print(
        "Ultima camada original:",
        net.classifier[-1]
    )

    assert (
        net.classifier[-1].out_features
        == 1000
    )


    # =================================================
    # 12 - ADAPTAR O CLASSIFIER
    # =================================================

    # Modelo original:
    #
    # Linear 25088 -> 4096
    # ReLU
    # Dropout
    # Linear 4096 -> 4096
    # ReLU
    # Dropout
    # Linear 4096 -> 1000
    #
    # Remover as ultimas tres camadas:
    #
    # ReLU
    # Dropout
    # Linear 4096 -> 1000
    #
    # Manter saida com 4096 atributos.

    camadas = list(
        net.classifier.children()
    )

    novo_classifier = nn.Sequential(
        *camadas[:-3]
    )

    net.classifier = novo_classifier

    print(
        "\n7 - CLASSIFIER ADAPTADO"
    )

    print(
        net.classifier
    )

    assert isinstance(
        net.classifier[-1],
        nn.Linear
    )

    assert (
        net.classifier[-1].out_features
        == 4096
    )


    # =================================================
    # 13 - CONGELAR PESOS DA VGG16
    # =================================================

    net.requires_grad_(False)

    net = net.to(
        dispositivo
    )

    net.eval()

    assert not any(
        parametro.requires_grad
        for parametro in net.parameters()
    )

    print(
        "Pesos da VGG16 congelados: OK"
    )


    # =================================================
    # 14 - GRAFICO DO EXTRATOR
    # =================================================

    etapas = [

        (
            "CIFAR-10",
            "Imagem RGB 32x32"
        ),

        (
            "ImageNet",
            "Transformacoes para 224x224"
        ),

        (
            "VGG16-BN",
            "Rede pre-treinada"
        ),

        (
            "Extrator",
            "4096 atributos por imagem"
        ),

        (
            "SVM",
            "Classificacao de 10 categorias"
        )

    ]

    fig, ax = plt.subplots(
        figsize=(11, 4.2)
    )

    ax.axis("off")

    for i, (nome, detalhe) in enumerate(
        etapas
    ):

        y = 0.89 - i * 0.19

        ax.text(
            0.03,
            y,
            nome,
            weight="bold",
            fontsize=12,
            transform=ax.transAxes
        )

        ax.text(
            0.40,
            y,
            detalhe,
            fontsize=11,
            transform=ax.transAxes
        )

        if i < len(etapas) - 1:

            ax.annotate(
                "",
                xy=(0.34, y - 0.11),
                xytext=(0.34, y - 0.035),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 1.2
                },
                xycoords=ax.transAxes
            )

    ax.set_title(
        "VGG16-BN como extrator de caracteristicas"
    )

    salvar(
        fig,
        "87_vgg16_extrator_4096.png"
    )


    # =================================================
    # 15 - EXTRAIR AS FEATURES
    # =================================================

    print(
        "\n8 - EXTRACAO DE CARACTERISTICAS"
    )

    X_treino, y_treino = extrair(

        net,

        loader_treino,

        dispositivo,

        "Treino"

    )

    X_teste, y_teste = extrair(

        net,

        loader_teste,

        dispositivo,

        "Teste"

    )

    print(
        "X_treino:",
        X_treino.shape
    )

    print(
        "y_treino:",
        y_treino.shape
    )

    print(
        "X_teste:",
        X_teste.shape
    )

    print(
        "y_teste:",
        y_teste.shape
    )

    print(
        "VALIDACAO DAS 4096 FEATURES: OK"
    )


    # =================================================
    # 16 - SALVAR CARACTERISTICAS
    # =================================================

    arquivo_features = (
        RESULTADOS
        / "aula22_features_vgg16_amostra.npz"
    )

    np.savez_compressed(

        arquivo_features,

        X_treino=X_treino,

        y_treino=y_treino,

        X_teste=X_teste,

        y_teste=y_teste

    )

    print(
        "Features salvas:",
        arquivo_features
    )


    # =================================================
    # 17 - CLASSIFICADOR SVM
    # =================================================

    # A CNN somente extrai as features.
    #
    # O SVM recebe os arrays NumPy:
    #
    # X = (amostras, 4096)
    # y = (amostras,)
    #
    # Apenas o SVM sera treinado.

    print(
        "\n9 - TREINAMENTO DO SVM LINEAR"
    )

    classificador = SVC(

        kernel="linear",

        C=1.0

    )

    classificador.fit(
        X_treino,
        y_treino
    )

    pred_treino = classificador.predict(
        X_treino
    )

    pred_teste = classificador.predict(
        X_teste
    )

    acc_treino = accuracy_score(
        y_treino,
        pred_treino
    ) * 100

    acc_teste = accuracy_score(
        y_teste,
        pred_teste
    ) * 100

    print(
        "Acuracia treino SVM:",
        f"{acc_treino:.2f}%"
    )

    print(
        "Acuracia teste SVM:",
        f"{acc_teste:.2f}%"
    )

    print(
        "AVISO: estas acuracias foram calculadas "
        "sobre amostras reduzidas."
    )


    # =================================================
    # 18 - GRAFICO DE ACURACIA
    # =================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    valores = [
        acc_treino,
        acc_teste
    ]

    barras = ax.bar(
        ["Treino", "Teste"],
        valores
    )

    ax.set_ylim(
        0, 105
    )

    ax.set_ylabel(
        "Acuracia (%)"
    )

    ax.set_title(
        "SVM linear + caracteristicas VGG16"
    )

    for barra, valor in zip(
        barras,
        valores
    ):

        ax.text(

            barra.get_x()
            + barra.get_width() / 2,

            valor + 2,

            f"{valor:.1f}%",

            ha="center"

        )

    salvar(
        fig,
        "88_vgg16_svm_acuracia.png"
    )


    # =================================================
    # 19 - MATRIZ DE CONFUSAO
    # =================================================

    matriz = confusion_matrix(

        y_teste,

        pred_teste,

        labels=list(range(10))

    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    imagem = ax.imshow(
        matriz,
        cmap="Blues"
    )

    ax.set_xticks(
        range(10),
        labels=CLASSES,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(10),
        labels=CLASSES
    )

    ax.set_xlabel(
        "Classe prevista"
    )

    ax.set_ylabel(
        "Classe real"
    )

    ax.set_title(
        "Matriz de confusao - CIFAR-10"
    )

    fig.colorbar(
        imagem,
        ax=ax
    )

    for i in range(10):

        for j in range(10):

            valor = matriz[i, j]

            ax.text(

                j,
                i,
                str(valor),

                ha="center",
                va="center",
                fontsize=8,

                color=(
                    "white"
                    if valor > matriz.max() / 2
                    else "black"
                )

            )

    salvar(
        fig,
        "89_vgg16_svm_matriz_confusao.png"
    )


    # =================================================
    # 20 - VALIDACAO FINAL
    # =================================================

    print(
        "\n10 - CONCEITOS APRENDIDOS"
    )

    print(
        "1 - VGG16-BN pre-treinada no ImageNet."
    )

    print(
        "2 - Pre-processamento oficial do ImageNet."
    )

    print(
        "3 - Acesso as camadas com named_children()."
    )

    print(
        "4 - Remocao das camadas finais."
    )

    print(
        "5 - Extracao de 4096 atributos."
    )

    print(
        "6 - Conversao de tensores para NumPy."
    )

    print(
        "7 - Treinamento de SVM linear com fit()."
    )

    print(
        "8 - Classificacao com predict()."
    )

    print(
        "9 - Acuracia e matriz de confusao."
    )

    print(
        "10 - Nenhum peso da VGG16 foi atualizado."
    )

    print(
        "\nAULA 22 EXECUTADA COM SUCESSO!"
    )


if __name__ == "__main__":

    main()
