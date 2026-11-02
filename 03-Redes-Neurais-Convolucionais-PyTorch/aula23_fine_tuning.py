# ALURA - AULA 23: FINE-TUNING VGG16-BN / CIFAR-10
# Reutiliza os pesos do ImageNet baixados na Aula 22.
# Inicial: fine-tuning parcial, com os pesos iniciais congelados.
# Opcional --integral: libera tambem as camadas anteriores (alto custo).

from pathlib import Path
import argparse
import csv
import time
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.models import VGG16_BN_Weights


BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
MODELOS = BASE / "modelos"

RESULTADOS.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(parents=True, exist_ok=True)

CLASSES = (
    "aviao", "automovel", "passaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhao"
)


# =====================================================
# 1 - SALVAR GRAFICOS
# =====================================================

def salvar(fig, nome):

    caminho = RESULTADOS / nome

    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)

    print(
        "Imagem salva:",
        caminho,
        flush=True
    )


# =====================================================
# 2 - SELECIONAR AMOSTRAS EQUILIBRADAS
# =====================================================

def selecionar_por_classe(
    dataset,
    indices_permitidos,
    por_classe,
    semente
):

    rng = np.random.default_rng(semente)

    rotulos = np.asarray(dataset.targets)

    selecionados = []

    for classe in range(10):

        candidatos = np.asarray(
            indices_permitidos,
            dtype=np.int64
        )

        candidatos = candidatos[
            rotulos[candidatos] == classe
        ]

        if len(candidatos) < por_classe:

            raise RuntimeError(
                f"Nao ha {por_classe} "
                f"amostras na classe {classe}."
            )

        escolhidos = rng.choice(
            candidatos,
            por_classe,
            replace=False
        )

        selecionados.extend(
            escolhidos.tolist()
        )

    rng.shuffle(selecionados)

    return Subset(
        dataset,
        selecionados
    )


# =====================================================
# 3 - TREINAMENTO E VALIDACAO
# =====================================================

def ciclo(
    modelo,
    loader,
    criterio,
    device,
    optimizer,
    limite,
    nome
):

    treinando = optimizer is not None

    modelo.train(treinando)

    # Preservar estatisticas acumuladas
    # pelas BatchNorm durante o ImageNet.
    #
    # Parametros afins descongelados
    # continuam podendo receber gradientes.

    for modulo in modelo.modules():

        if isinstance(
            modulo,
            nn.BatchNorm2d
        ):

            modulo.eval()

    perda_acumulada = 0.0
    acertos = 0
    exemplos = 0
    passos = 0

    inicio = time.perf_counter()

    for x, y in loader:

        if (
            limite is not None
            and passos >= limite
        ):
            break

        x = x.to(device)
        y = y.to(device)

        if treinando:

            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward
            logits = modelo(x)

            # Calcular CrossEntropyLoss
            loss = criterio(
                logits,
                y
            )

            # Backpropagation
            loss.backward()

            # Atualizacao dos pesos
            optimizer.step()

        else:

            # Validacao sem gradientes.

            with torch.inference_mode():

                logits = modelo(x)

                loss = criterio(
                    logits,
                    y
                )

        quantidade = int(
            y.numel()
        )

        perda_acumulada += (
            float(loss.detach()) * quantidade
        )

        predicoes = logits.argmax(
            dim=1
        )

        acertos += int(
            (predicoes == y).sum()
        )

        exemplos += quantidade
        passos += 1

        if (
            passos == 1
            or passos % 10 == 0
            or passos == len(loader)
            or passos == limite
        ):

            tempo = (
                time.perf_counter()
                - inicio
            )

            print(
                f"  {nome}: lote {passos}"
                f" | imagens {exemplos}"
                f" | {tempo:.1f}s",
                flush=True
            )

    if not exemplos:

        raise RuntimeError(
            "Nenhuma amostra foi processada."
        )

    perda_media = (
        perda_acumulada / exemplos
    )

    acuracia = (
        100 * acertos / exemplos
    )

    return (
        perda_media,
        acuracia,
        exemplos
    )


# =====================================================
# 4 - SALVAR HISTORICO CSV
# =====================================================

def salvar_csv(historico):

    destino = (
        RESULTADOS
        / "aula23_historico_fine_tuning.csv"
    )

    with destino.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        campos = (
            "epoca",
            "loss_treino",
            "loss_validacao",
            "acuracia_treino",
            "acuracia_validacao"
        )

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(historico)

    print(
        "Historico:",
        destino,
        flush=True
    )


# =====================================================
# 5 - GRAFICOS
# =====================================================

def graficos(historico):

    epocas = [
        r["epoca"]
        for r in historico
    ]

    # LOSS

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.plot(
        epocas,
        [
            r["loss_treino"]
            for r in historico
        ],
        "o-",
        label="Treino"
    )

    ax.plot(
        epocas,
        [
            r["loss_validacao"]
            for r in historico
        ],
        "o-",
        label="Validacao"
    )

    ax.set(
        xlabel="Epoca",
        ylabel="CrossEntropyLoss",
        title="VGG16-BN: fine-tuning - perda"
    )

    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(alpha=0.25)

    salvar(
        fig,
        "91_vgg16_finetuning_loss.png"
    )

    # ACURACIA

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.plot(
        epocas,
        [
            r["acuracia_treino"]
            for r in historico
        ],
        "o-",
        label="Treino"
    )

    ax.plot(
        epocas,
        [
            r["acuracia_validacao"]
            for r in historico
        ],
        "o-",
        label="Validacao"
    )

    ax.set(
        xlabel="Epoca",
        ylabel="Acuracia (%)",
        title="VGG16-BN: fine-tuning - acuracia"
    )

    ax.set_ylim(0, 100)
    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(alpha=0.25)

    salvar(
        fig,
        "92_vgg16_finetuning_acuracia.png"
    )


# =====================================================
# 6 - FUNCAO PRINCIPAL
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aula 23 - Fine-tuning VGG16-BN"
        )
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lotes-treino",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lotes-validacao",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lotes-teste",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005
    )

    parser.add_argument(
        "--integral",
        action="store_true",
        help="Descongelar toda a VGG16"
    )

    parser.add_argument(
        "--todos-os-dados",
        action="store_true",
        help="Usar todos os lotes"
    )

    args = parser.parse_args()

    if (
        args.epocas < 1
        or args.batch_size < 1
        or args.batch_size > 4
    ):

        parser.error(
            "Epocas >= 1; "
            "batch-size entre 1 e 4."
        )

    if (
        args.lr <= 0
        or args.weight_decay < 0
    ):

        parser.error(
            "Learning rate deve ser positivo."
        )

    for limite in (
        args.lotes_treino,
        args.lotes_validacao,
        args.lotes_teste
    ):

        if limite < 1:

            parser.error(
                "Limites de lotes devem ser >= 1."
            )

    torch.manual_seed(23)
    np.random.seed(23)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cpu":

        torch.set_num_threads(
            min(
                4,
                torch.get_num_threads()
            )
        )

    print("=" * 65)
    print(
        "AULA 23 - FINE-TUNING DA VGG16-BN"
    )
    print("=" * 65)

    print(
        "Device:",
        device
    )

    print(
        "Modo:",
        "INTEGRAL (alto consumo de RAM)"
        if args.integral
        else "PARCIAL: ultimo bloco + camada final"
    )

    print(
        "Epocas:",
        args.epocas
    )

    print(
        "LR base:",
        args.lr * 0.2
    )

    print(
        "LR classifier:",
        args.lr
    )

    print(
        "Weight decay base:",
        args.weight_decay * 0.2
    )

    print(
        "Weight decay classifier:",
        args.weight_decay
    )


    # =================================================
    # 7 - PRE-PROCESSAMENTO IMAGENET
    # =================================================

    pesos = (
        VGG16_BN_Weights.IMAGENET1K_V1
    )

    transformacao = pesos.transforms()

    print(
        "\n1 - PRE-PROCESSAMENTO IMAGENET"
    )

    print(
        transformacao
    )


    # =================================================
    # 8 - CARREGAR CIFAR-10 REAL
    # =================================================

    dataset_treino = datasets.CIFAR10(

        root=str(BASE / "dados"),

        train=True,

        download=False,

        transform=transformacao

    )

    dataset_teste = datasets.CIFAR10(

        root=str(BASE / "dados"),

        train=False,

        download=False,

        transform=transformacao

    )

    assert len(dataset_treino) == 50000
    assert len(dataset_teste) == 10000

    # Separar 45 mil imagens para treino
    # e 5 mil para validacao.

    permutacao = np.random.default_rng(
        23
    ).permutation(50000)

    ids_treino = permutacao[:45000]
    ids_val = permutacao[45000:]

    if args.todos_os_dados:

        treino = Subset(
            dataset_treino,
            ids_treino.tolist()
        )

        validacao = Subset(
            dataset_treino,
            ids_val.tolist()
        )

        teste = dataset_teste

        limite_t = None
        limite_v = None
        limite_teste = None

    else:

        # Amostras pequenas, equilibradas
        # e sem sobreposicao entre os conjuntos.

        treino = selecionar_por_classe(
            dataset_treino,
            ids_treino,
            10,
            231
        )

        validacao = selecionar_por_classe(
            dataset_treino,
            ids_val,
            3,
            232
        )

        teste = selecionar_por_classe(
            dataset_teste,
            np.arange(10000),
            3,
            233
        )

        limite_t = args.lotes_treino
        limite_v = args.lotes_validacao
        limite_teste = args.lotes_teste

    def criar_loader(dataset, shuffle):

        return DataLoader(

            dataset,

            batch_size=args.batch_size,

            shuffle=shuffle,

            num_workers=0

        )

    loader_t = criar_loader(
        treino,
        True
    )

    loader_v = criar_loader(
        validacao,
        False
    )

    loader_teste = criar_loader(
        teste,
        False
    )

    print(
        "\n2 - CIFAR-10 REAL"
    )

    print(
        "Treino:",
        len(treino)
    )

    print(
        "Validacao:",
        len(validacao)
    )

    print(
        "Teste:",
        len(teste)
    )

    print(
        "Lotes disponiveis:",
        len(loader_t),
        len(loader_v),
        len(loader_teste)
    )


    # =================================================
    # 9 - CARREGAR VGG16-BN PRE-TREINADA
    # =================================================

    print(
        "\n3 - VGG16-BN PRE-TREINADA"
    )

    print(
        "Reutilizando pesos ImageNet "
        "da Aula 22...",
        flush=True
    )

    try:

        net = models.vgg16_bn(
            weights=pesos
        )

    except Exception as erro:

        raise SystemExit(
            "Nao foi possivel carregar "
            "os pesos ImageNet: "
            + str(erro)
        ) from erro

    filhos = [
        (
            nome,
            type(modulo).__name__
        )
        for nome, modulo
        in net.named_children()
    ]

    print(
        "Filhos:",
        filhos
    )


    # =================================================
    # 10 - ADAPTAR O CLASSIFIER
    # =================================================

    # Original:
    # Linear(4096, 1000)
    #
    # Novo:
    # Linear(4096, 10)

    ultima = list(
        net.classifier.children()
    )[-1]

    entrada = ultima.in_features
    saida_antiga = ultima.out_features

    assert entrada == 4096
    assert saida_antiga == 1000

    camadas_classifier = list(
        net.classifier.children()
    )[:-1]

    camadas_classifier.append(
        nn.Linear(
            entrada,
            10
        )
    )

    net.classifier = nn.Sequential(
        *camadas_classifier
    )

    assert (
        net.classifier[-1].out_features
        == 10
    )

    print(
        "\n4 - CLASSIFIER ADAPTADO"
    )

    print(
        f"Linear({entrada}, {saida_antiga})"
        f" -> Linear({entrada}, 10)"
    )


    # =================================================
    # 11 - CONTROLAR PESOS TREINAVEIS
    # =================================================

    if not args.integral:

        # Congelar toda a rede.

        net.requires_grad_(False)

        # Liberar o ultimo bloco de
        # convolucoes da VGG16-BN.

        for parametro in (
            net.features[34:].parameters()
        ):

            parametro.requires_grad_(True)

        # Liberar apenas a nova
        # camada Linear(4096,10).

        for parametro in (
            net.classifier[-1].parameters()
        ):

            parametro.requires_grad_(True)

    # Se --integral for utilizado,
    # todas as camadas ficam treinaveis.

    parametros_features = [

        p for p in net.features.parameters()

        if p.requires_grad

    ]

    parametros_classifier = [

        p for p in net.classifier.parameters()

        if p.requires_grad

    ]

    assert parametros_features
    assert parametros_classifier

    net = net.to(device)

    print(
        "\n5 - PARAMETROS TREINAVEIS"
    )

    print(
        "Features:",
        sum(
            p.numel()
            for p in parametros_features
        )
    )

    print(
        "Classifier:",
        sum(
            p.numel()
            for p in parametros_classifier
        )
    )

    print(
        "Demais pesos:",
        "tambem treinaveis"
        if args.integral
        else "congelados"
    )


    # =================================================
    # 12 - ADAM COM TAXAS DIFERENTES
    # =================================================

    optimizer = optim.Adam([

        {

            "params": parametros_features,

            "lr": args.lr * 0.2,

            "weight_decay":
                args.weight_decay * 0.2

        },

        {

            "params": parametros_classifier,

            "lr": args.lr,

            "weight_decay":
                args.weight_decay

        }

    ])

    criterion = nn.CrossEntropyLoss().to(
        device
    )

    assert math.isclose(

        optimizer.param_groups[0]["lr"],

        args.lr * 0.2

    )

    assert math.isclose(

        optimizer.param_groups[1]["lr"],

        args.lr

    )

    print(
        "\n6 - OTIMIZADOR ADAM / GRUPOS"
    )

    for i, grupo in enumerate(
        optimizer.param_groups
    ):

        quantidade = sum(
            p.numel()
            for p in grupo["params"]
        )

        print(
            f"Grupo {i}"
            f" | lr={grupo['lr']}"
            f" | weight_decay="
            f"{grupo['weight_decay']}"
            f" | parametros={quantidade}"
        )

    print(
        "Loss: CrossEntropyLoss (logits)"
    )


    # =================================================
    # 13 - GRAFICO DA ARQUITETURA
    # =================================================

    fig, ax = plt.subplots(
        figsize=(10, 4.2)
    )

    ax.axis("off")

    linhas = [

        (
            "Entrada CIFAR-10",
            "RGB 32x32 -> ImageNet 224x224"
        ),

        (
            "Features VGG16-BN",
            "Pesos pre-treinados; LR menor"
        ),

        (
            "Classifier VGG16",
            "Saida adaptada para 10 classes"
        ),

        (
            "Adam",
            "Dois grupos de parametros"
        ),

        (
            "Resultado",
            "10 logits por imagem"
        )

    ]

    for i, (nome, descricao) in enumerate(
        linhas
    ):

        y = 0.89 - i * 0.18

        ax.text(
            0.02,
            y,
            nome,
            weight="bold",
            transform=ax.transAxes
        )

        ax.text(
            0.30,
            y,
            descricao,
            transform=ax.transAxes
        )

    ax.set_title(
        "Aula 23: VGG16-BN para fine-tuning"
    )

    salvar(
        fig,
        "90_vgg16_finetuning_arquitetura.png"
    )


    # =================================================
    # 14 - GUARDAR PESOS INICIAIS
    # =================================================

    pesos_features_antes = (

        net.features[40]
        .weight.detach()
        .clone()

    )

    pesos_saida_antes = (

        net.classifier[-1]
        .weight.detach()
        .clone()

    )

    historico = []

    melhor_loss_val = float("inf")
    melhor_epoca = None

    melhor = (
        MODELOS
        / "vgg16_cifar10_aula23_melhor_parcial.pth"
    )


    # =================================================
    # 15 - TREINAMENTO E VALIDACAO
    # =================================================

    print(
        "\n7 - TREINAMENTO E VALIDACAO",
        flush=True
    )

    for epoca in range(
        1,
        args.epocas + 1
    ):

        print(
            f"\nEPOCA {epoca}/{args.epocas}",
            flush=True
        )

        # Treinamento

        loss_t, acc_t, n_t = ciclo(

            net,

            loader_t,

            criterion,

            device,

            optimizer,

            limite_t,

            "Treino"

        )

        # Validacao

        loss_v, acc_v, n_v = ciclo(

            net,

            loader_v,

            criterion,

            device,

            None,

            limite_v,

            "Validacao"

        )

        historico.append({

            "epoca": epoca,

            "loss_treino": loss_t,

            "loss_validacao": loss_v,

            "acuracia_treino": acc_t,

            "acuracia_validacao": acc_v

        })

        print(

            f"Epoca {epoca:03d}/{args.epocas}"

            f" | treino loss={loss_t:.4f}"

            f" acc={acc_t:.2f}%"

            f" ({n_t} imagens)"

            f" | validacao loss={loss_v:.4f}"

            f" acc={acc_v:.2f}%"

            f" ({n_v} imagens)",

            flush=True

        )

        salvar_csv(
            historico
        )

        if loss_v < melhor_loss_val:

            melhor_loss_val = loss_v
            melhor_epoca = epoca

            # Salvar apenas os parametros
            # adaptados, para economizar
            # espaco em disco.
            #
            # A reconstrucao completa
            # depende dos pesos ImageNet.

            selecionados = {}

            for nome, tensor in (
                net.state_dict().items()
            ):

                guardar = (

                    args.integral

                    or nome.startswith(
                        "classifier.6."
                    )

                    or (

                        nome.startswith(
                            "features."
                        )

                        and int(
                            nome.split(".")[1]
                        ) >= 34

                    )

                )

                if guardar:

                    selecionados[nome] = (

                        tensor.detach()
                        .cpu()
                        .clone()

                    )

            torch.save(

                {

                    "epoca": epoca,

                    "modo": (
                        "integral"
                        if args.integral
                        else "parcial"
                    ),

                    "melhor_loss_validacao":
                        loss_v,

                    "pesos_adaptados":
                        selecionados,

                    "modelo_base":
                        "vgg16_bn ImageNet IMAGENET1K_V1",

                    "classes":
                        CLASSES

                },

                melhor

            )

            print(
                "Melhor checkpoint salvo:",
                melhor,
                flush=True
            )


    # =================================================
    # 16 - COMPROVAR ATUALIZACAO DE PESOS
    # =================================================

    pesos_features_depois = (

        net.features[40]
        .weight.detach()

    )

    pesos_saida_depois = (

        net.classifier[-1]
        .weight.detach()

    )

    atualizou_features = not torch.equal(

        pesos_features_antes,

        pesos_features_depois

    )

    atualizou_saida = not torch.equal(

        pesos_saida_antes,

        pesos_saida_depois

    )

    assert (
        atualizou_features
        and atualizou_saida
    ), "Nenhuma atualizacao detectada."

    assert len(
        optimizer.state
    ) > 0

    print(
        "\n8 - VALIDACAO DOS PESOS"
    )

    print(
        "Ultimo bloco convolucional atualizado:",
        atualizou_features
    )

    print(
        "Nova camada classificadora atualizada:",
        atualizou_saida
    )

    print(
        "VALIDACAO DO FINE-TUNING: OK"
    )


    # =================================================
    # 17 - GRAFICOS
    # =================================================

    graficos(
        historico
    )


    # =================================================
    # 18 - AVALIAR O MELHOR CHECKPOINT
    # =================================================

    checkpoint = torch.load(

        melhor,

        map_location="cpu",

        weights_only=True

    )

    net.load_state_dict(

        checkpoint["pesos_adaptados"],

        strict=False

    )

    perda_teste, acc_teste, n_teste = ciclo(

        net,

        loader_teste,

        criterion,

        device,

        None,

        limite_teste,

        "Teste"

    )

    print(
        "\n9 - TESTE"
    )

    print(
        "Melhor epoca:",
        melhor_epoca
    )

    print(
        f"Teste loss={perda_teste:.4f}"

        f" | acc={acc_teste:.2f}%"

        f" | {n_teste} imagens"
    )

    print(
        "Resultado:",
        "TESTE COMPLETO"

        if args.todos_os_dados

        else (
            "TESTE REDUZIDO; "
            "NAO REPRESENTA O CIFAR-10 COMPLETO"
        )
    )

    print(
        "\nAULA 23 EXECUTADA COM SUCESSO!"
    )


if __name__ == "__main__":

    main()
