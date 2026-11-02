# ALURA - AULA 21: FLUXO DE TREINAMENTO DA CNN
# Reutiliza a LeNet-5 RGB criada na Aula 19.
# Dados reais CIFAR-10; treino, validacao e teste separados.

from pathlib import Path
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from aula19_cnn_do_zero import construir_lenet_rgb, CLASSES


BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
MODELOS = BASE / "modelos"

RESULTADOS.mkdir(exist_ok=True, parents=True)
MODELOS.mkdir(exist_ok=True, parents=True)


# =====================================================
# 1 - SALVAR GRAFICOS
# =====================================================

def salvar(fig, nome):

    caminho = RESULTADOS / nome

    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)

    print("Grafico salvo:", caminho, flush=True)


# =====================================================
# 2 - FUNCAO DE TREINAMENTO E VALIDACAO
# =====================================================

def ciclo(
    modelo,
    carregador,
    criterio,
    dispositivo,
    otimizador=None,
    limite=None
):

    # Quando existe otimizador:
    # treinamento com atualizacao de pesos.
    #
    # Sem otimizador:
    # avaliacao, sem gradientes.

    treinamento = otimizador is not None

    modelo.train(treinamento)

    perda_total = 0.0
    acertos = 0
    total = 0
    passos = 0

    for imagens, rotulos in carregador:

        if limite is not None and passos >= limite:
            break

        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        if treinamento:

            # Etapa 1: limpar gradientes.
            otimizador.zero_grad(set_to_none=True)

            # Etapa 2: forward.
            logits = modelo(imagens)

            # Etapa 3: calcular a perda.
            perda = criterio(logits, rotulos)

            # Etapa 4: calcular gradientes.
            perda.backward()

            # Etapa 5: atualizar os pesos.
            otimizador.step()

        else:

            # Validacao/teste:
            # nao atualizar pesos.

            with torch.inference_mode():

                logits = modelo(imagens)
                perda = criterio(logits, rotulos)

        # Escolher a classe com maior logit.
        # Nao e necessario Softmax para argmax.

        predicoes = logits.argmax(dim=1)

        acertos += int(
            (predicoes == rotulos).sum().item()
        )

        n = int(rotulos.size(0))

        perda_total += (
            float(perda.detach().item()) * n
        )

        total += n
        passos += 1

    if total == 0:
        raise RuntimeError(
            "Nenhum exemplo foi processado."
        )

    perda_media = perda_total / total
    acuracia = 100.0 * acertos / total

    return perda_media, acuracia, total, passos


# =====================================================
# 3 - SALVAR HISTORICO EM CSV
# =====================================================

def salvar_historico(historico):

    destino = (
        RESULTADOS
        / "aula21_historico_treinamento.csv"
    )

    with destino.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=[
                "epoca",
                "loss_treino",
                "loss_validacao",
                "acuracia_treino",
                "acuracia_validacao",
                "imagens_treino",
                "imagens_validacao"
            ]
        )

        escritor.writeheader()
        escritor.writerows(historico)

    print(
        "Historico salvo:",
        destino,
        flush=True
    )


# =====================================================
# 4 - GRAFICOS DE LOSS E ACURACIA
# =====================================================

def salvar_curvas(historico):

    epocas = [
        h["epoca"]
        for h in historico
    ]

    # Curva da perda.
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        epocas,
        [
            h["loss_treino"]
            for h in historico
        ],
        "o-",
        label="Treino"
    )

    ax.plot(
        epocas,
        [
            h["loss_validacao"]
            for h in historico
        ],
        "o-",
        label="Validacao"
    )

    ax.set(
        xlabel="Epoca",
        ylabel="CrossEntropyLoss",
        title="LeNet-5: perda por epoca"
    )

    if len(epocas) <= 15:
        ax.set_xticks(epocas)

    ax.legend()
    ax.grid(alpha=0.25)

    salvar(
        fig,
        "83_lenet5_loss_treino_validacao.png"
    )

    # Curva de acuracia.
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        epocas,
        [
            h["acuracia_treino"]
            for h in historico
        ],
        "o-",
        label="Treino"
    )

    ax.plot(
        epocas,
        [
            h["acuracia_validacao"]
            for h in historico
        ],
        "o-",
        label="Validacao"
    )

    ax.set(
        xlabel="Epoca",
        ylabel="Acuracia (%)",
        title="LeNet-5: acuracia por epoca"
    )

    ax.set_ylim(0, 100)

    if len(epocas) <= 15:
        ax.set_xticks(epocas)

    ax.legend()
    ax.grid(alpha=0.25)

    salvar(
        fig,
        "84_lenet5_acuracia_treino_validacao.png"
    )


# =====================================================
# 5 - VISUALIZAR PREDICOES DO MODELO
# =====================================================

def mostrar_predicoes(
    modelo,
    carregador,
    dispositivo
):

    modelo.eval()

    imagens, rotulos = next(
        iter(carregador)
    )

    with torch.inference_mode():

        logits = modelo(
            imagens.to(dispositivo)
        )

        indices = (
            logits.argmax(dim=1).cpu()
        )

        probabilidades = (
            torch.softmax(logits, dim=1)
            .max(dim=1)
            .values.cpu()
        )

    fig, axs = plt.subplots(
        2, 4,
        figsize=(12, 6)
    )

    for i, ax in enumerate(axs.flat):

        imagem = (
            imagens[i]
            .permute(1, 2, 0)
            .numpy()
            .clip(0, 1)
        )

        ax.imshow(imagem)

        classe_real = CLASSES[
            int(rotulos[i])
        ]

        classe_prevista = CLASSES[
            int(indices[i])
        ]

        confianca = (
            probabilidades[i].item()
        )

        ax.set_title(
            f"Real: {classe_real}\n"
            f"Prevista: {classe_prevista} "
            f"({confianca:.1%})",
            fontsize=9
        )

        ax.axis("off")

    fig.suptitle(
        "CIFAR-10: previsoes no conjunto de teste"
    )

    salvar(
        fig,
        "85_lenet5_previsoes_teste.png"
    )


# =====================================================
# 6 - FUNCAO PRINCIPAL
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Treinamento LeNet-5/CIFAR-10 - Aula 21"
        )
    )

    parser.add_argument(
        "--completo",
        action="store_true",
        help="Usar todos os dados e 100 epocas"
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=None
    )

    parser.add_argument(
        "--lotes-treino",
        type=int,
        default=None
    )

    parser.add_argument(
        "--lotes-validacao",
        type=int,
        default=None
    )

    parser.add_argument(
        "--lotes-teste",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # Modo inicial:
    # 2 epocas com numero reduzido de lotes.
    #
    # --completo:
    # 100 epocas com todos os lotes.

    epocas = (
        args.epocas
        if args.epocas is not None
        else (100 if args.completo else 2)
    )

    limite_treino = (
        args.lotes_treino
        if args.lotes_treino is not None
        else (None if args.completo else 100)
    )

    limite_validacao = (
        args.lotes_validacao
        if args.lotes_validacao is not None
        else (None if args.completo else 25)
    )

    limite_teste = (
        args.lotes_teste
        if args.lotes_teste is not None
        else (None if args.completo else 25)
    )

    if epocas <= 0:
        parser.error(
            "A quantidade de epocas deve ser positiva."
        )

    for limite in [
        limite_treino,
        limite_validacao,
        limite_teste
    ]:
        if limite is not None and limite <= 0:
            parser.error(
                "Os limites devem ser positivos."
            )

    # Reprodutibilidade.
    torch.manual_seed(21)
    np.random.seed(21)

    # Detectar CPU ou GPU.
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if dispositivo.type == "cpu":

        torch.set_num_threads(
            min(4, torch.get_num_threads())
        )

    hiperparametros = {
        "batch_size": 32,
        "learning_rate": 10e-3,
        "weight_decay": 5e-4,
        "device": dispositivo
    }

    print("=" * 66)
    print(
        "AULA 21 - FLUXO DE TREINAMENTO | ALURA"
    )
    print("=" * 66)

    print(
        "Dispositivo:",
        dispositivo
    )

    print(
        "Epocas:",
        epocas
    )

    print(
        "Batch size:",
        hiperparametros["batch_size"]
    )

    print(
        "Adam: LR=0.01 | weight_decay=0.0005"
    )

    print(
        "Lotes de treino por epoca:",
        limite_treino
        if limite_treino is not None
        else "TODOS"
    )

    print(
        "Lotes de validacao:",
        limite_validacao
        if limite_validacao is not None
        else "TODOS"
    )

    print(
        "Lotes de teste:",
        limite_teste
        if limite_teste is not None
        else "TODOS"
    )

    if not args.completo:

        print(
            "MODO REDUZIDO: resultados parciais."
        )

    # =================================================
    # 7 - CARREGAR CIFAR-10 REAL
    # =================================================

    raiz = BASE / "dados"

    converter = transforms.ToTensor()

    treino_total = datasets.CIFAR10(
        root=str(raiz),
        train=True,
        download=False,
        transform=converter
    )

    teste = datasets.CIFAR10(
        root=str(raiz),
        train=False,
        download=False,
        transform=converter
    )

    assert len(treino_total) == 50000
    assert len(teste) == 10000

    # Separacao:
    #
    # 45.000 treino
    # 5.000 validacao
    # 10.000 teste final
    #
    # A validacao e retirada do conjunto
    # de treinamento, preservando o teste
    # para avaliacao final.

    treino, validacao = random_split(
        treino_total,
        [45000, 5000],
        generator=torch.Generator().manual_seed(21)
    )

    treino_loader = DataLoader(
        treino,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        validacao,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    teste_loader = DataLoader(
        teste,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(
        "\nCIFAR-10 REAL:"
        " treino=45000,"
        " validacao=5000,"
        " teste=10000"
    )

    print(
        "Lotes totais:",
        len(treino_loader),
        len(val_loader),
        len(teste_loader)
    )

    # =================================================
    # 8 - REUTILIZAR A LENET-5 DA AULA 19
    # =================================================

    modelo = construir_lenet_rgb()

    modelo = modelo.to(dispositivo)

    criterio = nn.CrossEntropyLoss()

    criterio = criterio.to(dispositivo)

    optimizer = optim.Adam(
        modelo.parameters(),
        lr=hiperparametros["learning_rate"],
        weight_decay=hiperparametros["weight_decay"]
    )

    # Guardar pesos iniciais para comprovar
    # que o treinamento os modificou.

    pesos_antes = (
        modelo[0][0]
        .weight.detach()
        .clone()
    )

    historico = []

    melhor_val_loss = float("inf")
    melhor_epoca = 0

    melhor_arquivo = (
        MODELOS
        / "lenet5_cifar10_aula21_melhor.pth"
    )

    ultimo_arquivo = (
        MODELOS
        / "lenet5_cifar10_aula21_ultimo.pth"
    )

    inicio = time.perf_counter()

    # =================================================
    # 9 - TREINAR E VALIDAR POR EPOCA
    # =================================================

    for epoca in range(1, epocas + 1):

        # Treinamento:
        # forward + loss + backward + Adam.

        loss_t, acc_t, n_t, passos_t = ciclo(
            modelo,
            treino_loader,
            criterio,
            dispositivo,
            otimizador=optimizer,
            limite=limite_treino
        )

        # Validacao:
        # forward + loss + accuracy.
        # Sem atualizar pesos.

        loss_v, acc_v, n_v, passos_v = ciclo(
            modelo,
            val_loader,
            criterio,
            dispositivo,
            otimizador=None,
            limite=limite_validacao
        )

        historico.append({
            "epoca": epoca,
            "loss_treino": loss_t,
            "loss_validacao": loss_v,
            "acuracia_treino": acc_t,
            "acuracia_validacao": acc_v,
            "imagens_treino": n_t,
            "imagens_validacao": n_v
        })

        print(
            f"\nEpoca {epoca:03d}/{epocas}"
            f" | treino loss={loss_t:.4f}"
            f" acc={acc_t:.2f}%"
            f" ({n_t} imagens, {passos_t} lotes)"
            f" | validacao loss={loss_v:.4f}"
            f" acc={acc_v:.2f}%"
            f" ({n_v} imagens, {passos_v} lotes)",
            flush=True
        )

        # Salvar CSV apos cada epoca.
        salvar_historico(historico)

        # Checkpoint para guardar os pesos
        # e as configuracoes do modelo.

        estado = {
            "epoca": epoca,
            "modelo": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hiperparametros": {
                "batch_size": 32,
                "learning_rate": 0.01,
                "weight_decay": 0.0005
            },
            "loss_validacao": loss_v,
            "acuracia_validacao": acc_v,
            "treinamento_completo": bool(
                args.completo
                and limite_treino is None
                and limite_validacao is None
            )
        }

        torch.save(
            estado,
            ultimo_arquivo
        )

        # Guardar a epoca com a menor
        # perda de validacao.

        if loss_v < melhor_val_loss:

            melhor_val_loss = loss_v
            melhor_epoca = epoca

            torch.save(
                estado,
                melhor_arquivo
            )

    # =================================================
    # 10 - VALIDAR ATUALIZACAO DOS PESOS
    # =================================================

    pesos_depois = (
        modelo[0][0].weight.detach()
    )

    assert not torch.equal(
        pesos_antes,
        pesos_depois
    ), "Os pesos nao foram atualizados."

    assert len(optimizer.state) > 0

    print(
        "\nVALIDACAO:"
        " backward() e optimizer.step()"
        " atualizaram os pesos: OK"
    )

    # =================================================
    # 11 - GERAR GRAFICOS
    # =================================================

    salvar_curvas(historico)

    # =================================================
    # 12 - TESTAR O MELHOR MODELO
    # =================================================

    # Usar a melhor epoca de validacao.
    # O teste e realizado somente ao final.

    checkpoint = torch.load(
        melhor_arquivo,
        map_location=dispositivo,
        weights_only=True
    )

    modelo.load_state_dict(
        checkpoint["modelo"]
    )

    loss_teste, acc_teste, n_teste, lotes_teste = ciclo(
        modelo,
        teste_loader,
        criterio,
        dispositivo,
        otimizador=None,
        limite=limite_teste
    )

    mostrar_predicoes(
        modelo,
        teste_loader,
        dispositivo
    )

    # =================================================
    # 13 - RESUMO
    # =================================================

    print(
        "\nMELHOR EPOCA NA VALIDACAO:",
        melhor_epoca
    )

    print(
        f"TESTE FINAL:"
        f" loss={loss_teste:.4f}"
        f" | accuracy={acc_teste:.2f}%"
        f" | {n_teste} imagens"
        f" / {lotes_teste} lotes"
    )

    print(
        "Teste:",
        "COMPLETO"
        if limite_teste is None
        else "PARCIAL (modo reduzido)"
    )

    print(
        "Tempo decorrido:",
        round(
            time.perf_counter() - inicio,
            1
        ),
        "segundos"
    )

    print(
        "Melhor modelo salvo:",
        melhor_arquivo
    )

    print(
        "Ultimo modelo salvo:",
        ultimo_arquivo
    )

    print(
        "\nOBSERVACAO:"
    )

    print(
        "A loss de treino pode oscilar."
    )

    print(
        "A validacao mede generalizacao."
    )

    print(
        "Uma loss menor no treino nao"
        " garante melhora na validacao."
    )

    print(
        "\nAULA 21 EXECUTADA COM SUCESSO!"
    )


if __name__ == "__main__":
    main()
