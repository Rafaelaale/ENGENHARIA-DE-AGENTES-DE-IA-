# ============================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 11 - TREINAMENTO DA MINI U-NET E DDPM
# ============================================================

from pathlib import Path
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader, Subset

import argparse
import ast
import csv
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
AULAS = BASE / "aulas"
DADOS = BASE / "dados" / "pixel-art"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["teste", "reduzido", "completo"],
    default="teste"
)

parser.add_argument(
    "--salvar-cada",
    type=int,
    default=100
)

args = parser.parse_args()

MODO = args.modo
T = 500
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

CONFIG = {
    "teste": {
        "epochs": 1,
        "images": 64,
        "max_batches": 3
    },
    "reduzido": {
        "epochs": 10,
        "images": 512,
        "max_batches": 0
    },
    "completo": {
        "epochs": 10,
        "images": None,
        "max_batches": 0
    }
}[MODO]

EPOCHS = CONFIG["epochs"]
MAX_BATCHES = CONFIG["max_batches"]

CHECKPOINT = (
    RESULTADOS
    / f"aula11_{MODO}_checkpoint.pth"
)

PESOS_FINAIS = (
    RESULTADOS
    / f"aula11_{MODO}_modelo_treinado.pth"
)

CSV_LOSS = (
    RESULTADOS
    / f"aula11_{MODO}_loss.csv"
)

PNG_LOSS = (
    RESULTADOS
    / f"aula11_{MODO}_curva_loss.png"
)

GIF = (
    RESULTADOS
    / f"aula11_{MODO}_diffusion_process.gif"
)

PNG_FINAL = (
    RESULTADOS
    / f"aula11_{MODO}_imagem_final.png"
)

print("=" * 65)
print("AULA 11 - TREINAMENTO DA MINI U-NET")
print("=" * 65)

print("Modo:", MODO)
print("Dispositivo:", device)
print("Epocas:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Timesteps:", T)


# ============================================================
# 2 - REUTILIZAR AS CLASSES DAS AULAS ANTERIORES
# SEM EXECUTAR OS SCRIPTS INTEIROS
# ============================================================

print("\n1 - CARREGANDO ARQUITETURA E DATASET")


def carregar_definicoes(caminho, nomes):

    codigo = caminho.read_text(
        encoding="utf-8-sig"
    )

    arvore = ast.parse(
        codigo,
        filename=str(caminho)
    )

    encontrados = {
        node.name: node
        for node in arvore.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.ClassDef)
        )
    }

    faltando = [
        nome
        for nome in nomes
        if nome not in encontrados
    ]

    if faltando:
        raise RuntimeError(
            f"Definicoes ausentes em {caminho.name}: "
            f"{faltando}"
        )

    modulo = ast.Module(
        body=[
            encontrados[nome]
            for nome in nomes
        ],
        type_ignores=[]
    )

    ast.fix_missing_locations(modulo)

    # Reutiliza somente definicoes do projeto local.
    exec(
        compile(
            modulo,
            str(caminho),
            "exec"
        ),
        globals()
    )


carregar_definicoes(
    AULAS / "aula10_mini_unet.py",
    [
        "sinusoidal_embedding",
        "ConvBlock",
        "Down",
        "Up",
        "MiniUNet"
    ]
)

carregar_definicoes(
    AULAS / "aula09_unet_preparacao_dataset.py",
    [
        "random_horizontal_flip",
        "ajustar_hue",
        "color_jitter",
        "transform",
        "PixelArtDataset"
    ]
)

print("MiniUNet da Aula 10: OK")
print("PixelArtDataset da Aula 09: OK")


# ============================================================
# 3 - DATASET FILTRADO
# ============================================================

print("\n2 - CARREGANDO PIXELART")

dataset = PixelArtDataset(
    images_dir=DADOS,
    captions_file=DADOS / "labels.csv",
    transform=transform,
    filter_label_value=1.0,
    filter_index=3
)

assert len(dataset) > 0

print("Imagens filtradas:", len(dataset))

gerador_indices = torch.Generator()
gerador_indices.manual_seed(2026)

indices_sorteados = torch.randperm(
    len(dataset),
    generator=gerador_indices
).tolist()

if CONFIG["images"] is not None:
    indices_sorteados = indices_sorteados[
        :CONFIG["images"]
    ]

N_IMAGES = len(indices_sorteados)

print("Imagens usadas no modo:", N_IMAGES)


# ============================================================
# 4 - MODELO, OTIMIZADOR E Q_SAMPLE
# ============================================================

print("\n3 - CONFIGURANDO MODELO")

model = MiniUNet().to(device)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4
)

betas = torch.linspace(
    1e-4,
    0.02,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)


def q_sample(x0, t, noise):

    sqrt_alpha = torch.sqrt(
        alpha_bar[t]
    ).view(-1, 1, 1, 1)

    sqrt_one_minus = torch.sqrt(
        1 - alpha_bar[t]
    ).view(-1, 1, 1, 1)

    return (
        sqrt_alpha * x0
        + sqrt_one_minus * noise
    )


print("MiniUNet: OK")
print("AdamW: OK")
print("q_sample: OK")


# ============================================================
# 5 - CHECKPOINTS
# ============================================================

historico = []

estado = {
    "epoch": 0,
    "next_batch": 0,
    "loss_sum": 0.0,
    "batches": 0
}


def salvar_checkpoint():

    conteudo = {
        "mode": MODO,
        "images": N_IMAGES,
        "epochs": EPOCHS,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "state": estado.copy(),
        "history": list(historico)
    }

    temporario = CHECKPOINT.with_suffix(".tmp")

    torch.save(
        conteudo,
        temporario
    )

    os.replace(
        temporario,
        CHECKPOINT
    )


if CHECKPOINT.exists():

    print("\nCheckpoint encontrado. Retomando...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    if (
        checkpoint["mode"] != MODO
        or checkpoint["images"] != N_IMAGES
        or checkpoint["epochs"] != EPOCHS
    ):
        raise RuntimeError(
            "Checkpoint incompatível com o modo."
        )

    model.load_state_dict(
        checkpoint["model"]
    )

    opt.load_state_dict(
        checkpoint["optimizer"]
    )

    estado.update(
        checkpoint["state"]
    )

    historico = checkpoint["history"]

    print(
        "Retomada na epoca:",
        estado["epoch"] + 1
    )

    print(
        "Proximo batch:",
        estado["next_batch"]
    )

else:

    arquivo_inicial = (
        RESULTADOS
        / "aula10_pesos_iniciais_nao_treinados.pth"
    )

    if arquivo_inicial.exists():

        pesos = torch.load(
            arquivo_inicial,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(pesos)

        print(
            "Pesos iniciais da Aula 10: carregados"
        )

    else:

        print(
            "Pesos iniciais da Aula 10 ausentes;"
            " usando inicializacao da arquitetura."
        )


# ============================================================
# 6 - FUNCOES DE RESULTADO
# ============================================================

def salvar_historico():

    with CSV_LOSS.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        writer = csv.writer(arquivo)

        writer.writerow([
            "epoch",
            "loss_medio",
            "batches"
        ])

        for item in historico:

            writer.writerow([
                item["epoch"],
                item["loss"],
                item["batches"]
            ])


def salvar_grafico():

    if not historico:
        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        [x["epoch"] for x in historico],
        [x["loss"] for x in historico],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE medio")

    ax.set_title(
        f"Treinamento Mini U-Net - {MODO}"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        PNG_LOSS,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 7 - TREINAMENTO
# ============================================================

print("\n4 - INICIANDO TREINAMENTO")
print(f"Treinando modelo com {EPOCHS} epoca(s)...")

model.train()

try:

    for epoch in range(
        estado["epoch"],
        EPOCHS
    ):

        # Embaralhamento deterministico por epoca.
        gerador_epoca = torch.Generator()
        gerador_epoca.manual_seed(
            SEED + epoch
        )

        permutacao = torch.randperm(
            N_IMAGES,
            generator=gerador_epoca
        ).tolist()

        indices_epoca = [
            indices_sorteados[i]
            for i in permutacao
        ]

        if epoch == estado["epoch"]:

            inicio_batch = estado["next_batch"]
            epoch_loss = estado["loss_sum"]
            num_batches = estado["batches"]

        else:

            inicio_batch = 0
            epoch_loss = 0.0
            num_batches = 0

        # O modo teste executa somente tres batches.
        if MAX_BATCHES:

            indices_epoca = indices_epoca[
                :MAX_BATCHES * BATCH_SIZE
            ]

        total_batches = math.ceil(
            len(indices_epoca) / BATCH_SIZE
        )

        inicio_imagem = (
            inicio_batch * BATCH_SIZE
        )

        indices_restantes = indices_epoca[
            inicio_imagem:
        ]

        loader = DataLoader(
            Subset(dataset, indices_restantes),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        print(
            f"\nEpoca {epoch + 1}/{EPOCHS}"
            f" - total de batches: {total_batches}"
        )

        for local_idx, batch in enumerate(loader):

            batch_idx = (
                inicio_batch + local_idx
            )

            x0 = batch["image"].to(device)

            noise = torch.randn_like(x0)

            t = torch.randint(
                0,
                T,
                (x0.size(0),),
                device=device
            )

            xt = q_sample(
                x0,
                t,
                noise
            )

            pred = model(xt, t)

            loss = F.mse_loss(
                pred,
                noise
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Loss nao finito."
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            num_batches += 1

            estado.update({
                "epoch": epoch,
                "next_batch": batch_idx + 1,
                "loss_sum": epoch_loss,
                "batches": num_batches
            })

            if (
                num_batches % args.salvar_cada == 0
                or num_batches == total_batches
            ):

                salvar_checkpoint()

                print(
                    f"Batch {num_batches}/{total_batches}"
                    f" | loss medio:"
                    f" {epoch_loss / num_batches:.5f}"
                    f" | checkpoint: OK",
                    flush=True
                )

        if num_batches == 0:
            raise RuntimeError(
                "Nenhum batch treinado."
            )

        avg_loss = epoch_loss / num_batches

        print(
            f"Epoca {epoch + 1}/{EPOCHS}"
            f" - loss medio: {avg_loss:.6f}",
            flush=True
        )

        historico.append({
            "epoch": epoch + 1,
            "loss": float(avg_loss),
            "batches": num_batches
        })

        salvar_historico()
        salvar_grafico()

        estado.update({
            "epoch": epoch + 1,
            "next_batch": 0,
            "loss_sum": 0.0,
            "batches": 0
        })

        salvar_checkpoint()

except KeyboardInterrupt:

    salvar_checkpoint()

    print(
        "\nTreinamento interrompido."
        " Checkpoint salvo."
    )

    raise SystemExit(130)


# ============================================================
# 8 - SALVAR MODELO TREINADO
# ============================================================

torch.save(
    model.state_dict(),
    PESOS_FINAIS
)

print("\n5 - MODELO SALVO")
print("Pesos:", PESOS_FINAIS)

# O teste rapido valida o treinamento,
# mas nao executa geracao de imagens.

if MODO == "teste":

    assert len(historico) == 1
    assert PESOS_FINAIS.exists()

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - TESTE")
    print("=" * 65)

    print("Modelo: OK")
    print("Forward: OK")
    print("Loss MSE: OK")
    print("Backward: OK")
    print("AdamW: OK")
    print("Checkpoint: OK")
    print("Pesos salvos: OK")

    print("\nTESTE DE TREINAMENTO CONCLUIDO!")
    raise SystemExit(0)


# ============================================================
# 9 - PROCESSO REVERSO DDPM
# ============================================================

print("\n6 - EXECUTANDO PROCESSO REVERSO DDPM")

model.eval()


@torch.no_grad()
def p_sample_ddpm(
    model,
    x,
    t,
    betas,
    alphas,
    alpha_bar,
    add_noise=True
):

    B = x.size(0)

    t_tensor = torch.full(
        (B,),
        t,
        device=device,
        dtype=torch.long
    )

    pred_noise = model(
        x,
        t_tensor
    )

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]

    coeff1 = 1 / torch.sqrt(
        alpha_t
    )

    coeff2 = beta_t / torch.sqrt(
        1 - alpha_bar_t
    )

    mean = coeff1 * (
        x - coeff2 * pred_noise
    )

    if add_noise and t > 0:

        alpha_bar_prev = alpha_bar[t - 1]

        posterior_variance = (
            beta_t
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        noise = torch.randn_like(x)

        return (
            mean
            + torch.sqrt(
                posterior_variance
            ) * noise
        )

    return mean


# ============================================================
# 10 - GERAR OITO IMAGENS E GIF
# ============================================================

print("\n7 - GERANDO GIF COM 8 IMAGENS")

x = torch.randn(
    8, 3, 16, 16,
    device=device
)

frames = []


def criar_grid(x):

    imagens = (
        ((x.detach().cpu().clamp(-1, 1) + 1) / 2)
        .permute(0, 2, 3, 1)
        .numpy()
    )

    partes = []

    for i, imagem in enumerate(imagens):

        partes.append(imagem)

        if i < len(imagens) - 1:

            partes.append(
                np.ones(
                    (16, 2, 3),
                    dtype=np.float32
                )
            )

    grid = np.concatenate(
        partes,
        axis=1
    )

    grid = (
        grid * 255
    ).astype(np.uint8)

    # Ampliar apenas para visualizar melhor.
    imagem_pil = Image.fromarray(grid)

    imagem_pil = imagem_pil.resize(
        (
            imagem_pil.width * 4,
            imagem_pil.height * 4
        ),
        Image.Resampling.NEAREST
    )

    return np.array(imagem_pil)


frames.append(criar_grid(x))

with torch.no_grad():

    for t in reversed(range(T)):

        x = p_sample_ddpm(
            model,
            x,
            t,
            betas,
            alphas,
            alpha_bar
        )

        if t % 50 == 0 or t < 5:

            frames.append(
                criar_grid(x)
            )

            print(
                f"Reverse t={t}: quadro salvo",
                flush=True
            )

imageio.mimsave(
    GIF,
    frames,
    format="GIF",
    duration=0.25,
    loop=0
)

print("GIF gerado:", GIF)


# ============================================================
# 11 - GERAR UMA IMAGEM INDEPENDENTE
# ============================================================

print("\n8 - GERANDO IMAGEM INDIVIDUAL")

x_unica = torch.randn(
    1, 3, 16, 16,
    device=device
)

with torch.no_grad():

    for t in reversed(range(T)):

        x_unica = p_sample_ddpm(
            model,
            x_unica,
            t,
            betas,
            alphas,
            alpha_bar
        )

imagem_final = (
    ((x_unica.detach().cpu().clamp(-1, 1) + 1) / 2)
    .squeeze(0)
    .permute(1, 2, 0)
    .numpy()
)

imagem_final = (
    imagem_final * 255
).astype(np.uint8)

Image.fromarray(
    imagem_final
).resize(
    (256, 256),
    Image.Resampling.NEAREST
).save(PNG_FINAL)

print("Imagem gerada:", PNG_FINAL)


# ============================================================
# 12 - METADADOS E VALIDACAO
# ============================================================

arquivo_json = (
    RESULTADOS
    / f"aula11_{MODO}_resultados.json"
)

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        {
            "aula": 11,
            "modo": MODO,
            "imagens_treinamento": N_IMAGES,
            "epocas": EPOCHS,
            "batch_size": BATCH_SIZE,
            "timesteps": T,
            "learning_rate": 2e-4,
            "device": str(device),
            "historico": historico,
            "frames_gif": len(frames),
            "treinamento_concluido": True
        },
        arquivo,
        indent=4,
        ensure_ascii=False
    )

for arquivo in [
    CHECKPOINT,
    PESOS_FINAIS,
    CSV_LOSS,
    PNG_LOSS,
    GIF,
    PNG_FINAL,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

assert len(historico) == EPOCHS

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Mini U-Net: OK")
print("Dataset PixelArt: OK")
print("Treinamento:", EPOCHS, "epocas")
print("Loss MSE: OK")
print("AdamW: OK")
print("Checkpoints: OK")
print("Modelo treinado: OK")
print("Processo reverso DDPM: OK")
print("GIF de 8 imagens: OK")
print("Imagem individual: OK")
print("Curva de treinamento: OK")
print("CSV e JSON: OK")

print("\nAULA 11 EXECUTADA COM SUCESSO!")
