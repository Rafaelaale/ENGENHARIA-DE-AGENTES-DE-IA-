# ============================================================
# PROJETO 04 - AULA 17
# TREINAMENTO E GERACAO CONDICIONAL BRAIN MRI
# ============================================================

from pathlib import Path
import argparse
import ast
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
AULAS = BASE / "aulas"
DADOS = BASE / "dados" / "brain-mri-images"
SAIDA = BASE / "resultados"

SAIDA.mkdir(parents=True, exist_ok=True)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

T = 1000
img_size = 64
SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=[
        "teste",
        "reduzido",
        "completo",
        "gerar",
        "progresso"
    ],
    default="teste"
)

parser.add_argument(
    "--pesos",
    choices=["reduzido", "completo"],
    default="reduzido"
)

parser.add_argument(
    "--classe",
    choices=["x", "y", "z", "todas"],
    default="todas"
)

parser.add_argument(
    "--amostras",
    type=int,
    default=1
)

parser.add_argument(
    "--checkpoint-cada",
    type=int,
    default=8
)

args = parser.parse_args()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

print("=" * 65)
print("AULA 17 - TREINAMENTO E GERACAO CONDICIONAL")
print("=" * 65)
print("Modo:", args.modo)
print("Dispositivo:", device)


# ============================================================
# 2 - REUTILIZAR AS AULAS 15 E 16
# ============================================================

def importar_definicoes(caminho, nomes):

    arvore = ast.parse(
        caminho.read_text(encoding="utf-8-sig")
    )

    definicoes = {
        n.name: n
        for n in arvore.body
        if isinstance(
            n, (ast.FunctionDef, ast.ClassDef)
        )
    }

    ausentes = [
        n for n in nomes
        if n not in definicoes
    ]

    if ausentes:
        raise RuntimeError(
            f"Definicoes ausentes em "
            f"{caminho.name}: {ausentes}"
        )

    modulo = ast.Module(
        body=[
            definicoes[n]
            for n in nomes
        ],
        type_ignores=[]
    )

    ast.fix_missing_locations(modulo)

    exec(
        compile(
            modulo,
            str(caminho),
            "exec"
        ),
        globals()
    )


print("\n1 - CARREGANDO ARQUITETURA CONDICIONAL")

importar_definicoes(
    AULAS / "aula15_unet_condicional.py",
    [
        "BrainMRIConditionalDataset",
        "transform_mri",
        "sinusoidal_embedding",
        "LabelEmbedding",
        "ConditionalTimeEmbedding",
        "ConditionalResidualBlock",
        "ConditionalDown",
        "ConditionalUp",
        "ConditionalUNet"
    ]
)

# Scheduler da Aula 16
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

importar_definicoes(
    AULAS / "aula16_difusao_condicional_guidance.py",
    [
        "q_sample",
        "p_sample_conditional"
    ]
)

print("ConditionalUNet: OK")
print("q_sample: OK")
print("p_sample_conditional: OK")


# ============================================================
# 3 - DATASET X/Y/Z
# ============================================================

print("\n2 - CARREGANDO DATASET MRI")

dataset = BrainMRIConditionalDataset(
    DADOS,
    transform=transform_mri,
    slice_plane=None
)

assert dataset.label_to_idx == {
    "x": 0,
    "y": 1,
    "z": 2
}

print("Total de imagens:", len(dataset))
print("Classes:", dataset.label_to_idx)


# ============================================================
# 4 - MODOS DE TREINAMENTO
# ============================================================

CONFIGS = {
    "teste": {
        "epocas": 1,
        "por_classe": 1,
        "base": 16,
        "micro": 1,
        "acum": 1
    },
    "reduzido": {
        "epocas": 3,
        "por_classe": 16,
        "base": 16,
        "micro": 2,
        "acum": 2
    },
    "completo": {
        "epocas": 100,
        "por_classe": None,
        "base": 64,
        "micro": 1,
        "acum": 128
    }
}

modo_pesos = (
    args.modo
    if args.modo in CONFIGS
    else args.pesos
)

cfg = CONFIGS[modo_pesos]


# ============================================================
# 5 - MODELO CONDICIONAL
# ============================================================

print("\n3 - CONSTRUINDO MODELO")

model = ConditionalUNet(
    in_channels=1,
    base_channels=cfg["base"],
    num_classes=3,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

numero_parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Parametros:", numero_parametros)
print("Base channels:", cfg["base"])
print("Classe nula:", model.null_label)
print("Scheduler:", T, "passos")

assert model.null_label == 3


# ============================================================
# 6 - FUNCAO DE PERDA CONDICIONAL
# ============================================================

def conditional_loss_fn(model, x0, labels, t):

    noise = torch.randn_like(x0)

    xt = q_sample(
        x0,
        t,
        noise
    )

    pred = model(
        xt,
        t,
        labels
    )

    return F.mse_loss(
        pred.float(),
        noise.float()
    )


PREFIXO = f"aula17_{modo_pesos}"

CHECKPOINT = (
    SAIDA / f"{PREFIXO}_checkpoint.pth"
)

PESOS = (
    SAIDA / f"{PREFIXO}_modelo_treinado.pth"
)

HIST_CSV = (
    SAIDA / f"{PREFIXO}_loss.csv"
)

HIST_PNG = (
    SAIDA / f"{PREFIXO}_curva_loss.png"
)


# ============================================================
# 7 - SALVAR HISTORICO
# ============================================================

def registrar_loss(historico):

    with HIST_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        writer = csv.writer(arquivo)

        writer.writerow([
            "epoca",
            "mse_medio",
            "microbatches"
        ])

        for linha in historico:

            writer.writerow([
                linha["epoca"],
                linha["loss"],
                linha["microbatches"]
            ])

    if not historico:
        return

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        [r["epoca"] for r in historico],
        [r["loss"] for r in historico],
        marker="o"
    )

    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE medio")

    ax.set_title(
        f"U-Net condicional - {modo_pesos}"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        HIST_PNG,
        dpi=140
    )

    plt.close(fig)


# ============================================================
# 8 - SNAPSHOTS PARA GIF DO TREINAMENTO
# ============================================================

def salvar_snapshot(epoca):

    if modo_pesos == "reduzido":

        destino = (
            SAIDA
            / f"aula17_reduzido_epoca_{epoca:03d}.pth"
        )

        torch.save(
            model.state_dict(),
            destino
        )


# ============================================================
# 9 - SELECAO EQUILIBRADA DE IMAGENS
# ============================================================

def selecionar_indices():

    if cfg["por_classe"] is None:
        return list(range(len(dataset)))

    rng = np.random.default_rng(2026)

    selecionados = []

    for classe in ("x", "y", "z"):

        lista = [
            i
            for i, valor in enumerate(
                dataset.slice_planes
            )
            if valor == classe
        ]

        escolhidos = rng.choice(
            lista,
            size=cfg["por_classe"],
            replace=False
        )

        selecionados.extend(
            map(int, escolhidos)
        )

    return selecionados


# ============================================================
# 10 - TREINAMENTO
# ============================================================

if args.modo in CONFIGS:

    indices = selecionar_indices()

    total_imagens = len(indices)

    print("\n4 - INICIANDO TREINAMENTO")
    print("Imagens:", total_imagens)
    print("Epocas:", cfg["epocas"])
    print("Microbatch:", cfg["micro"])
    print("Acumulacao:", cfg["acum"])

    print(
        "Batch efetivo:",
        cfg["micro"] * cfg["acum"]
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    print(
        "Mixed precision CUDA:",
        scaler.is_enabled()
    )

    historico = []

    estado = {
        "epoca": 0,
        "prox_batch": 0,
        "loss_total": 0.0,
        "n_batches": 0,
        "updates": 0
    }

    if CHECKPOINT.exists():

        anterior = torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        if (
            anterior["cfg"] != cfg
            or anterior["n"] != total_imagens
        ):

            raise RuntimeError(
                "Checkpoint incompativel."
            )

        model.load_state_dict(
            anterior["model"]
        )

        opt.load_state_dict(
            anterior["optimizer"]
        )

        scaler.load_state_dict(
            anterior["scaler"]
        )

        estado = anterior["estado"]
        historico = anterior["historico"]

        print(
            "Checkpoint: epoca",
            estado["epoca"] + 1
        )

        print(
            "Proximo microbatch:",
            estado["prox_batch"]
        )

    def salvar_checkpoint():

        conteudo = {
            "cfg": cfg,
            "n": total_imagens,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "estado": estado.copy(),
            "historico": historico
        }

        temporario = CHECKPOINT.with_suffix(
            ".tmp"
        )

        torch.save(
            conteudo,
            temporario
        )

        os.replace(
            temporario,
            CHECKPOINT
        )

    if estado["epoca"] >= cfg["epocas"]:

        print(
            "Treinamento ja concluido no checkpoint."
        )

        print(
            "Nenhum batch sera repetido."
        )

    else:

        for epoca in range(
            estado["epoca"],
            cfg["epocas"]
        ):

            gerador = torch.Generator()
            gerador.manual_seed(
                SEED + epoca
            )

            permutacao = torch.randperm(
                total_imagens,
                generator=gerador
            ).tolist()

            ordem = [
                indices[i]
                for i in permutacao
            ]

            total_batches = math.ceil(
                total_imagens / cfg["micro"]
            )

            inicio = (
                estado["prox_batch"]
                if epoca == estado["epoca"]
                else 0
            )

            soma = (
                estado["loss_total"]
                if epoca == estado["epoca"]
                else 0.0
            )

            n_batches = (
                estado["n_batches"]
                if epoca == estado["epoca"]
                else 0
            )

            restantes = ordem[
                inicio * cfg["micro"]:
            ]

            loader = DataLoader(
                Subset(dataset, restantes),
                batch_size=cfg["micro"],
                shuffle=False,
                num_workers=0
            )

            model.train()

            opt.zero_grad(
                set_to_none=True
            )

            print(
                f"\nEpoca {epoca + 1}/"
                f"{cfg['epocas']}"
                f" | microbatches: {total_batches}",
                flush=True
            )

            for offset, batch in enumerate(loader):

                idx = inicio + offset

                seed_batch = (
                    500000
                    + epoca * 100000
                    + idx
                )

                torch.manual_seed(
                    seed_batch
                )

                if device.type == "cuda":

                    torch.cuda.manual_seed_all(
                        seed_batch
                    )

                x0 = batch["image"].to(device)

                labels = batch["label"].to(
                    device
                ).long()

                t = torch.randint(
                    0,
                    T,
                    (len(x0),),
                    device=device
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(device.type == "cuda")
                ):

                    loss = conditional_loss_fn(
                        model,
                        x0,
                        labels,
                        t
                    )

                if not torch.isfinite(loss):

                    raise RuntimeError(
                        "Loss nao finito."
                    )

                ini_grupo = (
                    idx // cfg["acum"]
                ) * cfg["acum"]

                fim_grupo = min(
                    ini_grupo + cfg["acum"],
                    total_batches
                )

                n_grupo = (
                    min(
                        fim_grupo * cfg["micro"],
                        total_imagens
                    )
                    - ini_grupo * cfg["micro"]
                )

                # Peso correto para o ultimo
                # microbatch de tamanho reduzido.
                loss_ajustada = (
                    loss * (len(x0) / n_grupo)
                )

                scaler.scale(
                    loss_ajustada
                ).backward()

                soma += float(loss.item())

                n_batches += 1

                if idx + 1 == fim_grupo:

                    scaler.unscale_(opt)

                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    scaler.step(opt)
                    scaler.update()

                    opt.zero_grad(
                        set_to_none=True
                    )

                    estado = {
                        "epoca": epoca,
                        "prox_batch": idx + 1,
                        "loss_total": soma,
                        "n_batches": n_batches,
                        "updates": estado["updates"] + 1
                    }

                    if (
                        estado["updates"]
                        % max(
                            1,
                            args.checkpoint_cada
                        ) == 0
                    ):

                        salvar_checkpoint()

                        print(
                            f"Batch {idx + 1}/"
                            f"{total_batches}"
                            f" | MSE {soma / n_batches:.6f}"
                            f" | checkpoint OK",
                            flush=True
                        )

            media = soma / n_batches

            historico.append({
                "epoca": epoca + 1,
                "loss": media,
                "microbatches": n_batches
            })

            estado = {
                "epoca": epoca + 1,
                "prox_batch": 0,
                "loss_total": 0.0,
                "n_batches": 0,
                "updates": estado["updates"]
            }

            salvar_checkpoint()
            salvar_snapshot(epoca + 1)
            registrar_loss(historico)

            print(
                f"Epoca {epoca + 1}/"
                f"{cfg['epocas']}"
                f" | loss medio: {media:.6f}"
                f" | checkpoint OK",
                flush=True
            )

    # Salvar pesos finais.
    torch.save(
        model.state_dict(),
        PESOS
    )

    registrar_loss(historico)

    assert len(historico) == cfg["epocas"]
    assert PESOS.exists()
    assert CHECKPOINT.exists()

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - TREINAMENTO")
    print("=" * 65)

    print("Dataset X/Y/Z: OK")
    print("Predicao condicional: OK")
    print("Dropout CFG: OK")
    print("Loss MSE: OK")
    print("Backward: OK")
    print("AdamW: OK")
    print("Gradient clipping: OK")
    print("Checkpoints: OK")
    print("Pesos salvos: OK")

    print(
        "Epocas concluidas:",
        len(historico)
    )

    print(
        "\nGERACAO: EXECUCAO SEPARADA"
    )

    print(
        f"AULA 17 - {args.modo.upper()}"
        f" CONCLUIDO!"
    )

    sys.exit(0)


# ============================================================
# 11 - FUNCOES PARA GERACAO
# ============================================================

def imagem_uint8(x):

    return (
        (
            x.detach()
            .float()
            .cpu()
            .clamp(-1, 1)
            + 1
        ) / 2 * 255
    ).to(
        torch.uint8
    ).numpy()


def grade(x):

    imagens = imagem_uint8(
        x[:, 0]
    )

    return np.concatenate(
        list(imagens),
        axis=1
    )


# ============================================================
# 12 - GERACAO CONDICIONAL DDPM
# ============================================================

def gerar_classe(
    model,
    classe,
    n_amostras,
    prefixo
):

    model.eval()

    label = dataset.label_to_idx[
        classe
    ]

    labels = torch.full(
        (n_amostras,),
        label,
        device=device,
        dtype=torch.long
    )

    x = torch.randn(
        n_amostras,
        1,
        img_size,
        img_size,
        device=device
    )

    primeiro = grade(x)

    frames = [
        np.repeat(
            primeiro[:, :, None],
            3,
            axis=2
        )
    ]

    snapshots = [
        (
            T,
            imagem_uint8(x[0, 0])
        )
    ]

    print(
        f"Gerando classe {classe.upper()}"
        f" | 1000 passos"
        f" | {n_amostras} imagem(ns)",
        flush=True
    )

    with torch.inference_mode():

        for t in reversed(range(T)):

            x = p_sample_conditional(
                model,
                x,
                t,
                labels,
                guidance_scale=3.0
            )

            if t % 50 == 0 or t < 5:

                quadro = grade(x)

                frames.append(
                    np.repeat(
                        quadro[:, :, None],
                        3,
                        axis=2
                    )
                )

                print(
                    f"t={t}: quadro OK",
                    flush=True
                )

            if t % 100 == 0:

                snapshots.append(
                    (
                        t,
                        imagem_uint8(x[0, 0])
                    )
                )

    # GIF da difusao reversa
    gif_path = (
        SAIDA
        / f"{prefixo}_{classe}_diffusion.gif"
    )

    imageio.mimsave(
        gif_path,
        frames,
        duration=0.3,
        loop=0
    )

    # Imagem final da classe
    final_path = (
        SAIDA
        / f"{prefixo}_{classe}_final.png"
    )

    Image.fromarray(
        imagem_uint8(x[0, 0]),
        mode="L"
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST
    ).save(final_path)

    # Snapshots do processo reverso
    fig, axes = plt.subplots(
        1,
        len(snapshots),
        figsize=(
            1.8 * len(snapshots),
            2.4
        )
    )

    for ax, (passo, imagem) in zip(
        axes,
        snapshots
    ):

        ax.imshow(
            imagem,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            f"t={passo}",
            fontsize=9
        )

        ax.axis("off")

    fig.suptitle(
        f"Progresso condicional - {classe.upper()}"
    )

    fig.tight_layout()

    snapshots_path = (
        SAIDA
        / f"{prefixo}_{classe}_snapshots.png"
    )

    fig.savefig(
        snapshots_path,
        dpi=130
    )

    plt.close(fig)

    print("GIF:", gif_path)
    print("Imagem final:", final_path)
    print("Snapshots:", snapshots_path)

    return {
        "classe": classe,
        "gif": str(gif_path),
        "imagem": str(final_path),
        "snapshots": str(snapshots_path)
    }


# ============================================================
# 13 - EXECUTAR GERACAO APOS TREINAMENTO
# ============================================================

if args.modo == "gerar":

    if not 1 <= args.amostras <= 6:

        raise ValueError(
            "Use --amostras de 1 a 6."
        )

    if not PESOS.exists():

        raise FileNotFoundError(
            f"Pesos nao encontrados: {PESOS}. "
            "Treine antes de gerar."
        )

    model.load_state_dict(
        torch.load(
            PESOS,
            map_location=device,
            weights_only=True
        )
    )

    classes = (
        ("x", "y", "z")
        if args.classe == "todas"
        else (args.classe,)
    )

    saidas = []

    for classe in classes:

        saidas.append(
            gerar_classe(
                model,
                classe,
                args.amostras,
                PREFIXO
            )
        )

    resumo = (
        SAIDA
        / f"{PREFIXO}_geracao.json"
    )

    resumo.write_text(
        json.dumps(
            {
                "aula": 17,
                "classes": saidas,
                "guidance": 3.0,
                "timesteps": 1000,
                "amostras": args.amostras,
                "modelo_base": cfg["base"]
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - GERACAO")
    print("=" * 65)

    print("Modelo condicional treinado: OK")
    print("CFG guidance_scale=3.0: OK")

    print(
        "GIFs e imagens:",
        len(saidas),
        "classe(s)"
    )

    print(
        "\nAULA 17 - GERACOES CONCLUIDAS!"
    )

    sys.exit(0)


# ============================================================
# 14 - GIF DO PROGRESSO ENTRE EPOCAS
# ============================================================

if args.modo == "progresso":

    snapshots_pesos = sorted(
        SAIDA.glob(
            f"aula17_{modo_pesos}_epoca_*.pth"
        )
    )

    if not snapshots_pesos:

        raise FileNotFoundError(
            "Snapshots de epocas nao encontrados. "
            "O modo reduzido salva um por epoca."
        )

    frames = []

    for caminho in snapshots_pesos:

        epoca = int(
            caminho.stem.split("_")[-1]
        )

        model.load_state_dict(
            torch.load(
                caminho,
                map_location=device,
                weights_only=True
            )
        )

        model.eval()

        imagens = []

        for classe in ("x", "y", "z"):

            label = torch.tensor(
                [dataset.label_to_idx[classe]],
                device=device
            )

            x = torch.randn(
                1, 1, 64, 64,
                device=device
            )

            with torch.inference_mode():

                for t in reversed(range(T)):

                    x = p_sample_conditional(
                        model,
                        x,
                        t,
                        label,
                        guidance_scale=3.0
                    )

            imagens.append(
                imagem_uint8(x[0, 0])
            )

        quadro = np.repeat(
            np.concatenate(
                imagens,
                axis=1
            )[:, :, None],
            3,
            axis=2
        )

        imagem = Image.fromarray(
            quadro
        )

        ImageDraw.Draw(imagem).text(
            (3, 3),
            f"Epoca {epoca} | X Y Z",
            fill=(255, 0, 0)
        )

        frames.append(
            np.asarray(imagem)
        )

        print(
            f"Epoca {epoca}: "
            "quadro X/Y/Z pronto",
            flush=True
        )

    destino = (
        SAIDA
        / f"{PREFIXO}_training_process.gif"
    )

    imageio.mimsave(
        destino,
        frames,
        duration=0.65,
        loop=0
    )

    print("GIF de treinamento:", destino)

    print(
        "\nAULA 17 - GIF DE TREINAMENTO CONCLUIDO!"
    )
