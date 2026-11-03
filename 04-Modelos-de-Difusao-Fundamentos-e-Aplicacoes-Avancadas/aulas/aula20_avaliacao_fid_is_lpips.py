# ============================================================
# PROJETO 04 - AULA 20
# AVALIACAO FID, IS, LPIPS E MLOPS
# ============================================================

from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent
AULAS = BASE / "aulas"
DADOS = BASE / "dados" / "brain-mri-images"
SAIDA = BASE / "resultados"

SAIDA.mkdir(parents=True, exist_ok=True)

T = 1000
SEED = 42

DESCRICOES = {
    "x": "sagittal",
    "y": "coronal",
    "z": "axial"
}

CONFIGS = {
    "teste": {
        "epocas": 1,
        "por_classe": 1,
        "validacao": 1,
        "base": 16,
        "micro": 1,
        "acum": 1
    },
    "reduzido": {
        "epocas": 3,
        "por_classe": 16,
        "validacao": 4,
        "base": 16,
        "micro": 2,
        "acum": 2
    },
    "completo": {
        "epocas": 50,
        "por_classe": None,
        "validacao": 257,
        "base": 64,
        "micro": 1,
        "acum": 128
    }
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=[
        "teste",
        "reduzido",
        "completo",
        "gerar",
        "avaliar"
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
    default=60
)

parser.add_argument(
    "--passos",
    type=int,
    default=50
)

parser.add_argument(
    "--fid-feature",
    type=int,
    choices=[64, 192, 768, 2048],
    default=64
)

parser.add_argument(
    "--lpips-net",
    choices=["alex", "vgg", "squeeze"],
    default="alex"
)

parser.add_argument(
    "--checkpoint-cada",
    type=int,
    default=8
)

parser.add_argument(
    "--iniciar-aula17",
    action="store_true"
)

args = parser.parse_args()

if (
    not 2 <= args.passos <= T
    or args.amostras < 1
    or args.checkpoint_cada < 1
):
    parser.error(
        "Parametros de passos/amostras/checkpoint invalidos."
    )

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

modo_pesos = (
    args.modo
    if args.modo in CONFIGS
    else args.pesos
)

cfg = CONFIGS[modo_pesos]

PREFIXO = f"aula20_{modo_pesos}"

CHECKPOINT = SAIDA / f"{PREFIXO}_checkpoint.pth"
MELHOR = SAIDA / f"{PREFIXO}_modelo_melhor.pth"
ULTIMO = SAIDA / f"{PREFIXO}_modelo_ultimo.pth"
CSV = SAIDA / f"{PREFIXO}_historico.csv"
CURVA = SAIDA / f"{PREFIXO}_curva.png"
CONFIG = SAIDA / f"{PREFIXO}_configuracao.json"

print("=" * 65)
print("AULA 20 - AVALIACAO DE MODELOS DE DIFUSAO / MLOPS")
print("=" * 65)

print("Modo:", args.modo)
print("Dispositivo:", device)


# ============================================================
# 2 - REUTILIZAR ARQUITETURA DA AULA 15
# ============================================================

def importar_definicoes(caminho, nomes):

    arvore = ast.parse(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )

    definicoes = {
        n.name: n
        for n in arvore.body
        if isinstance(
            n,
            (ast.FunctionDef, ast.ClassDef)
        )
    }

    faltantes = [
        n for n in nomes
        if n not in definicoes
    ]

    if faltantes:
        raise RuntimeError(
            f"Definicoes ausentes: {faltantes}"
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


print("\n1 - REUTILIZANDO ARQUITETURA DA AULA 15")

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

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

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

print("Dataset:", len(dataset))
print("Classes:", dataset.label_to_idx)
print("Descricoes:", DESCRICOES)


# ============================================================
# 3 - MODELO CONDICIONAL
# ============================================================

print("\n2 - MODELO CONDICIONAL")

model = ConditionalUNet(
    in_channels=1,
    base_channels=cfg["base"],
    num_classes=3,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

assert model.null_label == 3

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Parametros:", parametros)
print("Base channels:", cfg["base"])
print("Classe nula CFG:", model.null_label)


# ============================================================
# 4 - SCHEDULER DE DIFUSAO
# ============================================================

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

    ab = alpha_bar[t].reshape(
        -1, 1, 1, 1
    )

    return (
        ab.sqrt() * x0
        + (1 - ab).sqrt() * noise
    )


# ============================================================
# 5 - FUNCAO DE PERDA CONDICIONAL
# ============================================================

def conditional_loss_fn(model, x0, labels, t):

    noise = torch.randn_like(x0)

    x_t = q_sample(
        x0,
        t,
        noise
    )

    pred_noise = model(
        x_t,
        t,
        labels
    )

    return F.mse_loss(
        pred_noise.float(),
        noise.float()
    )


# ============================================================
# 6 - SEPARACAO TREINO / VALIDACAO
# ============================================================

def selecionar_indices():

    rng = np.random.default_rng(2026)

    treino = []
    validacao = []

    for classe in range(3):

        disponiveis = np.asarray([
            i
            for i, label in enumerate(
                dataset.labels
            )
            if label == classe
        ])

        ordem = rng.permutation(
            disponiveis
        ).tolist()

        n_val = cfg["validacao"]

        if cfg["por_classe"] is None:
            n_treino = len(ordem) - n_val
        else:
            n_treino = cfg["por_classe"]

        if n_treino + n_val > len(ordem):
            raise RuntimeError(
                "Dataset insuficiente."
            )

        treino.extend(
            ordem[:n_treino]
        )

        validacao.extend(
            ordem[
                n_treino:n_treino + n_val
            ]
        )

    return treino, validacao


train_indices, val_indices = selecionar_indices()

identificador_dados = hashlib.sha256(
    (
        " ".join(
            str(dataset.image_files[i])
            for i in train_indices + val_indices
        )
    ).encode()
).hexdigest()

print("Imagens de treino:", len(train_indices))
print("Imagens de validacao:", len(val_indices))


# ============================================================
# 7 - METADADOS E CONFIGURACAO
# ============================================================

def salvar_config():

    pacote = {
        "aula": 20,
        "origem": "Brain MRI - U-Net condicional",
        "modelo": "ConditionalUNet",
        "parametros": parametros,
        "base_channels": cfg["base"],
        "classes": dataset.label_to_idx,
        "descricoes": DESCRICOES,
        "classe_nula": 3,
        "diffusion_scheduler": {
            "T": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "treinamento": cfg,
        "n_treino": len(train_indices),
        "n_validacao": len(val_indices),
        "hash_indices": identificador_dados,
        "device": str(device),
        "torch": torch.__version__,
        "inicializacao_aula17_solicitada": bool(
            args.iniciar_aula17
        ),
        "uso": "Educacional. Sem validacao clinica."
    }

    CONFIG.write_text(
        json.dumps(
            pacote,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# 8 - REGISTRAR HISTORICO E CURVAS
# ============================================================

def registrar_historico(historico):

    with CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=[
                "epoca",
                "train_mse",
                "val_mse",
                "learning_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(historico)

    if not historico:
        return

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    epocas = [
        r["epoca"]
        for r in historico
    ]

    ax.plot(
        epocas,
        [r["train_mse"] for r in historico],
        marker="o",
        label="Treino"
    )

    ax.plot(
        epocas,
        [r["val_mse"] for r in historico],
        marker="o",
        label="Validacao"
    )

    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE de ruido")
    ax.set_title("Aula 20 - Treino e Validacao")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        CURVA,
        dpi=140
    )

    plt.close(fig)


# ============================================================
# 9 - LOSS DE VALIDACAO
# ============================================================

@torch.no_grad()
def avaliar_loss_validacao():

    model.eval()

    soma = 0.0
    n = 0

    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg["micro"],
        shuffle=False,
        num_workers=0
    )

    for i, batch in enumerate(loader):

        torch.manual_seed(
            20000 + i
        )

        x = batch["image"].to(device)

        labels = batch["label"].to(
            device,
            dtype=torch.long
        )

        t = torch.randint(
            0,
            T,
            (len(x),),
            device=device
        )

        loss = conditional_loss_fn(
            model,
            x,
            labels,
            t
        )

        soma += float(loss.item()) * len(x)
        n += len(x)

    return soma / n


# ============================================================
# 10 - TREINAMENTO
# ============================================================

if args.modo in CONFIGS:

    print("\n3 - TREINAMENTO MSE / ADAMW / COSINE LR")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epocas"]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    print("Epocas:", cfg["epocas"])
    print("Microbatch:", cfg["micro"])
    print("Acumulacao:", cfg["acum"])

    print(
        "Batch efetivo:",
        cfg["micro"] * cfg["acum"]
    )

    print(
        "Mixed precision CUDA:",
        scaler.is_enabled()
    )

    estado = {
        "epoca": 0,
        "prox_batch": 0,
        "soma": 0.0,
        "n_imagens": 0,
        "updates": 0,
        "sem_melhora": 0,
        "best_val": float("inf")
    }

    historico = []

    if CHECKPOINT.exists():

        ck = torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        if (
            ck["cfg"] != cfg
            or ck["dados_id"] != identificador_dados
        ):
            raise RuntimeError(
                "Checkpoint incompativel."
            )

        model.load_state_dict(
            ck["model"]
        )

        optimizer.load_state_dict(
            ck["optimizer"]
        )

        scheduler.load_state_dict(
            ck["scheduler"]
        )

        scaler.load_state_dict(
            ck["scaler"]
        )

        estado = ck["estado"]
        historico = ck["historico"]

        print(
            "Checkpoint retomado. Proxima epoca:",
            estado["epoca"] + 1
        )

        print(
            "Proximo microbatch:",
            estado["prox_batch"]
        )

    elif args.iniciar_aula17:

        origem = (
            SAIDA
            / f"aula17_{modo_pesos}_modelo_treinado.pth"
        )

        if not origem.exists():
            raise FileNotFoundError(
                f"Pesos da Aula 17 ausentes: {origem}"
            )

        model.load_state_dict(
            torch.load(
                origem,
                map_location=device,
                weights_only=True
            )
        )

        print(
            "Pesos iniciais da Aula 17:",
            origem.name
        )

    else:

        print(
            "Inicializacao: pesos aleatorios."
        )

    salvar_config()

    def salvar_checkpoint():

        ck = {
            "cfg": cfg,
            "dados_id": identificador_dados,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "estado": estado.copy(),
            "historico": historico
        }

        temporario = CHECKPOINT.with_suffix(
            ".tmp"
        )

        torch.save(
            ck,
            temporario
        )

        os.replace(
            temporario,
            CHECKPOINT
        )

    if (
        estado["epoca"] >= cfg["epocas"]
        or estado.get("early_stop", False)
    ):

        print(
            "Treinamento ja finalizado no checkpoint."
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
                len(train_indices),
                generator=gerador
            ).tolist()

            ordem = [
                train_indices[i]
                for i in permutacao
            ]

            total_batches = math.ceil(
                len(ordem) / cfg["micro"]
            )

            inicio = (
                estado["prox_batch"]
                if epoca == estado["epoca"]
                else 0
            )

            soma = (
                estado["soma"]
                if epoca == estado["epoca"]
                else 0.0
            )

            n_imagens = (
                estado["n_imagens"]
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

            optimizer.zero_grad(
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
                    device,
                    dtype=torch.long
                )

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

                if not bool(torch.isfinite(loss)):
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

                total_grupo = (
                    min(
                        fim_grupo * cfg["micro"],
                        len(ordem)
                    )
                    - ini_grupo * cfg["micro"]
                )

                loss_ajustada = (
                    loss
                    * (len(x0) / total_grupo)
                )

                scaler.scale(
                    loss_ajustada
                ).backward()

                soma += (
                    float(loss.item())
                    * len(x0)
                )

                n_imagens += len(x0)

                if idx + 1 == fim_grupo:

                    scaler.unscale_(
                        optimizer
                    )

                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    estado.update(
                        epoca=epoca,
                        prox_batch=idx + 1,
                        soma=soma,
                        n_imagens=n_imagens,
                        updates=estado["updates"] + 1
                    )

                    if (
                        estado["updates"]
                        % args.checkpoint_cada == 0
                    ):

                        salvar_checkpoint()

                        print(
                            f"Batch {idx + 1}/{total_batches}"
                            f" | MSE {soma / n_imagens:.6f}"
                            " | checkpoint OK",
                            flush=True
                        )

            train_mse = soma / n_imagens

            val_mse = avaliar_loss_validacao()

            if (
                val_mse
                < estado["best_val"] - 1e-6
            ):

                estado["best_val"] = val_mse
                estado["sem_melhora"] = 0

                torch.save(
                    model.state_dict(),
                    MELHOR
                )

            else:

                estado["sem_melhora"] += 1

            scheduler.step()

            historico.append({
                "epoca": epoca + 1,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                )
            })

            estado.update(
                epoca=epoca + 1,
                prox_batch=0,
                soma=0.0,
                n_imagens=0
            )

            if (
                modo_pesos == "completo"
                and estado["sem_melhora"] >= 25
            ):
                estado["early_stop"] = True

            salvar_checkpoint()
            registrar_historico(historico)

            print(
                f"Epoca {epoca + 1}/{cfg['epocas']}"
                f" | treino={train_mse:.6f}"
                f" | val={val_mse:.6f}"
                " | checkpoint OK",
                flush=True
            )

            if estado.get("early_stop"):

                print(
                    "Early stopping: 25 epocas "
                    "sem melhora na validacao."
                )

                break

    torch.save(
        model.state_dict(),
        ULTIMO
    )

    registrar_historico(historico)

    assert len(historico) > 0
    assert CHECKPOINT.exists()
    assert MELHOR.exists()
    assert CSV.exists()

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - AULA 20 TREINAMENTO")
    print("=" * 65)

    print("Dataset X/Y/Z: OK")
    print("Loss MSE e backward: OK")
    print("AdamW / CosineAnnealingLR: OK")
    print("GradScaler e gradient clipping: OK")
    print("Checkpoint / melhores pesos / ultimo: OK")

    print(
        "Epocas concluidas:",
        len(historico)
    )

    print(
        "METRICAS FID/IS/LPIPS: execucao separada"
    )

    print(
        f"AULA 20 - {args.modo.upper()} CONCLUIDO!"
    )

    sys.exit(0)


# ============================================================
# 11 - CARREGAR MELHOR MODELO TREINADO
# ============================================================

if not MELHOR.exists():

    raise FileNotFoundError(
        f"Pesos ausentes: {MELHOR}. "
        "Execute --modo reduzido primeiro."
    )

model.load_state_dict(
    torch.load(
        MELHOR,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# 12 - AMOSTRAGEM DDIM COM SALTOS CORRETOS
# ============================================================

@torch.inference_mode()
def amostrar_ddim(
    labels,
    passos=50,
    guidance=3.0,
    seed=None
):

    if seed is not None:

        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    labels = labels.to(
        device=device,
        dtype=torch.long
    )

    x = torch.randn(
        len(labels),
        1,
        64,
        64,
        device=device
    )

    tempos = torch.linspace(
        T - 1,
        0,
        passos
    ).round().long().unique_consecutive().tolist()

    for i, t in enumerate(tempos):

        anterior = (
            tempos[i + 1]
            if i + 1 < len(tempos)
            else -1
        )

        tt = torch.full(
            (len(labels),),
            t,
            device=device,
            dtype=torch.long
        )

        eps_cond = model(
            x,
            tt,
            labels
        )

        eps_uncond = model(
            x,
            tt,
            None
        )

        eps = (
            eps_uncond
            + guidance * (
                eps_cond - eps_uncond
            )
        )

        ab = alpha_bar[t]

        ab_prev = (
            alpha_bar[anterior]
            if anterior >= 0
            else x.new_tensor(1.0)
        )

        x0 = (
            (
                x
                - (1 - ab).sqrt() * eps
            ) / ab.sqrt()
        ).clamp(-1, 1)

        x = (
            ab_prev.sqrt() * x0
            + (1 - ab_prev).sqrt() * eps
        )

    return x.clamp(-1, 1)


# ============================================================
# 13 - PREPARACAO PARA METRICAS
# ============================================================

def grayscale_rgb01(x):

    # FID e IS: RGB em [0,1]
    return (
        (
            x.detach().float()
            .clamp(-1, 1)
            + 1
        ) / 2
    ).repeat(
        1, 3, 1, 1
    )


def grayscale_rgb11(x):

    # LPIPS: RGB em [-1,1]
    return (
        x.detach().float()
        .clamp(-1, 1)
    ).repeat(
        1, 3, 1, 1
    )


def salvar_imagem(imagem, caminho):

    matriz = (
        (
            imagem[0]
            .detach()
            .cpu()
            .clamp(-1, 1)
            + 1
        ) / 2 * 255
    ).to(torch.uint8).numpy()

    Image.fromarray(
        matriz
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST
    ).save(caminho)


# ============================================================
# 14 - GERACAO CONDICIONAL
# ============================================================

if args.modo == "gerar":

    print("\n3 - GERACAO CONDICIONAL DDIM")

    if args.amostras > 6:
        raise ValueError(
            "Use ate 6 amostras por classe."
        )

    classes = (
        ("x", "y", "z")
        if args.classe == "todas"
        else (args.classe,)
    )

    resultados = []

    for classe in classes:

        label = dataset.label_to_idx[
            classe
        ]

        labels = torch.full(
            (args.amostras,),
            label,
            dtype=torch.long,
            device=device
        )

        imagens = amostrar_ddim(
            labels,
            passos=args.passos,
            seed=2026 + label
        )

        for idx, img in enumerate(imagens):

            destino = (
                SAIDA
                / f"{PREFIXO}_{classe}_ddim_{idx + 1:02d}.png"
            )

            salvar_imagem(
                img,
                destino
            )

            resultados.append(
                str(destino)
            )

            print(
                "Imagem:",
                destino.name
            )

    arquivo = (
        SAIDA
        / f"{PREFIXO}_geracao.json"
    )

    arquivo.write_text(
        json.dumps(
            {
                "amostrador": "DDIM eta=0",
                "timesteps_treino": T,
                "passos_inferencia": args.passos,
                "guidance": 3.0,
                "arquivos": resultados
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "AULA 20 - GERACAO CONCLUIDA!"
    )

    sys.exit(0)


# ============================================================
# 15 - AVALIACAO FID / IS / LPIPS
# ============================================================

if args.modo == "avaliar":

    print("\n3 - AVALIACAO FID / IS / LPIPS")

    if (
        args.amostras < 30
        or args.amostras % 3 != 0
    ):
        raise ValueError(
            "Use --amostras multiplo de 3, "
            "com no minimo 30."
        )

    try:

        import torch_fidelity

        from torchmetrics.image.fid import (
            FrechetInceptionDistance
        )

        from torchmetrics.image.inception import (
            InceptionScore
        )

        from torchmetrics.image.lpip import (
            LearnedPerceptualImagePatchSimilarity
        )

    except (ImportError, ModuleNotFoundError) as erro:

        raise RuntimeError(
            "Instale torchmetrics, "
            "torch-fidelity e torchvision."
        ) from erro

    print(
        "AVALIACAO EXPLORATORIA: "
        "amostras pequenas nao validam "
        "qualidade clinica."
    )

    print(
        "FID features:",
        args.fid_feature
    )

    print(
        "LPIPS: diversidade entre pares "
        "sinteticos da mesma classe."
    )

    n_por = args.amostras // 3

    reais = []
    falsas = []

    treino_set = set(train_indices)

    with torch.inference_mode():

        for label in range(3):

            possiveis = [
                i
                for i, classe in enumerate(
                    dataset.labels
                )
                if (
                    classe == label
                    and i not in treino_set
                )
            ]

            if len(possiveis) < n_por:
                raise ValueError(
                    f"Imagens reais insuficientes "
                    f"para classe {label}."
                )

            rng = np.random.default_rng(
                2030 + label
            )

            escolhidos = rng.choice(
                possiveis,
                size=n_por,
                replace=False
            ).tolist()

            for inicio in range(
                0,
                n_por,
                2
            ):

                qtd = min(
                    2,
                    n_por - inicio
                )

                rotulos = torch.full(
                    (qtd,),
                    label,
                    dtype=torch.long,
                    device=device
                )

                fakes = amostrar_ddim(
                    rotulos,
                    passos=args.passos,
                    seed=(
                        123000
                        + label * 1000
                        + inicio
                    )
                )

                falsas.append(
                    fakes.cpu()
                )

                bloco = torch.stack([
                    dataset[i]["image"]
                    for i in escolhidos[
                        inicio:inicio + qtd
                    ]
                ])

                reais.append(bloco)

                print(
                    f"Classe {dataset.idx_to_label[label].upper()}"
                    f" {inicio + qtd}/{n_por}",
                    flush=True
                )

    real = torch.cat(
        reais,
        dim=0
    )

    fake = torch.cat(
        falsas,
        dim=0
    )

    try:

        fid = FrechetInceptionDistance(
            feature=args.fid_feature,
            normalize=True
        ).to(device)

        iscore = InceptionScore(
            splits=10,
            normalize=True
        ).to(device)

        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=args.lpips_net,
            reduction="none",
            normalize=False
        ).to(device)

        with torch.inference_mode():

            for i in range(
                0,
                len(real),
                4
            ):

                fid.update(
                    grayscale_rgb01(
                        real[i:i + 4].to(device)
                    ),
                    real=True
                )

                imagens_geradas = grayscale_rgb01(
                    fake[i:i + 4].to(device)
                )

                fid.update(
                    imagens_geradas,
                    real=False
                )

                iscore.update(
                    imagens_geradas
                )

            fid_num = float(
                fid.compute().item()
            )

            is_mean, is_std = iscore.compute()

            # LPIPS entre amostras sintéticas
            # da MESMA classe para medir
            # diversidade, nao fidelidade.
            pares1 = []
            pares2 = []

            for i in range(
                0,
                len(fake),
                n_por
            ):

                bloco = fake[
                    i:i + n_por
                ]

                if len(bloco) >= 2:

                    pares1.append(
                        bloco[:-1]
                    )

                    pares2.append(
                        bloco[1:]
                    )

            lp_scores = []

            for x1, x2 in zip(
                pares1,
                pares2
            ):

                for j in range(
                    0,
                    len(x1),
                    4
                ):

                    scores = lpips(
                        grayscale_rgb11(
                            x1[j:j + 4].to(device)
                        ),
                        grayscale_rgb11(
                            x2[j:j + 4].to(device)
                        )
                    )

                    lp_scores.extend(
                        scores.detach()
                        .cpu()
                        .flatten()
                        .tolist()
                    )

            lp_mean = float(
                np.mean(lp_scores)
            )

            lp_std = float(
                np.std(lp_scores)
            )

    except Exception as erro:

        print(
            "Erro ao executar metricas:",
            type(erro).__name__,
            str(erro)
        )

        print(
            "Os pesos pre-treinados "
            "de Inception e LPIPS podem "
            "precisar ser baixados."
        )

        raise

    relatorio = {
        "amostras_reais": len(real),
        "amostras_sinteticas": len(fake),
        "fid": fid_num,
        "fid_features": args.fid_feature,
        "is_mean": float(is_mean),
        "is_std": float(is_std),
        "lpips_diversidade_media": lp_mean,
        "lpips_diversidade_std": lp_std,
        "lpips_net": args.lpips_net,
        "amostrador": "DDIM eta=0",
        "passos": args.passos,
        "guidance_scale": 3.0,
        "limitacoes": (
            "Avaliacao exploratoria com amostra "
            "pequena. Inception treinado fora "
            "do dominio de imagens medicas. "
            "LPIPS usado apenas para diversidade "
            "intraclasse. Sem validade clinica."
        )
    }

    destino = (
        SAIDA
        / f"{PREFIXO}_metricas.json"
    )

    destino.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nRESULTADOS DAS METRICAS")

    print("FID:", fid_num)
    print("IS:", float(is_mean))
    print("LPIPS diversidade:", lp_mean)

    print("Relatorio:", destino)

    print(
        "\nAULA 20 - METRICAS CONCLUIDAS!"
    )
