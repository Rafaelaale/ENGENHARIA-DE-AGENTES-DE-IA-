# ============================================================
# PROJETO 04 - AULA 19
# TREINAMENTO COM CONDICIONAMENTO ESPACIAL
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

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageDraw

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

SEED = 42
T = 1000
IMG = 64

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

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
    choices=["teste", "reduzido", "completo"],
    default="reduzido"
)

parser.add_argument(
    "--tipo",
    choices=[
        "intensity",
        "low_intensity",
        "edge",
        "gradient",
        "scharr"
    ],
    default="intensity"
)

parser.add_argument(
    "--amostras-media",
    type=int,
    default=10
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

parser.add_argument(
    "--snapshots-completo",
    action="store_true"
)

args = parser.parse_args()

if (
    args.amostras_media < 1
    or not 1 <= args.amostras <= 6
    or args.checkpoint_cada < 1
):
    parser.error(
        "Parametros de amostras/checkpoint invalidos"
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

print("=" * 64)
print("AULA 19 - TREINAMENTO ESPACIAL / CONTROLNET DIDATICO")
print("=" * 64)
print("Modo:", args.modo)
print("Dispositivo:", device)


# ============================================================
# 2 - IMPORTAR ARQUITETURAS ANTERIORES
# ============================================================

def importar(caminho, nomes):

    arvore = ast.parse(
        caminho.read_text(encoding="utf-8-sig")
    )

    definicoes = {
        n.name: n
        for n in arvore.body
        if isinstance(
            n,
            (ast.ClassDef, ast.FunctionDef)
        )
    }

    faltantes = [
        n for n in nomes
        if n not in definicoes
    ]

    if faltantes:
        raise RuntimeError(
            f"Definicoes ausentes em "
            f"{caminho.name}: {faltantes}"
        )

    modulo = ast.Module(
        body=[definicoes[n] for n in nomes],
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


print("\n1 - IMPORTANDO AULAS 15 E 18")

importar(
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

importar(
    AULAS / "aula18_controlnet_espacial.py",
    [
        "CondicionamentoEspacial",
        "ControlNet",
        "CondSpacialNetPipeline",
        "create_spatial_condition",
        "create_average_spatial_condition",
        "p_sample_condspacial"
    ]
)

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


# ============================================================
# 3 - SCHEDULER DDPM
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


# ============================================================
# 4 - MODOS
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
        "epocas": 120,
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

PREFIXO = (
    f"aula19_{modo_pesos}_"
    f"{args.tipo}_m{args.amostras_media}"
)

CHECKPOINT = SAIDA / f"{PREFIXO}_checkpoint.pth"
PESOS = SAIDA / f"{PREFIXO}_modelo_treinado.pth"
MASCARAS = SAIDA / f"{PREFIXO}_mascaras.pt"
CSV = SAIDA / f"{PREFIXO}_loss.csv"
CURVA = SAIDA / f"{PREFIXO}_loss.png"


# ============================================================
# 5 - PIPELINE CORRIGIDO PARA CFG ESPACIAL
# ============================================================

class PipelineEspacialTreinavel(
    CondSpacialNetPipeline
):

    """
    Durante o dropout CFG, remove tanto
    o rotulo quanto os hints espaciais
    da mesma amostra.
    """

    def forward(
        self,
        x,
        t,
        labels=None,
        spatial_condition=None
    ):

        B = x.size(0)

        t = t.to(
            device=x.device,
            dtype=torch.long
        )

        if t.shape != (B,):
            raise ValueError(
                "Formato de timestep invalido"
            )

        if labels is None:

            ids = torch.full(
                (B,),
                self.null_label,
                device=x.device,
                dtype=torch.long
            )

        else:

            ids = labels.to(
                device=x.device,
                dtype=torch.long
            ).clone()

            validos = (
                (ids >= 0)
                & (ids <= self.null_label)
            )

            if (
                ids.shape != (B,)
                or not bool(validos.all())
            ):
                raise ValueError(
                    "Labels devem estar entre 0 e 3"
                )

            if self.training:

                dropout_mask = (
                    torch.rand(
                        B,
                        device=x.device
                    )
                    < self.unet.dropout_prob
                )

                ids[dropout_mask] = self.null_label

        ativo = ids != self.null_label

        label_emb = self.unet.label_embedding(
            ids
        )

        cond_emb = self.unet.time_embedding(
            t,
            label_emb
        )

        hints = None

        if (
            spatial_condition is not None
            and bool(ativo.any())
        ):

            if spatial_condition.shape != x.shape:
                raise ValueError(
                    "Mascara com dimensoes incorretas"
                )

            hints = self.condspacial(
                spatial_condition.to(
                    device=x.device,
                    dtype=x.dtype
                ),
                cond_emb
            )

            keep = ativo.reshape(
                B, 1, 1, 1
            )

            hints = [
                h * keep.to(h.dtype)
                for h in hints
            ]

        return self.unet(
            x,
            cond_emb,
            hints
        )


model = PipelineEspacialTreinavel(
    in_channels=1,
    base_channels=cfg["base"],
    num_classes=3,
    time_dim=256,
    label_dim=128
).to(device)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

assert model.null_label == 3

print("\n2 - MODELO E MASCARAS")
print("Parametros:", parametros)
print("Base channels:", cfg["base"])
print("Classe nula CFG:", model.null_label)


# ============================================================
# 6 - SALVAR E RECUPERAR MASCARAS
# ============================================================

def obter_mascaras():

    if MASCARAS.exists():

        pacote = torch.load(
            MASCARAS,
            map_location="cpu",
            weights_only=True
        )

        if (
            pacote["tipo"] != args.tipo
            or pacote["amostras_media"]
            != args.amostras_media
            or pacote["dataset_n"]
            != len(dataset)
        ):
            raise RuntimeError(
                "Arquivo de mascaras incompativel"
            )

        print(
            "Mascaras salvas: CARREGADAS"
        )

    elif args.modo in (
        "teste",
        "reduzido",
        "completo"
    ):

        medias = {}

        for classe in range(3):

            media = create_average_spatial_condition(
                dataset,
                classe,
                IMG,
                args.tipo,
                args.amostras_media
            ).cpu()

            assert media.shape == (
                1, 1, 64, 64
            )

            assert torch.isfinite(
                media
            ).all()

            medias[str(classe)] = media

            print(
                f"Mascara media {classe}: OK"
            )

        pacote = {
            "tipo": args.tipo,
            "amostras_media": args.amostras_media,
            "dataset_n": len(dataset),
            "mascaras": medias
        }

        torch.save(
            pacote,
            MASCARAS
        )

        print(
            "Mascaras salvas:",
            MASCARAS.name
        )

    else:

        raise FileNotFoundError(
            f"Mascaras nao encontradas: {MASCARAS}"
        )

    return {
        int(k): v.to(device)
        for k, v in pacote["mascaras"].items()
    }


mascaras = obter_mascaras()

assert set(mascaras) == {0, 1, 2}


def condition_batch(labels):

    return torch.cat(
        [
            mascaras[int(i)]
            for i in labels.tolist()
        ],
        dim=0
    )


# ============================================================
# 7 - FUNCAO DE PERDA ESPACIAL
# ============================================================

def q_sample(x0, t, noise):

    ab = alpha_bar[t].reshape(
        -1, 1, 1, 1
    )

    return (
        ab.sqrt() * x0
        + (1 - ab).sqrt() * noise
    )


def loss_fn_condspace(
    model,
    x0,
    labels,
    spatial_condition,
    t
):

    noise = torch.randn_like(x0)

    x_t = q_sample(
        x0,
        t,
        noise
    )

    pred_noise = model(
        x_t,
        t,
        labels,
        spatial_condition
    )

    return F.mse_loss(
        pred_noise.float(),
        noise.float()
    )


# ============================================================
# 8 - HISTORICO DO TREINAMENTO
# ============================================================

def registrar_loss(historico):

    with CSV.open(
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

        for item in historico:

            writer.writerow([
                item["epoca"],
                item["loss"],
                item["microbatches"]
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
        f"Aula 19 - {modo_pesos} - {args.tipo}"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        CURVA,
        dpi=140
    )

    plt.close(fig)


# ============================================================
# 9 - SELECAO EQUILIBRADA
# ============================================================

def selecionar_indices():

    if cfg["por_classe"] is None:
        return list(range(len(dataset)))

    rng = np.random.default_rng(
        2026
    )

    escolhidos = []

    for classe in range(3):

        disponiveis = [
            i
            for i, label in enumerate(
                dataset.labels
            )
            if label == classe
        ]

        selecao = rng.choice(
            disponiveis,
            size=cfg["por_classe"],
            replace=False
        )

        escolhidos.extend(
            map(int, selecao)
        )

    return escolhidos


# ============================================================
# 10 - TREINAMENTO
# ============================================================

if args.modo in CONFIGS:

    indices = selecionar_indices()
    n_total = len(indices)

    print("\n3 - INICIANDO TREINAMENTO ESPACIAL")

    print("Imagens:", n_total)
    print("Epocas:", cfg["epocas"])
    print("Microbatch:", cfg["micro"])
    print("Acumulacao:", cfg["acum"])

    print(
        "Batch efetivo:",
        cfg["micro"] * cfg["acum"]
    )

    optimizer = torch.optim.AdamW(
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
        "soma": 0.0,
        "n_batches": 0,
        "updates": 0
    }

    # Recuperar checkpoint
    if CHECKPOINT.exists():

        anterior = torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        if (
            anterior["cfg"] != cfg
            or anterior["n_total"] != n_total
        ):
            raise RuntimeError(
                "Checkpoint incompativel"
            )

        model.load_state_dict(
            anterior["model"]
        )

        optimizer.load_state_dict(
            anterior["optimizer"]
        )

        scaler.load_state_dict(
            anterior["scaler"]
        )

        historico = anterior["historico"]
        estado = anterior["estado"]

        print(
            "Checkpoint encontrado."
        )

        print(
            "Retomando epoca:",
            estado["epoca"] + 1
        )

        print(
            "Proximo microbatch:",
            estado["prox_batch"]
        )

    def salvar_checkpoint():

        conteudo = {
            "cfg": cfg,
            "n_total": n_total,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
                n_total,
                generator=gerador
            ).tolist()

            ordem = [
                indices[i]
                for i in permutacao
            ]

            total_batches = math.ceil(
                n_total / cfg["micro"]
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

            n_batches = (
                estado["n_batches"]
                if epoca == estado["epoca"]
                else 0
            )

            restantes = ordem[
                inicio * cfg["micro"]:
            ]

            loader = DataLoader(
                Subset(
                    dataset,
                    restantes
                ),
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
                f"{cfg['epocas']} | "
                f"microbatches: {total_batches}",
                flush=True
            )

            for offset, batch in enumerate(loader):

                idx = inicio + offset

                # Reprodutibilidade ao retomar
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

                spatial_condition = condition_batch(
                    labels
                )

                t = torch.randint(
                    0,
                    T,
                    (len(x0),),
                    device=device
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=(device.type == "cuda")
                ):

                    loss = loss_fn_condspace(
                        model,
                        x0,
                        labels,
                        spatial_condition,
                        t
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Loss nao finito"
                    )

                # Acumulacao ponderada
                ini_grupo = (
                    idx // cfg["acum"]
                ) * cfg["acum"]

                fim_grupo = min(
                    ini_grupo + cfg["acum"],
                    total_batches
                )

                n_imagens_grupo = (
                    min(
                        fim_grupo * cfg["micro"],
                        n_total
                    )
                    - ini_grupo * cfg["micro"]
                )

                loss_ajustada = (
                    loss
                    * (
                        len(x0)
                        / n_imagens_grupo
                    )
                )

                scaler.scale(
                    loss_ajustada
                ).backward()

                soma += float(
                    loss.item()
                )

                n_batches += 1

                if idx + 1 == fim_grupo:

                    scaler.unscale_(
                        optimizer
                    )

                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    estado = {
                        "epoca": epoca,
                        "prox_batch": idx + 1,
                        "soma": soma,
                        "n_batches": n_batches,
                        "updates": estado["updates"] + 1
                    }

                    if (
                        estado["updates"]
                        % args.checkpoint_cada == 0
                    ):

                        salvar_checkpoint()

                        print(
                            f"Batch {idx + 1}/"
                            f"{total_batches} | "
                            f"MSE {soma / n_batches:.6f}"
                            " | checkpoint OK",
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
                "soma": 0.0,
                "n_batches": 0,
                "updates": estado["updates"]
            }

            salvar_checkpoint()
            registrar_loss(historico)

            # Pesos de cada epoca no modo reduzido.
            # No completo, snapshots sao opcionais
            # para evitar ocupar muito disco.
            salvar_snapshot = (
                modo_pesos == "reduzido"
                or (
                    modo_pesos == "completo"
                    and args.snapshots_completo
                    and (epoca + 1) % 10 == 0
                )
            )

            if salvar_snapshot:

                destino = (
                    SAIDA
                    / f"{PREFIXO}_epoca_{epoca + 1:03d}.pth"
                )

                torch.save(
                    model.state_dict(),
                    destino
                )

            print(
                f"Epoca {epoca + 1}/"
                f"{cfg['epocas']} - "
                f"loss medio: {media:.6f}"
                " | checkpoint OK",
                flush=True
            )

    assert len(historico) == cfg["epocas"]

    torch.save(
        model.state_dict(),
        PESOS
    )

    registrar_loss(
        historico
    )

    assert PESOS.exists()
    assert CHECKPOINT.exists()
    assert MASCARAS.exists()

    print("\n" + "=" * 64)
    print("VALIDACAO FINAL - TREINAMENTO ESPACIAL")
    print("=" * 64)

    print("Dataset X/Y/Z: OK")
    print("Mascaras medias salvas: OK")
    print("Loss condicional espacial: OK")
    print("CFG descarta rotulo e mascara: IMPLEMENTADO")
    print("Backward / AdamW / Gradient clipping: OK")
    print("Checkpoints e pesos: OK")

    print(
        "Epocas concluidas:",
        len(historico)
    )

    print(
        "\nGERACAO: EXECUCAO SEPARADA"
    )

    print(
        f"AULA 19 - {args.modo.upper()} CONCLUIDO!"
    )

    sys.exit(0)


# ============================================================
# 11 - CARREGAR PESOS PARA GERACAO
# ============================================================

if not PESOS.exists():
    raise FileNotFoundError(
        f"Pesos treinados nao encontrados: {PESOS}"
    )

model.load_state_dict(
    torch.load(
        PESOS,
        map_location=device,
        weights_only=True
    )
)

model.eval()


def uint8(x):

    x = x.detach().float().cpu()

    x = (
        (x.clamp(-1, 1) + 1)
        / 2 * 255
    )

    return x.to(
        torch.uint8
    ).numpy()


def grade(x):

    imagens = uint8(
        x[:, 0]
    )

    return np.concatenate(
        list(imagens),
        axis=1
    )


# ============================================================
# 12 - GERAR IMAGENS POR CLASSE
# ============================================================

def gerar(
    classe,
    n_amostras,
    prefixo,
    seed=None
):

    if seed is not None:

        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    label = dataset.label_to_idx[
        classe
    ]

    labels = torch.full(
        (n_amostras,),
        label,
        device=device,
        dtype=torch.long
    )

    spatial = mascaras[label].expand(
        n_amostras,
        -1, -1, -1
    )

    x = torch.randn(
        n_amostras,
        1,
        IMG,
        IMG,
        device=device
    )

    frames = [
        np.repeat(
            grade(x)[:, :, None],
            3,
            axis=2
        )
    ]

    snapshots = [
        ("inicio", uint8(x[0, 0]))
    ]

    print(
        f"Gerando classe {classe.upper()}"
        f" | {T} passos"
        f" | {n_amostras} imagem(ns)",
        flush=True
    )

    with torch.inference_mode():

        for t in reversed(range(T)):

            x = p_sample_condspacial(
                model,
                x,
                t,
                labels,
                spatial,
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
                        str(t),
                        uint8(x[0, 0])
                    )
                )

    gif_path = (
        SAIDA
        / f"{prefixo}_{classe}_diffusion.gif"
    )

    imagem_path = (
        SAIDA
        / f"{prefixo}_{classe}_final.png"
    )

    painel_path = (
        SAIDA
        / f"{prefixo}_{classe}_snapshots.png"
    )

    imageio.mimsave(
        gif_path,
        frames,
        duration=0.3,
        loop=0
    )

    Image.fromarray(
        uint8(x[0, 0])
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST
    ).save(imagem_path)

    fig, axs = plt.subplots(
        1,
        len(snapshots),
        figsize=(
            1.8 * len(snapshots),
            2.4
        )
    )

    for ax, (passo, imagem) in zip(
        axs,
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
        f"ControlNet espacial - {classe.upper()}"
    )

    fig.tight_layout()

    fig.savefig(
        painel_path,
        dpi=130
    )

    plt.close(fig)

    print("GIF:", gif_path)
    print("Imagem:", imagem_path)
    print("Snapshots:", painel_path)

    return {
        "classe": classe,
        "gif": str(gif_path),
        "imagem": str(imagem_path),
        "snapshots": str(painel_path)
    }


# ============================================================
# 13 - GERACAO COMPLETA X/Y/Z
# ============================================================

if args.modo == "gerar":

    classes = (
        ("x", "y", "z")
        if args.classe == "todas"
        else (args.classe,)
    )

    saidas = []

    for classe in classes:

        saidas.append(
            gerar(
                classe,
                args.amostras,
                PREFIXO
            )
        )

    destino = (
        SAIDA / f"{PREFIXO}_geracao.json"
    )

    destino.write_text(
        json.dumps(
            {
                "aula": 19,
                "classes": saidas,
                "timesteps": T,
                "guidance_scale": 3.0,
                "tipo_mascara": args.tipo
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nVALIDACAO FINAL - GERACAO ESPACIAL")
    print("Modelo + mascaras treinadas: OK")
    print("DDPM com CFG: OK")
    print("Classes geradas:", len(saidas))
    print("AULA 19 - GERACOES CONCLUIDAS!")

    sys.exit(0)


# ============================================================
# 14 - GIF DE PROGRESSO ENTRE EPOCAS
# ============================================================

if args.modo == "progresso":

    snapshots_pesos = sorted(
        SAIDA.glob(
            f"{PREFIXO}_epoca_*.pth"
        )
    )

    if not snapshots_pesos:
        raise FileNotFoundError(
            "Snapshots por epoca nao encontrados"
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

            label = dataset.label_to_idx[
                classe
            ]

            torch.manual_seed(
                2026 + label
            )

            x = torch.randn(
                1, 1, IMG, IMG,
                device=device
            )

            labels = torch.tensor(
                [label],
                dtype=torch.long,
                device=device
            )

            with torch.inference_mode():

                for t in reversed(range(T)):

                    x = p_sample_condspacial(
                        model,
                        x,
                        t,
                        labels,
                        mascaras[label],
                        guidance_scale=3.0
                    )

            imagens.append(
                uint8(x[0, 0])
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

        ImageDraw.Draw(
            imagem
        ).text(
            (3, 3),
            f"Epoca {epoca} - X Y Z",
            fill=(255, 0, 0)
        )

        frames.append(
            np.asarray(imagem)
        )

        print(
            f"Epoca {epoca}: X/Y/Z OK",
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

    print("GIF progresso:", destino)
    print(
        "AULA 19 - GIF DE TREINAMENTO CONCLUIDO!"
    )
