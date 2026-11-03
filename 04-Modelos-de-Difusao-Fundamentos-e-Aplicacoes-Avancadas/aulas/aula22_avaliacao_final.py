# ============================================================
# PROJETO 04 - AULA 22
# CALCULO DAS METRICAS DE AVALIACAO
# FID / IS / LPIPS / COMPARACAO VISUAL / MLOPS
# ============================================================

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

import argparse
import ast
import hashlib
import math
import importlib.metadata
import json
import os
import random
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent

SAIDA = BASE / "resultados" / "aula22_avaliacao_final"

SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

T = 1000
SEED = 2026

DESCRICOES = {
    "x": "sagital",
    "y": "coronal",
    "z": "axial"
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["teste", "avaliar", "exportar"],
    default="teste"
)

parser.add_argument(
    "--pesos",
    choices=["reduzido", "completo"],
    default="reduzido"
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
    "--sampler",
    choices=["ddim", "ddpm"],
    default="ddim"
)

parser.add_argument(
    "--guidance",
    type=float,
    default=3.0
)

parser.add_argument(
    "--batch",
    type=int,
    default=2
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
    "--forcar",
    action="store_true"
)

args = parser.parse_args()

if (
    args.batch < 1
    or args.batch > 8
    or args.passos < 2
    or args.passos > T
    or args.guidance < 0
):
    parser.error(
        "Parametros invalidos."
    )

if (
    args.sampler == "ddpm"
    and args.passos != T
    and args.modo != "teste"
):
    parser.error(
        "DDPM precisa de 1000 passos."
    )

if args.modo == "avaliar":
    if args.amostras < 30 or args.amostras % 3 != 0:
        parser.error(
            "Use 30, 60, 201 ou outro "
            "multiplo de 3."
        )

n_amostras = (
    6 if args.modo == "teste"
    else args.amostras
)

passos = (
    10 if args.modo == "teste"
    else args.passos
)

sampler = (
    "ddim" if args.modo == "teste"
    else args.sampler
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

print("=" * 65)
print("AULA 22 - AVALIACAO FINAL DE MODELOS DE DIFUSAO")
print("=" * 65)

print("Modo:", args.modo)
print("Dispositivo:", device)
print("Sampler:", sampler)
print("Passos:", passos)
print("Amostras:", n_amostras)


# ============================================================
# 2 - CARREGAR PESOS DA AULA 21
# ============================================================

PESOS = (
    BASE
    / "resultados"
    / f"aula21_{args.pesos}_melhor_modelo.pth"
)

if not PESOS.exists():
    raise FileNotFoundError(
        f"Modelo nao encontrado: {PESOS}"
    )


def sha256(caminho):

    h = hashlib.sha256()

    with caminho.open("rb") as arquivo:

        for bloco in iter(
            lambda: arquivo.read(1024 * 1024),
            b""
        ):
            h.update(bloco)

    return h.hexdigest()


hash_pesos = sha256(PESOS)

base_channels = (
    16 if args.pesos == "reduzido"
    else 64
)

print("\n1 - CARREGANDO MODELO TREINADO")
print("Pesos:", PESOS.name)


# ============================================================
# 3 - IMPORTAR ARQUITETURA DA AULA 15
# ============================================================

origem = (
    BASE / "aulas" / "aula15_unet_condicional.py"
)

arvore = ast.parse(
    origem.read_text(
        encoding="utf-8-sig"
    )
)

definicoes = {
    no.name: no
    for no in arvore.body
    if isinstance(
        no,
        (ast.FunctionDef, ast.ClassDef)
    )
}

nomes = [
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

faltantes = set(nomes) - set(definicoes)

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
        str(origem),
        "exec"
    ),
    globals()
)


# ============================================================
# 4 - DATASET BRAIN MRI
# ============================================================

dataset = BrainMRIConditionalDataset(
    BASE / "dados" / "brain-mri-images",
    transform=transform_mri,
    slice_plane=None
)

assert dataset.label_to_idx == {
    "x": 0,
    "y": 1,
    "z": 2
}

assert len(dataset) == 7692

model = ConditionalUNet(
    in_channels=1,
    base_channels=base_channels,
    num_classes=3,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

assert model.null_label == 3

model.load_state_dict(
    torch.load(
        PESOS,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("\n2 - VALIDACAO DO MODELO")

print("Dataset:", len(dataset))
print("Classes:", dataset.label_to_idx)
print("Descricoes:", DESCRICOES)

print(
    "Parametros:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print("Classe nula CFG:", model.null_label)
print("Modelo treinado: CARREGADO")


# ============================================================
# 5 - SCHEDULER DE DIFUSAO
# ============================================================

betas = torch.linspace(
    1e-4,
    0.02,
    T,
    device=device
)

alphas = 1 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)


# ============================================================
# 6 - INDICES REAIS FORA DO TREINO DA AULA 21
# ============================================================

rng = np.random.default_rng(2026)

train_por_classe = (
    16 if args.pesos == "reduzido"
    else 2564 - 257
)

excluir_treino = set()

for classe in range(3):

    ids = np.asarray([
        i
        for i, label in enumerate(
            dataset.labels
        )
        if label == classe
    ])

    ordem = rng.permutation(
        ids
    ).tolist()

    excluir_treino.update(
        ordem[:train_por_classe]
    )


# ============================================================
# 7 - GERACAO CONDICIONAL DDIM / DDPM
# ============================================================

@torch.inference_mode()
def gerar_condicional(labels, seed_batch):

    torch.manual_seed(seed_batch)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            seed_batch
        )

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

    if sampler == "ddim":

        tempos = (
            torch.linspace(
                T - 1,
                0,
                passos
            )
            .round()
            .long()
            .unique_consecutive()
            .tolist()
        )

    else:

        tempos = list(
            reversed(range(T))
        )

    for k, t in enumerate(tempos):

        ts = torch.full(
            (len(labels),),
            t,
            device=device,
            dtype=torch.long
        )

        pred_cond = model(
            x,
            ts,
            labels
        )

        pred_null = model(
            x,
            ts,
            None
        )

        pred_noise = (
            pred_null
            + args.guidance
            * (pred_cond - pred_null)
        )

        if sampler == "ddim":

            anterior = (
                tempos[k + 1]
                if k + 1 < len(tempos)
                else -1
            )

            a = alpha_bar[t]

            a_prev = (
                alpha_bar[anterior]
                if anterior >= 0
                else x.new_tensor(1.0)
            )

            x0 = (
                (
                    x
                    - (1 - a).sqrt()
                    * pred_noise
                )
                / a.sqrt()
            ).clamp(-1, 1)

            x = (
                a_prev.sqrt() * x0
                + (1 - a_prev).sqrt()
                * pred_noise
            )

        else:

            beta = betas[t]
            alpha = alphas[t]
            ab = alpha_bar[t]

            media = (
                x
                - beta
                / (1 - ab).sqrt()
                * pred_noise
            ) / alpha.sqrt()

            if t > 0:

                ab_prev = alpha_bar[t - 1]

                variancia = (
                    beta
                    * (1 - ab_prev)
                    / (1 - ab)
                )

                x = (
                    media
                    + variancia.clamp(
                        min=0
                    ).sqrt()
                    * torch.randn_like(x)
                )

            else:

                x = media

    return x.clamp(-1, 1).cpu()


# ============================================================
# 8 - GERAR AMOSTRAS PARA AVALIACAO
# ============================================================

def generate_evaluation_samples(num_samples):

    print(
        f"\n3 - GERANDO {num_samples} "
        "IMAGENS SINTETICAS"
    )

    imagens = []
    rotulos = []

    por_classe = num_samples // 3

    for classe in range(3):

        nome = dataset.idx_to_label[classe]

        print(
            f"Classe {classe} ({nome}): "
            f"{por_classe} imagens",
            flush=True
        )

        for inicio in range(
            0,
            por_classe,
            args.batch
        ):

            quantidade = min(
                args.batch,
                por_classe - inicio
            )

            labels = torch.full(
                (quantidade,),
                classe,
                dtype=torch.long
            )

            geradas = gerar_condicional(
                labels,
                40000 + classe * 1000 + inicio
            )

            imagens.append(geradas)

            rotulos.extend(
                [classe] * quantidade
            )

            print(
                f"  {inicio + quantidade}"
                f"/{por_classe}",
                flush=True
            )

    generated_imgs = torch.cat(
        imagens,
        dim=0
    )

    generated_labels = torch.tensor(
        rotulos,
        dtype=torch.long
    )

    print(
        "Amostras geradas:",
        len(generated_imgs)
    )

    print(
        "Shape:",
        tuple(generated_imgs.shape)
    )

    return generated_imgs, generated_labels


# ============================================================
# 9 - COLETAR AMOSTRAS REAIS
# ============================================================

def collect_real_samples(num_samples):

    print(
        f"\n4 - COLETANDO {num_samples} "
        "IMAGENS REAIS"
    )

    imagens = []
    rotulos = []
    indices = []

    rng = np.random.default_rng(3000)

    por_classe = num_samples // 3

    for classe in range(3):

        disponiveis = [
            i
            for i, label in enumerate(
                dataset.labels
            )
            if (
                label == classe
                and i not in excluir_treino
            )
        ]

        if len(disponiveis) < por_classe:

            raise RuntimeError(
                "Imagens reais insuficientes."
            )

        escolhidos = rng.choice(
            disponiveis,
            por_classe,
            replace=False
        ).tolist()

        for i in escolhidos:

            item = dataset[int(i)]

            imagens.append(
                item["image"].unsqueeze(0)
            )

            rotulos.append(classe)
            indices.append(int(i))

        print(
            f"Classe {classe}: "
            f"{por_classe} imagens reais"
        )

    real_imgs = torch.cat(
        imagens,
        dim=0
    )

    real_labels = torch.tensor(
        rotulos,
        dtype=torch.long
    )

    print(
        "Imagens reais:",
        len(real_imgs)
    )

    print(
        "Shape:",
        tuple(real_imgs.shape)
    )

    return real_imgs, real_labels, indices


# ============================================================
# 10 - CONVERTER TENSOR PARA IMAGEM
# ============================================================

def to_img(x):

    x = (
        (
            x[0].float()
            .clamp(-1, 1)
            + 1
        ) / 2 * 255
    )

    return x.to(
        torch.uint8
    ).numpy()


# ============================================================
# 11 - COMPARACAO REAL X SINTETICA
# ============================================================

def salvar_comparacao(real, fake, labels, pasta):

    selecionados = []

    for classe in range(3):

        encontrados = torch.where(
            labels == classe
        )[0]

        selecionados.extend(
            encontrados[:2].tolist()
        )

    if len(selecionados) != 6:

        raise RuntimeError(
            "Necessarias duas imagens por classe."
        )

    fig, axs = plt.subplots(
        2,
        6,
        figsize=(14, 5)
    )

    for coluna, i in enumerate(selecionados):

        classe = int(labels[i])

        nome = DESCRICOES[
            dataset.idx_to_label[classe]
        ]

        axs[0, coluna].imshow(
            to_img(real[i]),
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axs[1, coluna].imshow(
            to_img(fake[i]),
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axs[0, coluna].set_title(
            f"REAL - {nome}"
        )

        axs[1, coluna].set_title(
            f"GERADA - {nome}"
        )

        axs[0, coluna].axis("off")
        axs[1, coluna].axis("off")

    fig.suptitle(
        "Aula 22 - Imagens reais e sinteticas"
    )

    fig.tight_layout()

    comparacao = (
        pasta / "comparacao_real_gerada.png"
    )

    fig.savefig(
        comparacao,
        dpi=150
    )

    plt.close(fig)

    print(
        "Comparacao salva:",
        comparacao
    )

    # Teste cego
    celulas = []

    for i in selecionados:

        classe = dataset.idx_to_label[
            int(labels[i])
        ]

        celulas.append(
            ("real", classe, real[i])
        )

        celulas.append(
            ("gerada", classe, fake[i])
        )

    random.Random(2233).shuffle(
        celulas
    )

    fig, axs = plt.subplots(
        2,
        6,
        figsize=(12, 5)
    )

    gabarito = {}

    for pos, (
        tipo,
        classe,
        imagem
    ) in enumerate(celulas):

        numero = f"{pos + 1:02d}"

        ax = axs[
            pos // 6,
            pos % 6
        ]

        ax.imshow(
            to_img(imagem),
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            f"Imagem {numero}"
        )

        ax.axis("off")

        gabarito[numero] = {
            "origem": tipo,
            "classe": classe,
            "plano": DESCRICOES[classe]
        }

    fig.tight_layout()

    fig.savefig(
        pasta / "teste_cego.png",
        dpi=150
    )

    plt.close(fig)

    (
        pasta / "gabarito_teste_cego.json"
    ).write_text(
        json.dumps(
            gabarito,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Teste cego: SALVO")


# ============================================================
# 12 - PREPARAR IMAGENS PARA AS METRICAS
# ============================================================

def rgb01(x):

    return (
        (
            x.float().clamp(-1, 1)
            + 1
        ) / 2
    ).repeat(1, 3, 1, 1)


def rgb11(x):

    return x.float().clamp(
        -1, 1
    ).repeat(1, 3, 1, 1)


# ============================================================
# 13 - FID / INCEPTION SCORE / LPIPS
# ============================================================


def calcular_metricas(real, fake, labels):

    import gc

    from torchmetrics.image.fid import (
        FrechetInceptionDistance
    )

    from torchmetrics.image.inception import (
        InceptionScore
    )

    from torchmetrics.image.lpip import (
        LearnedPerceptualImagePatchSimilarity
    )

    print("\n5 - AVALIACAO SEQUENCIAL PARA ECONOMIZAR RAM")

    # ==========================================
    # FID
    # ==========================================

    print("\n1/3 - Calculando FID...", flush=True)

    fid = FrechetInceptionDistance(
        feature=args.fid_feature,
        normalize=True
    ).to(device)

    with torch.inference_mode():

        for i in range(0, len(real), 2):

            fid.update(
                rgb01(real[i:i+2].to(device)),
                real=True
            )

            fid.update(
                rgb01(fake[i:i+2].to(device)),
                real=False
            )

        fid_score = float(
            fid.compute().item()
        )

    print("FID:", fid_score, flush=True)

    del fid
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==========================================
    # INCEPTION SCORE
    # ==========================================

    print("\n2/3 - Calculando IS...", flush=True)

    inception = InceptionScore(
        splits=10,
        normalize=True
    ).to(device)

    with torch.inference_mode():

        for i in range(0, len(fake), 2):

            inception.update(
                rgb01(fake[i:i+2].to(device))
            )

        is_mean, is_std = inception.compute()

        is_valor = float(is_mean.item())
        is_desvio = float(is_std.item())

    print("IS:", is_valor, flush=True)

    del inception
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==========================================
    # LPIPS
    # ==========================================

    print(
        "\n3/3 - Calculando LPIPS "
        f"({args.lpips_net})...",
        flush=True
    )

    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=args.lpips_net,
        reduction="none",
        normalize=False
    ).to(device)

    lp_real_fake = []
    lp_diversidade = []

    with torch.inference_mode():

        for classe in range(3):

            selecionados = torch.where(
                labels == classe
            )[0]

            reais_classe = real[
                selecionados
            ]

            falsas_classe = fake[
                selecionados
            ]

            # LPIPS real x gerada nao pareadas
            for i in range(
                0,
                len(selecionados),
                2
            ):

                a = rgb11(
                    reais_classe[i:i+2].to(device)
                )

                b = rgb11(
                    falsas_classe[i:i+2].to(device)
                )

                valores = lpips(a, b)

                lp_real_fake.extend(
                    valores.detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )

            # LPIPS entre imagens sinteticas
            # da mesma classe
            for i in range(
                0,
                len(falsas_classe) - 1,
                2
            ):

                a = falsas_classe[
                    i:min(i+2, len(falsas_classe)-1)
                ]

                b = falsas_classe[
                    i+1:min(i+3, len(falsas_classe))
                ]

                valores = lpips(
                    rgb11(a.to(device)),
                    rgb11(b.to(device))
                )

                lp_diversidade.extend(
                    valores.detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )

    lp_media = float(np.mean(lp_real_fake))
    lp_dp = float(np.std(lp_real_fake))

    lp_div = float(np.mean(lp_diversidade))
    lp_div_dp = float(np.std(lp_diversidade))

    print(
        "LPIPS real/gerada:",
        lp_media,
        flush=True
    )

    print(
        "LPIPS diversidade:",
        lp_div,
        flush=True
    )

    del lpips
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fid": fid_score,
        "fid_feature": args.fid_feature,
        "inception_score_media": is_valor,
        "inception_score_desvio": is_desvio,
        "lpips_real_gerada_nao_pareadas": lp_media,
        "lpips_real_gerada_dp": lp_dp,
        "lpips_diversidade_sintetica": lp_div,
        "lpips_diversidade_dp": lp_div_dp,
        "lpips_net": args.lpips_net
    }


# ============================================================
# 14 - VERSOES DAS BIBLIOTECAS
# ============================================================

def versao(pacote):

    try:

        return importlib.metadata.version(
            pacote
        )

    except importlib.metadata.PackageNotFoundError:

        return None


# ============================================================
# 15 - EXPORTACAO MLOPS
# ============================================================

def exportar_manifesto(pasta, relatorio=None):

    destino = pasta / "exportacao"

    destino.mkdir(
        parents=True,
        exist_ok=True
    )

    modelo_destino = (
        destino / "conditional_unet_best.pth"
    )

    if (
        not modelo_destino.exists()
        or sha256(modelo_destino) != hash_pesos
    ):

        shutil.copy2(
            PESOS,
            modelo_destino
        )

    manifesto = {
        "aula": 22,
        "modelo": "ConditionalUNet",
        "origem_pesos": PESOS.name,
        "sha256_pesos": hash_pesos,
        "arquivo_pesos": modelo_destino.name,
        "base_channels": base_channels,
        "in_channels": 1,
        "num_classes_reais": 3,
        "null_label": 3,
        "label_to_idx": dataset.label_to_idx,
        "class_descriptions": DESCRICOES,
        "img_size": 64,
        "input_range": [-1, 1],
        "diffusion": {
            "T": T,
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "inferencia_avaliada": {
            "sampler": sampler,
            "passos": passos,
            "guidance_scale": args.guidance,
            "n_amostras": n_amostras
        },
        "bibliotecas": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torchmetrics": versao("torchmetrics"),
            "torch-fidelity": versao("torch-fidelity"),
            "Pillow": versao("Pillow")
        },
        "metricas": relatorio,
        "limitacoes": [
            "Modelo educacional, sem validacao clinica.",
            "FID de 64 features nao equivale ao FID2048.",
            "LPIPS nao pareado nao mede reconstrucao.",
            "Imagens podem ter sido vistas nas aulas anteriores.",
            "Modelos perceptuais fora do dominio MRI."
        ]
    }

    (
        destino / "manifesto.json"
    ).write_text(
        json.dumps(
            manifesto,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    (
        destino / "requirements_avaliacao.txt"
    ).write_text(
        "torch\n"
        "numpy\n"
        "Pillow\n"
        "matplotlib\n"
        "torchvision\n"
        "torchmetrics\n"
        "torch-fidelity\n",
        encoding="utf-8"
    )

    print(
        "Modelo e manifesto exportados:",
        destino
    )

    return destino


# ============================================================
# 16 - EXECUCAO PRINCIPAL
# ============================================================

if args.modo == "exportar":

    exportar_manifesto(
        SAIDA
    )

    print(
        "AULA 22 - EXPORTACAO CONCLUIDA!"
    )

else:

    pasta = (
        SAIDA
        / (
            f"{args.modo}_{sampler}{passos}"
            f"_n{n_amostras}"
            f"_g{args.guidance:g}"
        )
    )

    pasta.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_cache = (
        pasta / "amostras.pt"
    )

    config_cache = {
        "pesos_sha256": hash_pesos,
        "n": n_amostras,
        "passos": passos,
        "sampler": sampler,
        "guidance": args.guidance,
        "batch": args.batch
    }

    if (
        arquivo_cache.exists()
        and not args.forcar
    ):

        cache = torch.load(
            arquivo_cache,
            map_location="cpu",
            weights_only=True
        )

        if cache["config"] != config_cache:

            raise RuntimeError(
                "Cache incompativel. "
                "Use --forcar para regenerar."
            )

        real = cache["real"]
        fake = cache["fake"]
        labels = cache["labels"]
        indices = cache["indices"]

        print(
            "Amostras recuperadas do cache."
        )

    else:

        fake, labels_fake = (
            generate_evaluation_samples(
                n_amostras
            )
        )

        real, labels_real, indices = (
            collect_real_samples(
                n_amostras
            )
        )

        if not torch.equal(
            labels_fake,
            labels_real
        ):

            raise RuntimeError(
                "Labels reais e geradas diferentes."
            )

        labels = labels_real

        if not (
            torch.isfinite(real).all()
            and torch.isfinite(fake).all()
        ):

            raise RuntimeError(
                "Pixels nao finitos detectados."
            )

        torch.save(
            {
                "real": real,
                "fake": fake,
                "labels": labels,
                "indices": indices,
                "config": config_cache
            },
            arquivo_cache
        )

        print(
            "Cache salvo:",
            arquivo_cache
        )

    assert real.shape == fake.shape == (
        n_amostras,
        1,
        64,
        64
    )

    print(
        "\nDistribuicao por classe:",
        torch.bincount(
            labels,
            minlength=3
        ).tolist()
    )

    # Comparacao visual
    salvar_comparacao(
        real,
        fake,
        labels,
        pasta
    )

    if args.modo == "avaliar":

        resultados = calcular_metricas(
            real,
            fake,
            labels
        )

        resultados.update(
            {
                "num_amostras_reais": len(real),
                "num_amostras_geradas": len(fake),
                "classes": dataset.label_to_idx,
                "sampling": config_cache,
                "observacao": (
                    "Avaliacao exploratoria. "
                    "Sem validade clinica."
                )
            }
        )

        relatorio = (
            pasta / "metricas.json"
        )

        relatorio.write_text(
            json.dumps(
                resultados,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print("\nRESULTADOS DAS METRICAS")

        print(
            "FID:",
            resultados["fid"]
        )

        print(
            "IS:",
            resultados["inception_score_media"]
        )

        print(
            "LPIPS real/gerada:",
            resultados[
                "lpips_real_gerada_nao_pareadas"
            ]
        )

        print(
            "LPIPS diversidade:",
            resultados[
                "lpips_diversidade_sintetica"
            ]
        )

        print(
            "Relatorio:",
            relatorio
        )

        exportar_manifesto(
            pasta,
            resultados
        )

        print(
            "\nAULA 22 - AVALIACAO CONCLUIDA!"
        )

    else:

        print(
            "\nFID / IS / LPIPS: "
            "NAO calculadas no modo teste"
        )

        print(
            "AULA 22 - TESTE CONCLUIDO!"
        )
