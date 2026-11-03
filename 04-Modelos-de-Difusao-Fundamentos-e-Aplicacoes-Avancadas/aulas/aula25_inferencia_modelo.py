# ============================================================
# PROJETO 04 - AULA 25
# INFERENCIA DO MODELO CONDICIONAL
# ============================================================

import argparse
import hashlib
import importlib.util
import json
import math
import os
import secrets

from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from PIL import Image, ImageDraw


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent

EXPERIMENTO = (
    BASE
    / "resultados"
    / "model_artifacts_conditional_diffusion_20261102_235758_73aa15d9"
)

SAIDA = (
    BASE
    / "resultados"
    / "aula25_inferencia"
    / "generated_images"
)

SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=[
        "teste",
        "interface",
        "todas",
        "multiplas",
        "unica"
    ],
    default="teste"
)

parser.add_argument(
    "--classe",
    type=int,
    default=0
)

parser.add_argument(
    "--quantidade",
    type=int,
    default=5
)

parser.add_argument(
    "--guidance",
    type=float,
    default=3.0
)

parser.add_argument(
    "--seed",
    type=int,
    default=None
)

parser.add_argument(
    "--sampler",
    choices=["ddim", "ddpm"],
    default="ddim"
)

parser.add_argument(
    "--passos",
    type=int,
    default=50
)

args = parser.parse_args()

if not 1 <= args.quantidade <= 20:
    parser.error("Quantidade entre 1 e 20.")

if not math.isfinite(args.guidance) or not 0 <= args.guidance <= 10:
    parser.error("Guidance entre 0 e 10.")

if not 2 <= args.passos <= 1000:
    parser.error("Passos entre 2 e 1000.")

if (
    args.sampler == "ddpm"
    and args.passos != 1000
    and args.modo != "teste"
):
    parser.error(
        "Para DDPM, utilize --passos 1000."
    )


# ============================================================
# 2 - CARREGAR CONFIGURACOES
# ============================================================

def ler_json(caminho):

    return json.loads(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )


def sha256(caminho):

    h = hashlib.sha256()

    with caminho.open("rb") as arquivo:

        for parte in iter(
            lambda: arquivo.read(1024 * 1024),
            b""
        ):
            h.update(parte)

    return h.hexdigest()


config_modelo_path = (
    EXPERIMENTO
    / "configs"
    / "model_config.json"
)

config_dataset_path = (
    EXPERIMENTO
    / "configs"
    / "dataset_config.json"
)

arquitetura_path = (
    EXPERIMENTO
    / "models"
    / "conditional_unet_architecture.py"
)

pesos_path = (
    EXPERIMENTO
    / "models"
    / "model_weights_only.pth"
)

for arquivo in (
    config_modelo_path,
    config_dataset_path,
    arquitetura_path,
    pesos_path
):
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Artefato ausente: {arquivo}"
        )

model_config = ler_json(
    config_modelo_path
)

dataset_config = ler_json(
    config_dataset_path
)

params = model_config[
    "model_params"
]

idx_to_label = {
    int(k): v
    for k, v in dataset_config[
        "idx_to_label"
    ].items()
}

assert idx_to_label == {
    0: "x",
    1: "y",
    2: "z"
}

assert params["num_classes"] == 3
assert params["null_label"] == 3

descricao = {
    "x": "sagital",
    "y": "coronal",
    "z": "axial"
}

resize = dataset_config[
    "preprocessing"
]["resize"]

if isinstance(resize, (list, tuple)):

    if len(resize) != 2:
        raise ValueError(
            "Formato de resize invalido."
        )

    if int(resize[0]) != int(resize[1]):
        raise ValueError(
            "O modelo exige dimensoes iguais."
        )

    IMG_SIZE = int(resize[0])

else:

    IMG_SIZE = int(resize)

assert IMG_SIZE == 64


# ============================================================
# 3 - CARREGAR ARQUITETURA
# ============================================================

print("=" * 60)
print("AULA 25 - INFERENCIA DO MODELO")
print("=" * 60)

spec = importlib.util.spec_from_file_location(
    "conditional_unet_inferencia",
    arquitetura_path
)

modulo = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    modulo
)

campos = [
    "in_channels",
    "base_channels",
    "num_classes",
    "time_dim",
    "label_dim",
    "dropout_prob"
]

kwargs = {
    campo: params[campo]
    for campo in campos
}


# ============================================================
# 4 - RECRIAR MODELO E CARREGAR PESOS
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cpu":

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

model = modulo.ConditionalUNet(
    **kwargs
).to(device)

pesos = torch.load(
    pesos_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    pesos,
    strict=True
)

model.eval()

assert model.null_label == 3

print("\n1 - MODELO CARREGADO")

print("Dispositivo:", device)
print("Resolucao:", IMG_SIZE)
print("Classes:", idx_to_label)

print(
    "Parametros:",
    sum(p.numel() for p in model.parameters())
)

print(
    "Classe nula CFG:",
    model.null_label
)

print("Pesos: CARREGADOS")


# ============================================================
# 5 - SCHEDULER DE DIFUSAO
# ============================================================

treino_cfg = model_config[
    "training_params"
]

T = int(
    treino_cfg["diffusion_steps"]
)

beta_start = float(
    treino_cfg["beta_start"]
)

beta_end = float(
    treino_cfg["beta_end"]
)

betas = torch.linspace(
    beta_start,
    beta_end,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)


# ============================================================
# 6 - TESTAR FORWARD
# ============================================================

with torch.inference_mode():

    teste = model(
        torch.zeros(
            1, 1, IMG_SIZE, IMG_SIZE,
            device=device
        ),
        torch.tensor(
            [0],
            dtype=torch.long,
            device=device
        ),
        torch.tensor(
            [0],
            dtype=torch.long,
            device=device
        )
    )

assert teste.shape == (
    1, 1, IMG_SIZE, IMG_SIZE
)

assert bool(
    torch.isfinite(teste).all()
)

print("Forward: OK")


# ============================================================
# 7 - INFERENCIA DDIM / DDPM COM CFG
# ============================================================

@torch.inference_mode()
def gerar_tensor(
    class_idx,
    guidance_scale=3.0,
    seed=None,
    passos=50,
    sampler="ddim"
):

    if class_idx not in idx_to_label:
        raise ValueError(
            "Classe deve ser 0, 1 ou 2."
        )

    if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= 10:
        raise ValueError(
            "Guidance fora do intervalo."
        )

    if not 2 <= passos <= T:
        raise ValueError(
            "Numero de passos invalido."
        )

    if sampler == "ddpm" and passos != T:
        raise ValueError(
            "DDPM exige 1000 passos."
        )

    if seed is None:

        seed = secrets.randbelow(
            2**31
        )

    seed = int(seed)

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(
        1, 1,
        IMG_SIZE,
        IMG_SIZE,
        device=device
    )

    labels = torch.tensor(
        [class_idx],
        dtype=torch.long,
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

    print(
        f"Classe {class_idx}"
        f" | {sampler.upper()}"
        f" | passos: {len(tempos)}"
        f" | seed: {seed}",
        flush=True
    )

    for i, t in enumerate(tempos):

        tt = torch.full(
            (1,),
            t,
            device=device,
            dtype=torch.long
        )

        ruido_cond = model(
            x,
            tt,
            labels
        )

        ruido_null = model(
            x,
            tt,
            None
        )

        ruido = (
            ruido_null
            + guidance_scale
            * (ruido_cond - ruido_null)
        )

        if sampler == "ddim":

            t_prev = (
                tempos[i + 1]
                if i + 1 < len(tempos)
                else -1
            )

            ab = alpha_bar[t]

            ab_prev = (
                alpha_bar[t_prev]
                if t_prev >= 0
                else x.new_tensor(1.0)
            )

            x0 = (
                (
                    x
                    - (1 - ab).sqrt() * ruido
                )
                / ab.sqrt()
            ).clamp(-1, 1)

            x = (
                ab_prev.sqrt() * x0
                + (1 - ab_prev).sqrt()
                * ruido
            )

        else:

            beta = betas[t]
            alpha = alphas[t]
            ab = alpha_bar[t]

            media = (
                x
                - beta
                / (1 - ab).sqrt()
                * ruido
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

        intervalo = max(
            1,
            len(tempos) // 5
        )

        if (
            (i + 1) % intervalo == 0
            or i == len(tempos) - 1
        ):

            print(
                f"  Progresso: "
                f"{i + 1}/{len(tempos)}",
                flush=True
            )

    x = x.clamp(-1, 1)

    img = (
        (x[0, 0] + 1)
        / 2 * 255
    ).round().to(
        torch.uint8
    ).cpu().numpy()

    return img, seed


# ============================================================
# 8 - SALVAR IMAGEM E METADADOS
# ============================================================

def generate_and_save_image(
    class_idx,
    guidance_scale=3.0,
    seed=None,
    passos=50,
    sampler="ddim"
):

    imagem, seed_utilizada = gerar_tensor(
        class_idx,
        guidance_scale,
        seed,
        passos,
        sampler
    )

    classe = idx_to_label[
        class_idx
    ]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    nome = (
        f"{classe}_{timestamp}"
        f"_seed{seed_utilizada}"
        f"_{sampler}{passos}.png"
    )

    caminho = SAIDA / nome

    Image.fromarray(
        imagem,
        mode="L"
    ).save(caminho)

    metadados = {
        "experimento": EXPERIMENTO.name,
        "classe_id": class_idx,
        "classe": classe,
        "plano": descricao[classe],
        "seed": seed_utilizada,
        "guidance_scale": guidance_scale,
        "sampler": sampler,
        "passos": passos,
        "peso_sha256": sha256(pesos_path),
        "data": datetime.now().isoformat(),
        "uso": (
            "Imagem sintetica educacional. "
            "Sem validacao clinica."
        )
    }

    caminho.with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            metadados,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "Imagem salva:",
        caminho
    )

    return caminho, imagem


# ============================================================
# 9 - CRIAR PAINEL VISUAL
# ============================================================

def salvar_painel(
    imagens,
    classes
):

    if not imagens:
        return

    cols = min(
        len(imagens),
        5
    )

    rows = math.ceil(
        len(imagens) / cols
    )

    largura = 256
    altura = 286

    painel = Image.new(
        "RGB",
        (cols * largura, rows * altura),
        "white"
    )

    draw = ImageDraw.Draw(
        painel
    )

    for i, imagem in enumerate(imagens):

        x = (i % cols) * largura
        y = (i // cols) * altura

        img = Image.fromarray(
            imagem
        ).resize(
            (256, 256)
        ).convert("RGB")

        painel.paste(
            img,
            (x, y + 30)
        )

        draw.text(
            (x + 8, y + 8),
            f"{i+1} - {classes[i]}",
            fill="black"
        )

    destino = (
        SAIDA
        / (
            "painel_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".png"
        )
    )

    painel.save(
        destino
    )

    print(
        "Painel salvo:",
        destino
    )


# ============================================================
# 10 - UMA IMAGEM POR CLASSE
# ============================================================

def generate_all_classes(
    guidance=3.0,
    passos=50,
    sampler="ddim"
):

    imagens = []
    classes = []

    for classe in range(3):

        _, img = generate_and_save_image(
            classe,
            guidance,
            seed=42 + classe,
            passos=passos,
            sampler=sampler
        )

        imagens.append(img)

        classes.append(
            descricao[idx_to_label[classe]]
        )

    salvar_painel(
        imagens,
        classes
    )

    print(
        "Tres classes geradas com sucesso!"
    )


# ============================================================
# 11 - MULTIPLAS IMAGENS
# ============================================================

def generate_multiple_images(
    class_idx,
    quantidade=5,
    guidance=3.0,
    passos=50,
    sampler="ddim",
    seed_inicial=100
):

    if class_idx not in idx_to_label:
        raise ValueError(
            "Classe invalida."
        )

    if not 1 <= quantidade <= 20:
        raise ValueError(
            "Quantidade invalida."
        )

    imagens = []
    classes = []

    for i in range(quantidade):

        _, img = generate_and_save_image(
            class_idx,
            guidance,
            seed=seed_inicial + i,
            passos=passos,
            sampler=sampler
        )

        imagens.append(img)

        classes.append(
            idx_to_label[class_idx]
        )

    salvar_painel(
        imagens,
        classes
    )

    print(
        f"{quantidade} imagens geradas!"
    )


# ============================================================
# 12 - INTERFACE INTERATIVA
# ============================================================

def interface():

    print("\n" + "=" * 45)
    print("INTERFACE DE INFERENCIA - AULA 25")
    print("=" * 45)

    print("\nClasses disponiveis:")

    for i, classe in idx_to_label.items():

        print(
            f"{i} - {classe.upper()} "
            f"({descricao[classe]})"
        )

    print(
        "\nSampler:",
        args.sampler,
        "| Passos:",
        args.passos
    )

    while True:

        print("\n" + "-" * 45)

        print(
            "1 - Gerar uma imagem de cada classe"
        )

        print(
            "2 - Gerar multiplas imagens de uma classe"
        )

        print(
            "3 - Gerar uma imagem especifica"
        )

        print(
            "4 - Sair"
        )

        try:

            opcao = input(
                "\nEscolha (1-4): "
            ).strip()

            if opcao == "4":

                print(
                    "Interface encerrada."
                )

                break

            if opcao not in (
                "1", "2", "3"
            ):

                print(
                    "Opcao invalida."
                )

                continue

            guidance = float(
                input(
                    "Guidance scale (padrao 3.0): "
                ).strip()
                or "3.0"
            )

            if not math.isfinite(guidance) or not 0 <= guidance <= 10:
                print("Guidance invalido.")
                continue

            if opcao == "1":

                generate_all_classes(
                    guidance,
                    args.passos,
                    args.sampler
                )

                continue

            classe = int(
                input(
                    "Classe (0, 1 ou 2): "
                ).strip()
            )

            if classe not in idx_to_label:

                print(
                    "Classe invalida."
                )

                continue

            if opcao == "2":

                quantidade = int(
                    input(
                        "Quantidade (padrao 5): "
                    ).strip()
                    or "5"
                )

                seed_inicial = int(
                    input(
                        "Seed inicial (padrao 100): "
                    ).strip()
                    or "100"
                )

                generate_multiple_images(
                    classe,
                    quantidade,
                    guidance,
                    args.passos,
                    args.sampler,
                    seed_inicial
                )

            if opcao == "3":

                texto_seed = input(
                    "Seed (Enter = aleatoria): "
                ).strip()

                seed = (
                    int(texto_seed)
                    if texto_seed
                    else None
                )

                generate_and_save_image(
                    classe,
                    guidance,
                    seed,
                    args.passos,
                    args.sampler
                )

        except (
            ValueError,
            OverflowError
        ) as erro:

            print(
                "Entrada invalida:",
                erro
            )

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nInterface encerrada."
            )

            break


# ============================================================
# 13 - EXECUCAO PRINCIPAL
# ============================================================

if args.modo == "teste":

    print(
        "\n2 - TESTE DE INFERENCIA"
    )

    caminho, imagem = generate_and_save_image(
        class_idx=0,
        guidance_scale=3.0,
        seed=42,
        passos=10,
        sampler="ddim"
    )

    assert caminho.exists()
    assert imagem.shape == (64, 64)

    print("\n" + "=" * 60)
    print("VALIDACAO FINAL - AULA 25")
    print("=" * 60)

    print("Modelo carregado: OK")
    print("Forward: OK")
    print("CFG: OK")
    print("DDIM: OK")
    print("Imagem PNG: OK")
    print("Metadados JSON: OK")

    print(
        "\nAULA 25 - TESTE CONCLUIDO!"
    )

elif args.modo == "interface":

    interface()

elif args.modo == "todas":

    generate_all_classes(
        args.guidance,
        args.passos,
        args.sampler
    )

elif args.modo == "multiplas":

    generate_multiple_images(
        args.classe,
        args.quantidade,
        args.guidance,
        args.passos,
        args.sampler,
        args.seed if args.seed is not None else 100
    )

elif args.modo == "unica":

    generate_and_save_image(
        args.classe,
        args.guidance,
        args.seed,
        args.passos,
        args.sampler
    )
