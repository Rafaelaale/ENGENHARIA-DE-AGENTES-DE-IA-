# ============================================================
# PROJETO 04 - AULA 14
# TREINAMENTO RESIDUAL U-NET / BRAIN MRI
# ============================================================

from pathlib import Path
from collections import defaultdict
import argparse
import ast
import csv
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

from PIL import Image, ImageDraw
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "brain-mri-images"
SAIDA = BASE / "resultados"

SAIDA.mkdir(parents=True, exist_ok=True)

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp"}

img_size = 64
T = 1000
SEED = 42

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=[
        "teste",
        "reduzido",
        "completo",
        "gerar"
    ],
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
    default=1
)

parser.add_argument(
    "--checkpoint-cada",
    type=int,
    default=50
)

args = parser.parse_args()


# ============================================================
# 2 - IMPORTAR ARQUITETURA DA AULA 12
# ============================================================

def importar_definicoes(arquivo, nomes):

    arvore = ast.parse(
        arquivo.read_text(encoding="utf-8-sig")
    )

    definicoes = {
        n.name: n
        for n in arvore.body
        if isinstance(
            n,
            (ast.FunctionDef, ast.ClassDef)
        )
    }

    faltas = [
        nome for nome in nomes
        if nome not in definicoes
    ]

    if faltas:
        raise RuntimeError(
            f"Definicoes ausentes: {faltas}"
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
            str(arquivo),
            "exec"
        ),
        globals()
    )


print("=" * 65)
print("AULA 14 - TREINAMENTO RESIDUAL U-NET / MRI")
print("=" * 65)

print("Modo:", args.modo)
print("Dispositivo:", device)

importar_definicoes(
    BASE / "aulas" / "aula12_unet_mri_arquitetura.py",
    [
        "BrainMRIDataset",
        "transform_mri",
        "sinusoidal_embedding",
        "ResidualTimeBlock",
        "Down",
        "Up",
        "UNet"
    ]
)

model = UNet().to(device)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("U-Net carregada: OK")
print("Parametros:", parametros)


# ============================================================
# 3 - SCHEDULER LINEAR
# ============================================================

betas = torch.linspace(
    0.001,
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
        torch.sqrt(ab) * x0
        + torch.sqrt(1 - ab) * noise
    )


@torch.no_grad()
def p_sample_ddpm(modelo, x, t):

    tempos = torch.full(
        (x.shape[0],),
        t,
        device=x.device,
        dtype=torch.long
    )

    pred = modelo(x, tempos)

    media = (
        x
        - betas[t]
        / torch.sqrt(1 - alpha_bar[t])
        * pred
    ) / torch.sqrt(alphas[t])

    if t == 0:
        return media

    variancia = (
        betas[t]
        * (1 - alpha_bar[t - 1])
        / (1 - alpha_bar[t])
    )

    return (
        media
        + torch.sqrt(variancia)
        * torch.randn_like(x)
    )


# ============================================================
# 4 - FUNCOES DE VISUALIZACAO
# ============================================================

def imagem_uint8(tensor):

    return (
        tensor.detach()
        .cpu()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .to(torch.uint8)
        .squeeze(0)
        .numpy()
    )


def salvar_curva(historico, modo):

    csv_path = (
        SAIDA / f"aula14_{modo}_loss.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as arquivo:

        writer = csv.writer(arquivo)

        writer.writerow([
            "epoca",
            "loss_medio",
            "microbatches"
        ])

        for registro in historico:

            writer.writerow([
                registro["epoca"],
                registro["loss"],
                registro["microbatches"]
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
    ax.set_ylabel("MSE")
    ax.set_title(
        f"Treinamento U-Net MRI - {modo}"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        SAIDA / f"aula14_{modo}_loss.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 5 - TREINAMENTO
# ============================================================

if args.modo != "gerar":

    configuracoes = {
        "teste": {
            "epochs": 1,
            "images": 2,
            "micro": 1,
            "accum": 1
        },
        "reduzido": {
            "epochs": 3,
            "images": 24,
            "micro": 2,
            "accum": 2
        },
        "completo": {
            "epochs": 150,
            "images": None,
            "micro": 2,
            "accum": 64
        }
    }

    config = configuracoes[args.modo]

    print("\n1 - CARREGANDO DATASET")

    dataset = BrainMRIDataset(
        DADOS,
        transform=transform_mri,
        slice_plane="z"
    )

    assert len(dataset) > 0

    n = config["images"] or len(dataset)

    print("Imagens do plano Z:", len(dataset))
    print("Imagens usadas:", n)
    print("Epocas:", config["epochs"])
    print("Microbatch:", config["micro"])
    print("Acumulacao:", config["accum"])
    print(
        "Batch efetivo:",
        config["micro"] * config["accum"]
    )

    gerador = torch.Generator()
    gerador.manual_seed(2026)

    indices = torch.randperm(
        len(dataset),
        generator=gerador
    ).tolist()[:n]

    print("\n2 - CONFIGURANDO OTIMIZADOR")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    print("AdamW: OK")
    print("Learning rate: 0.0001")
    print(
        "Mixed precision CUDA:",
        scaler.is_enabled()
    )

    checkpoint_path = (
        SAIDA
        / f"aula14_{args.modo}_checkpoint.pth"
    )

    pesos_path = (
        SAIDA
        / f"aula14_{args.modo}_modelo_treinado.pth"
    )

    historico = []

    progresso = {
        "epoca": 0,
        "batch": 0,
        "soma": 0.0,
        "quantidade": 0
    }

    if checkpoint_path.exists():

        print("\nCheckpoint encontrado.")

        dados = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        if (
            dados["config"] != config
            or dados["n"] != n
        ):
            raise RuntimeError(
                "Checkpoint incompativel."
            )

        model.load_state_dict(
            dados["model"]
        )

        opt.load_state_dict(
            dados["optimizer"]
        )

        scaler.load_state_dict(
            dados["scaler"]
        )

        historico = dados["history"]
        progresso = dados["progress"]

        print(
            "Retomando epoca:",
            progresso["epoca"] + 1
        )

        print(
            "Proximo microbatch:",
            progresso["batch"]
        )

    def salvar_checkpoint():

        conteudo = {
            "config": config,
            "n": n,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "history": historico,
            "progress": progresso.copy()
        }

        temporario = checkpoint_path.with_suffix(
            ".tmp"
        )

        torch.save(
            conteudo,
            temporario
        )

        os.replace(
            temporario,
            checkpoint_path
        )

    print("\n3 - INICIANDO TREINAMENTO")

    model.train()

    for epoca in range(
        progresso["epoca"],
        config["epochs"]
    ):

        permutacao = torch.randperm(
            n,
            generator=torch.Generator().manual_seed(
                SEED + epoca
            )
        ).tolist()

        ordem = [
            indices[i]
            for i in permutacao
        ]

        total = math.ceil(
            n / config["micro"]
        )

        if epoca == progresso["epoca"]:

            inicio = progresso["batch"]
            soma = progresso["soma"]
            quantidade = progresso["quantidade"]

        else:

            inicio = 0
            soma = 0.0
            quantidade = 0

        restantes = ordem[
            inicio * config["micro"]:
        ]

        loader = DataLoader(
            Subset(dataset, restantes),
            batch_size=config["micro"],
            shuffle=False,
            num_workers=0
        )

        opt.zero_grad(
            set_to_none=True
        )

        print(
            f"\nEpoca {epoca + 1}/"
            f"{config['epochs']} "
            f"| microbatches: {total}",
            flush=True
        )

        for offset, batch in enumerate(loader):

            indice = inicio + offset

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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(device.type == "cuda")
            ):

                pred = model(xt, t)

                loss = F.mse_loss(
                    pred.float(),
                    noise.float()
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Loss nao finito."
                )

            grupo_inicio = (
                indice // config["accum"]
            ) * config["accum"]

            grupo_tamanho = min(
                config["accum"],
                total - grupo_inicio
            )

            scaler.scale(
                loss / grupo_tamanho
            ).backward()

            soma += float(loss.item())
            quantidade += 1

            fim_grupo = (
                (indice + 1) % config["accum"] == 0
                or indice + 1 == total
            )

            if fim_grupo:

                # Desescalar antes do clipping.
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

                progresso = {
                    "epoca": epoca,
                    "batch": indice + 1,
                    "soma": soma,
                    "quantidade": quantidade
                }

                if (
                    args.modo == "teste"
                    or (
                        (indice + 1)
                        % max(1, args.checkpoint_cada)
                        == 0
                    )
                ):

                    salvar_checkpoint()

                    print(
                        f"Epoca {epoca + 1}/"
                        f"{config['epochs']} "
                        f"| batch {indice + 1}/{total} "
                        f"| MSE {soma / quantidade:.6f} "
                        f"| checkpoint OK",
                        flush=True
                    )

        media = soma / max(
            1,
            quantidade
        )

        historico.append({
            "epoca": epoca + 1,
            "loss": media,
            "microbatches": quantidade
        })

        progresso = {
            "epoca": epoca + 1,
            "batch": 0,
            "soma": 0.0,
            "quantidade": 0
        }

        salvar_checkpoint()

        salvar_curva(
            historico,
            args.modo
        )

        print(
            f"Epoca {epoca + 1}/"
            f"{config['epochs']} "
            f"- loss medio: {media:.6f} "
            f"| checkpoint OK",
            flush=True
        )

    torch.save(
        model.state_dict(),
        pesos_path
    )

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - TREINAMENTO")
    print("=" * 65)

    assert len(historico) == config["epochs"]
    assert checkpoint_path.exists()
    assert pesos_path.exists()

    print("Mini ou U-Net residual: U-Net residual")
    print("Dataset MRI: OK")
    print("q_sample: OK")
    print("Predicao de ruido: OK")
    print("MSE: OK")
    print("Backward: OK")
    print("AdamW: OK")
    print("Gradient clipping: OK")
    print("Checkpoints: OK")
    print("Pesos salvos: OK")
    print("Epocas concluidas:", len(historico))

    print(
        "\nGERACAO DE IMAGENS: EXECUCAO SEPARADA"
    )

    print(
        f"AULA 14 - {args.modo.upper()} CONCLUIDO!"
    )


# ============================================================
# 6 - GERACAO DDPM E FEATURE MAPS
# ============================================================

else:

    if not 1 <= args.amostras <= 6:
        raise ValueError(
            "Use --amostras entre 1 e 6."
        )

    pesos_path = (
        SAIDA
        / f"aula14_{args.pesos}_modelo_treinado.pth"
    )

    if not pesos_path.exists():
        raise FileNotFoundError(
            f"Pesos treinados nao encontrados: "
            f"{pesos_path}"
        )

    model.load_state_dict(
        torch.load(
            pesos_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    nome_saida = (
        f"aula14_{args.pesos}_"
        f"{args.amostras}imagens"
    )

    print("\n1 - GERACAO DDPM")
    print("Timesteps:", T)
    print("Amostras:", args.amostras)

    camadas = [
        "down1.block1.conv1",
        "bot1.conv1",
        "out"
    ]

    ativacoes = defaultdict(dict)

    handles = []

    tempo_atual = [-1]

    for nome, modulo in model.named_modules():

        if nome in camadas:

            def hook(
                m,
                entradas,
                saida,
                nome=nome
            ):

                if tempo_atual[0] in (
                    900, 500, 100, 0
                ):

                    ativacoes[
                        tempo_atual[0]
                    ][nome] = (
                        saida[:1]
                        .detach()
                        .float()
                        .cpu()
                    )

            handles.append(
                modulo.register_forward_hook(
                    hook
                )
            )

    x = torch.randn(
        args.amostras,
        1,
        64,
        64,
        device=device
    )

    frames = []
    passos = []

    def montar_quadro(x, t):

        imagens = [
            imagem_uint8(imagem)
            for imagem in x
        ]

        matriz = np.concatenate(
            imagens,
            axis=1
        )

        imagem = Image.fromarray(
            matriz,
            mode="L"
        ).resize(
            (
                matriz.shape[1] * 2,
                matriz.shape[0] * 2
            ),
            Image.Resampling.NEAREST
        ).convert("RGB")

        ImageDraw.Draw(imagem).text(
            (3, 3),
            f"t={t}",
            fill="red"
        )

        return np.asarray(imagem)

    frames.append(
        montar_quadro(x, T)
    )

    passos.append(T)

    try:

        with torch.inference_mode():

            for t in reversed(range(T)):

                tempo_atual[0] = t

                x = p_sample_ddpm(
                    model,
                    x,
                    t
                )

                if t % 50 == 0 or t < 5:

                    frames.append(
                        montar_quadro(x, t)
                    )

                    passos.append(t)

                    print(
                        f"Reverso t={t}: OK",
                        flush=True
                    )

    finally:

        for handle in handles:
            handle.remove()

    print("\n2 - SALVANDO GIF")

    arquivo_gif = (
        SAIDA
        / f"{nome_saida}_reverso.gif"
    )

    imageio.mimsave(
        arquivo_gif,
        frames,
        duration=0.35,
        loop=0
    )

    arquivo_final = (
        SAIDA
        / f"{nome_saida}_final.png"
    )

    Image.fromarray(
        imagem_uint8(x[0]),
        mode="L"
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST
    ).save(arquivo_final)

    print("GIF:", arquivo_gif)
    print("Imagem final:", arquivo_final)

    print("\n3 - SALVANDO FEATURE MAPS")

    total_paineis = 0

    for t, registros in ativacoes.items():

        for camada, fmap in registros.items():

            canais = min(
                20,
                fmap.shape[1]
            )

            fig, axs = plt.subplots(
                1,
                canais,
                figsize=(
                    max(3, canais * 1.5),
                    2.4
                ),
                squeeze=False
            )

            for i, ax in enumerate(axs[0]):

                ax.imshow(
                    fmap[0, i].numpy(),
                    cmap="inferno"
                )

                ax.axis("off")

            fig.suptitle(
                f"{camada} | t={t}"
            )

            fig.tight_layout()

            arquivo_feature = (
                SAIDA
                / f"{nome_saida}_feature_"
                  f"{camada.replace('.', '_')}"
                  f"_t{t}.png"
            )

            fig.savefig(
                arquivo_feature,
                dpi=110
            )

            plt.close(fig)

            total_paineis += 1

    print(
        "Paineis de feature maps:",
        total_paineis
    )

    arquivo_json = (
        SAIDA
        / f"{nome_saida}_geracao.json"
    )

    arquivo_json.write_text(
        json.dumps(
            {
                "aula": 14,
                "timesteps": T,
                "passos_registrados": passos,
                "tipo_pesos": args.pesos,
                "amostras": args.amostras,
                "pesos": str(pesos_path),
                "feature_steps": sorted(
                    ativacoes.keys(),
                    reverse=True
                ),
                "paineis": total_paineis
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    assert arquivo_gif.exists()
    assert arquivo_final.exists()
    assert arquivo_json.exists()

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - GERACAO")
    print("=" * 65)

    print("Modelo treinado carregado: OK")
    print("Reverse DDPM 1000 passos: OK")
    print("GIF: OK")
    print("Imagem final: OK")
    print("Feature maps:", total_paineis)
    print("JSON: OK")

    print(
        "\nAULA 14 - GERACAO CONCLUIDA!"
    )
