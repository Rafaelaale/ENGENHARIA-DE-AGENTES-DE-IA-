# ==========================================================
# PROJETO 04 - AULA 08
# SCHEDULER E FORWARD PROCESS - PIXELART
# ==========================================================

from pathlib import Path
import csv
import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1 - CONFIGURACOES
# ==========================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "pixel-art"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

steps = [0, 100, 250, 400, 600, 700, 999]

print("=" * 65)
print("AULA 08 - SCHEDULER E FORWARD PROCESS")
print("=" * 65)
print("PyTorch:", torch.__version__)
print("Dispositivo:", device)


# ==========================================================
# 2 - BETA SCHEDULER
# ==========================================================

def beta_schedule(timesteps: int, schedule="cosine"):

    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)

    elif schedule == "cosine":

        s = 0.008

        values = torch.arange(
            timesteps + 1,
            dtype=torch.float32
        )

        alphas_cumprod = torch.cos(
            ((values / timesteps) + s)
            / (1 + s)
            * math.pi
            * 0.5
        ) ** 2

        alphas_cumprod = (
            alphas_cumprod / alphas_cumprod[0].clone()
        )

        betas = 1 - (
            alphas_cumprod[1:] / alphas_cumprod[:-1]
        )

        return torch.clip(betas, 0.0001, 0.999)

    else:
        raise ValueError("Scheduler desconhecido")


# ==========================================================
# 3 - EXTRACT
# ==========================================================

def extract(a, t, x_shape):

    out = a.gather(0, t.long()).float()

    return out.view(
        -1,
        *((1,) * (len(x_shape) - 1))
    )


# ==========================================================
# 4 - DIFFUSION FORWARD
# ==========================================================

class DiffusionForward:

    def __init__(
        self,
        timesteps=1000,
        schedule="cosine",
        device="cpu"
    ):

        self.device = device
        self.timesteps = timesteps

        betas = beta_schedule(
            timesteps,
            schedule
        ).to(device)

        self.register(betas)

    def register(self, betas):

        self.betas = betas
        self.alphas = 1.0 - betas

        self.alphas_cumprod = torch.cumprod(
            self.alphas,
            dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(
            self.alphas_cumprod
        )

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1 - self.alphas_cumprod
        )

    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_a = extract(
            self.sqrt_alphas_cumprod,
            t,
            x0.shape
        )

        sqrt_m1_a = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x0.shape
        )

        xt = (
            sqrt_a * x0
            + sqrt_m1_a * noise
        )

        return xt, noise


# ==========================================================
# 5 - DATASET PIXELART
# ==========================================================

class PixelArtDataset(Dataset):

    def __init__(self, pasta, csv_labels):

        extensoes = {
            ".png", ".jpg", ".jpeg", ".bmp"
        }

        self.image_files = sorted(
            p for p in pasta.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensoes
        )

        self.captions = {}

        df = pd.read_csv(csv_labels)

        for path, label in zip(
            df["Image Path"],
            df["Label"]
        ):

            nome = (
                Path(str(path).replace("\\", "/"))
                .stem.lower()
                + ".png"
            )

            texto = re.sub(
                r"[\[\],]",
                " ",
                str(label)
            )

            valores = np.fromstring(
                texto,
                sep=" "
            )

            if len(valores) == 5:
                self.captions[nome] = (
                    valores.astype(np.float32).tolist()
                )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        caminho = self.image_files[idx]

        with Image.open(caminho) as arq:

            image = arq.convert("RGB").resize(
                (16, 16),
                Image.Resampling.BILINEAR
            )

        data = (
            np.asarray(image, dtype=np.float32)
            .copy() / 255.0
        )

        tensor = (
            torch.from_numpy(data)
            .permute(2, 0, 1)
        )

        # Normalizacao [-1, 1]
        tensor = tensor * 2.0 - 1.0

        nome = caminho.stem.lower() + ".png"

        label = self.captions.get(
            nome,
            [0.0] * 5
        )

        return {
            "image": tensor,
            "label": torch.tensor(
                label,
                dtype=torch.float32
            )
        }


# ==========================================================
# 6 - VISUALIZAR IMAGENS
# ==========================================================

def plot_grid_normalizado(
    imagens,
    tempos,
    indice,
    label,
    scheduler
):

    fig, axes = plt.subplots(
        1,
        len(tempos),
        figsize=(17, 3)
    )

    for ax, imagem, t in zip(
        axes,
        imagens,
        tempos
    ):

        ax.imshow(
            imagem,
            interpolation="nearest"
        )

        ax.set_title(f"t={t}")
        ax.axis("off")

    fig.suptitle(
        f"{scheduler.upper()} - amostra #{indice}"
    )

    fig.tight_layout()

    arquivo = RESULTADOS / (
        f"aula08_{scheduler}_amostra_{indice}.png"
    )

    fig.savefig(
        arquivo,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close(fig)

    return arquivo


# ==========================================================
# 7 - HISTOGRAMAS
# ==========================================================

def plot_histogramas_paralelos(
    imgs_raw,
    imgs_vis,
    tempos,
    indice
):

    fig, axes = plt.subplots(
        2,
        len(tempos),
        figsize=(21, 7)
    )

    for i, t in enumerate(tempos):

        for linha, valores, cor, nome in [
            (0, imgs_raw[i].flatten(),
             "lightcoral", "RAW"),
            (1, imgs_vis[i].flatten(),
             "skyblue", "NORM")
        ]:

            ax = axes[linha, i]

            ax.hist(
                valores,
                bins=50,
                density=True,
                color=cor,
                edgecolor="black",
                alpha=0.7
            )

            mu, std = norm.fit(valores)

            xs = np.linspace(
                float(valores.min()),
                float(valores.max()),
                100
            )

            if std > 1e-9:

                ax.plot(
                    xs,
                    norm.pdf(xs, mu, std),
                    color="black" if linha == 0 else "red",
                    linewidth=2
                )

            ax.set_title(
                f"{nome} t={t}\n"
                f"media={mu:.2f}, std={std:.2f}"
            )

            ax.grid(alpha=0.2)

    fig.suptitle(
        f"Histogramas - amostra #{indice}"
    )

    fig.tight_layout()

    arquivo = RESULTADOS / (
        f"aula08_linear_histogramas_{indice}.png"
    )

    fig.savefig(
        arquivo,
        dpi=115,
        bbox_inches="tight"
    )

    plt.close(fig)

    return arquivo


# ==========================================================
# 8 - MEDIA E DESVIO PADRAO
# ==========================================================

def plot_evolucao_mean_std(
    tempos,
    medias,
    desvios,
    indice
):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        tempos,
        medias,
        marker="o",
        label="Media"
    )

    ax.plot(
        tempos,
        desvios,
        marker="s",
        label="Desvio padrao"
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Valor")

    ax.set_title(
        f"Evolucao da media e desvio - #{indice}"
    )

    ax.grid(alpha=0.4)
    ax.legend()

    fig.tight_layout()

    arquivo = RESULTADOS / (
        f"aula08_linear_media_desvio_{indice}.png"
    )

    fig.savefig(
        arquivo,
        dpi=140
    )

    plt.close(fig)

    return arquivo


# ==========================================================
# 9 - EXECUTAR FORWARD
# ==========================================================

def plot_list(
    dataset,
    indices,
    pipe,
    scheduler,
    graphs=False
):

    arquivos = []
    estatisticas = []

    for numero, idx in enumerate(
        indices,
        start=1
    ):

        sample = dataset[idx]

        x0 = (
            sample["image"]
            .unsqueeze(0)
            .to(device)
        )

        imgs_vis = []
        imgs_raw = []
        medias = []
        desvios = []

        for t in steps:

            t_tensor = torch.tensor(
                [t],
                dtype=torch.long,
                device=device
            )

            # Mesmo ruido nos dois schedulers
            gerador = torch.Generator().manual_seed(
                10000 + idx * 1000 + t
            )

            ruido = torch.randn(
                x0.shape,
                generator=gerador,
                dtype=x0.dtype
            ).to(device)

            xt, _ = pipe.q_sample(
                x0,
                t_tensor,
                noise=ruido
            )

            raw = (
                xt.detach()
                .cpu()
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )

            vis = np.clip(
                raw * 0.5 + 0.5,
                0.0,
                1.0
            )

            imgs_raw.append(raw)
            imgs_vis.append(vis)

            media = float(xt.mean().item())
            desvio = float(xt.std().item())

            medias.append(media)
            desvios.append(desvio)

            estatisticas.append({
                "schedule": scheduler,
                "amostra": idx,
                "timestep": t,
                "media_raw": media,
                "desvio_raw": desvio,
                "alpha_acumulado": float(
                    pipe.alphas_cumprod[t].item()
                )
            })

        arquivos.append(
            plot_grid_normalizado(
                imgs_vis,
                steps,
                idx,
                sample["label"],
                scheduler
            )
        )

        if graphs:

            arquivos.append(
                plot_histogramas_paralelos(
                    imgs_raw,
                    imgs_vis,
                    steps,
                    idx
                )
            )

            arquivos.append(
                plot_evolucao_mean_std(
                    steps,
                    medias,
                    desvios,
                    idx
                )
            )

        print(
            f"{scheduler}: "
            f"amostra {numero}/{len(indices)} "
            f"(indice {idx}) - OK"
        )

    return arquivos, estatisticas


# ==========================================================
# 10 - CARREGAR DATASET EXISTENTE
# ==========================================================

print("\n1 - CARREGANDO DATASET DA AULA 07")

csv_labels = DADOS / "labels.csv"

if not csv_labels.exists():
    raise SystemExit("labels.csv nao encontrado")

dataset = PixelArtDataset(
    DADOS,
    csv_labels
)

assert len(dataset) > 0

print("Imagens:", len(dataset))
print("Labels indexados:", len(dataset.captions))

assert len(dataset.captions) > 0


# ==========================================================
# 11 - VALIDAR DATALOADER
# ==========================================================

print("\n2 - VALIDANDO DATALOADER")

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

assert batch["image"].shape == (8, 3, 16, 16)
assert batch["label"].shape == (8, 5)

print(
    "Imagens:",
    tuple(batch["image"].shape)
)

print(
    "Labels:",
    tuple(batch["label"].shape)
)


# ==========================================================
# 12 - SELECIONAR CINCO AMOSTRAS
# ==========================================================

print("\n3 - SELECIONANDO AMOSTRAS")

indices = random.sample(
    range(len(dataset)),
    k=min(5, len(dataset))
)

print("Timesteps:", steps)
print("Indices:", indices)


# ==========================================================
# 13 - SCHEDULER LINEAR
# ==========================================================

print("\n4 - EXECUTANDO SCHEDULER LINEAR")

pipe_linear = DiffusionForward(
    timesteps=1000,
    schedule="linear",
    device=device
)

arquivos_linear, estat_linear = plot_list(
    dataset,
    indices,
    pipe_linear,
    "linear",
    graphs=True
)


# ==========================================================
# 14 - SCHEDULER COSSENO
# ==========================================================

print("\n5 - EXECUTANDO SCHEDULER COSSENO")

pipe_cosine = DiffusionForward(
    timesteps=1000,
    schedule="cosine",
    device=device
)

arquivos_cosine, estat_cosine = plot_list(
    dataset,
    indices,
    pipe_cosine,
    "cosine",
    graphs=False
)


# ==========================================================
# 15 - SALVAR CSV E JSON
# ==========================================================

print("\n6 - SALVANDO RESULTADOS")

arquivo_csv = (
    RESULTADOS / "aula08_estatisticas_scheduler.csv"
)

with arquivo_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(estat_linear[0].keys())
    )

    writer.writeheader()
    writer.writerows(estat_linear + estat_cosine)


arquivo_json = (
    RESULTADOS / "aula08_parametros.json"
)

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "aula": 8,
            "total_imagens": len(dataset),
            "rotulos_indexados": len(dataset.captions),
            "timesteps": 1000,
            "steps_observados": steps,
            "indices": indices,
            "device": device,
            "seed": SEED,
            "scheduler_linear": True,
            "scheduler_cosine": True
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# ==========================================================
# 16 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert len(indices) == 5
assert len(arquivos_linear) == 15
assert len(arquivos_cosine) == 5

assert len(estat_linear) == 35
assert len(estat_cosine) == 35

assert len(pipe_linear.betas) == 1000
assert len(pipe_cosine.betas) == 1000

for arquivo in (
    arquivos_linear
    + arquivos_cosine
    + [arquivo_csv, arquivo_json]
):
    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("Dataset PixelArt: OK")
print("Labels: OK")
print("DataLoader: OK")
print("Scheduler linear: OK")
print("Scheduler cosseno: OK")
print("Funcao extract: OK")
print("Classe DiffusionForward: OK")
print("q_sample: OK")
print("5 amostras x 7 timesteps: OK")
print("10 paineis de imagem: OK")
print("5 paineis de histogramas: OK")
print("5 graficos de media e desvio: OK")
print("CSV e JSON: OK")

print("\nAULA 08 EXECUTADA COM SUCESSO!")
