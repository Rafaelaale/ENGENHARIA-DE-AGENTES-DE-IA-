# ============================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 06 - FORWARD PROCESS
# ============================================================

from pathlib import Path
from io import BytesIO
import csv
import json
import math

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import requests

from PIL import Image, ImageDraw
from scipy.stats import norm


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

rng = np.random.default_rng(42)

print("=" * 65)
print("AULA 06 - FORWARD PROCESS")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("Dispositivo: CPU")
print("Seed: 42")


# ============================================================
# 2 - BETA SCHEDULER
# ============================================================

def beta_schedule(timesteps, schedule="cosine"):

    if schedule == "linear":

        return torch.linspace(
            1e-4,
            0.02,
            timesteps
        )

    if schedule == "cosine":

        s = 0.008

        steps = torch.arange(
            timesteps + 1,
            dtype=torch.float32
        )

        alphas_cum = torch.cos(
            ((steps / timesteps) + s)
            / (1 + s)
            * math.pi
            * 0.5
        ) ** 2

        alphas_cum = (
            alphas_cum / alphas_cum[0]
        )

        betas = (
            1 - (
                alphas_cum[1:]
                / alphas_cum[:-1]
            )
        )

        return betas.clamp(
            min=0.000001,
            max=0.999
        )

    raise ValueError(
        "Scheduler invalido."
    )


print("\n1 - CONFIGURANDO BETA SCHEDULER")

timesteps = 1000

betas = beta_schedule(
    timesteps,
    schedule="linear"
)

betas_cosine = beta_schedule(
    timesteps,
    schedule="cosine"
)

assert len(betas) == 1000

assert torch.all(
    (betas > 0) & (betas < 1)
)

print("Scheduler linear: OK")
print("Scheduler cosseno: OK")
print("Timesteps:", timesteps)


# ============================================================
# 3 - CARREGAR IMAGEM ORIGINAL
# ============================================================

print("\n2 - CARREGANDO IMAGEM")

url = (
    "https://tse3.mm.bing.net/th/id/"
    "OIP.-SXhM0P_yHjJFvUZs2GUJAHaHa"
    "?rs=1&pid=ImgDetMain&o=7&rm=3"
)

arquivo_original = (
    RESULTADOS / "aula06_imagem_original.png"
)


def criar_imagem_local():

    imagem = Image.new(
        "RGB",
        (128, 128),
        (220, 235, 250)
    )

    desenho = ImageDraw.Draw(imagem)

    desenho.rectangle(
        (0, 90, 128, 128),
        fill=(90, 150, 110)
    )

    desenho.ellipse(
        (27, 25, 100, 103),
        fill=(230, 155, 65),
        outline=(120, 75, 35),
        width=2
    )

    desenho.polygon(
        [(31, 46), (42, 13), (55, 39)],
        fill=(230, 155, 65)
    )

    desenho.polygon(
        [(74, 39), (88, 13), (98, 47)],
        fill=(230, 155, 65)
    )

    desenho.ellipse(
        (48, 56, 55, 64),
        fill="black"
    )

    desenho.ellipse(
        (78, 56, 85, 64),
        fill="black"
    )

    desenho.polygon(
        [(61, 73), (73, 73), (67, 81)],
        fill=(140, 55, 65)
    )

    return imagem


if arquivo_original.exists():

    image = Image.open(
        arquivo_original
    ).convert("RGB").resize((128, 128))

    origem = "arquivo local existente"

else:

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB").resize((128, 128))

        origem = "imagem da internet"

    except Exception as erro:

        print(
            "Download indisponivel:",
            type(erro).__name__
        )

        image = criar_imagem_local()

        origem = "ilustracao local alternativa"

    image.save(arquivo_original)


image_array = (
    np.asarray(image, dtype=np.float32)
    / 255.0
)

assert image_array.shape == (128, 128, 3)

print("Origem:", origem)
print("Dimensoes:", image_array.shape)
print("Imagem carregada: OK")


# ============================================================
# 4 - ESTIMAR ENTROPIA
# ============================================================

def estimate_entropy(data, bins=50):

    valores = np.asarray(
        data,
        dtype=np.float64
    ).flatten()

    hist, bordas = np.histogram(
        valores,
        bins=bins,
        density=True
    )

    largura = bordas[1] - bordas[0]

    hist = np.maximum(
        hist,
        1e-12
    )

    return float(
        -np.sum(
            hist * np.log(hist) * largura
        )
    )


# ============================================================
# 5 - FORWARD DIFFUSION NA IMAGEM
# ============================================================

print("\n3 - EXECUTANDO 1000 ETAPAS DE RUIDO")

steps_visualizacao = [
    0,
    20,
    50,
    100,
    999
]

imagens_selecionadas = {}
estatisticas = []

alpha_cumprod = 1.0

for t in range(timesteps):

    beta_t = float(betas[t].item())

    alpha_cumprod *= (
        1.0 - beta_t
    )

    noise = rng.normal(
        0,
        1,
        image_array.shape
    )

    alpha = np.sqrt(alpha_cumprod)

    sigma = np.sqrt(
        1.0 - alpha_cumprod
    )

    # Formula acumulada apresentada na aula:
    # x_t = sqrt(alpha_bar_t)*x_0
    #     + sqrt(1-alpha_bar_t)*epsilon

    noisy_image = (
        alpha * image_array
        + sigma * noise
    )

    if t in steps_visualizacao:

        imagens_selecionadas[t] = (
            noisy_image.copy()
        )

        valores = noisy_image.flatten()

        mu, std = norm.fit(valores)

        estatisticas.append({
            "timestep": t,
            "beta": beta_t,
            "alpha_acumulado": alpha_cumprod,
            "media": float(mu),
            "desvio_padrao": float(std),
            "entropia": estimate_entropy(valores)
        })

        print(
            f"t={t:03d} | "
            f"alpha_bar={alpha_cumprod:.5f} | "
            f"media={mu:.4f} | "
            f"desvio={std:.4f}"
        )

assert len(imagens_selecionadas) == 5

print("Forward diffusion: OK")


# ============================================================
# 6 - VISUALIZAR IMAGEM E HISTOGRAMAS
# ============================================================

print("\n4 - CRIANDO PAINEL DE HISTOGRAMAS")

fig, axes = plt.subplots(
    2,
    len(steps_visualizacao),
    figsize=(19, 8)
)

for idx, step in enumerate(
    steps_visualizacao
):

    img = imagens_selecionadas[step]

    axes[0, idx].imshow(
        np.clip(img, 0, 1)
    )

    axes[0, idx].axis("off")

    axes[0, idx].set_title(
        f"t={step}"
    )

    data = img.flatten()

    axes[1, idx].hist(
        data,
        bins=50,
        density=True,
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )

    mu, std = norm.fit(data)

    xmin, xmax = axes[1, idx].get_xlim()

    x = np.linspace(
        xmin,
        xmax,
        100
    )

    p = norm.pdf(
        x,
        mu,
        std
    )

    axes[1, idx].plot(
        x,
        p,
        color="red",
        linewidth=2
    )

    axes[1, idx].set_title(
        f"media={mu:.2f} | sigma={std:.2f}"
    )

    axes[1, idx].set_xlabel(
        "Valor do pixel"
    )

    axes[1, idx].set_ylabel(
        "Densidade"
    )

fig.suptitle(
    "Forward Process: imagem e distribuicao dos pixels",
    fontsize=15
)

fig.tight_layout()

arquivo_painel = (
    RESULTADOS
    / "aula06_imagem_histogramas.png"
)

fig.savefig(
    arquivo_painel,
    dpi=140
)

plt.close(fig)

print("Imagem e histogramas: OK")


# ============================================================
# 7 - 20 IMAGENS DE RUIDO ALEATORIO
# ============================================================

print("\n5 - GERANDO 20 IMAGENS ALEATORIAS")

width, height = 26, 26

images = [
    rng.integers(
        0,
        256,
        size=(height, width),
        dtype=np.uint8
    )
    for _ in range(20)
]

fig, axes = plt.subplots(
    1,
    20,
    figsize=(18, 3)
)

for idx, (ax, img) in enumerate(
    zip(axes, images),
    start=1
):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(
        str(idx),
        fontsize=9
    )

    ax.axis("off")

fig.suptitle(
    "20 imagens aleatorias de 26 x 26 pixels"
)

fig.tight_layout()

arquivo_aleatorias = (
    RESULTADOS
    / "aula06_20_imagens_aleatorias.png"
)

fig.savefig(
    arquivo_aleatorias,
    dpi=130
)

plt.close(fig)

print("20 imagens de ruido: OK")


# ============================================================
# 8 - VISUALIZAR ENTROPIA
# ============================================================

print("\n6 - CALCULANDO ENTROPIA")

tempos = [
    item["timestep"]
    for item in estatisticas
]

entropias = [
    item["entropia"]
    for item in estatisticas
]

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    tempos,
    entropias,
    marker="o",
    linewidth=2,
    color="steelblue"
)

ax.set_xlabel(
    "Timestep"
)

ax.set_ylabel(
    "Entropia diferencial estimada"
)

ax.set_title(
    "Entropia estimada durante o Forward Process"
)

ax.grid(alpha=0.3)

fig.tight_layout()

arquivo_entropia = (
    RESULTADOS
    / "aula06_entropia_forward.png"
)

fig.savefig(
    arquivo_entropia,
    dpi=150
)

plt.close(fig)

for item in estatisticas:

    print(
        f"t={item['timestep']:03d} "
        f"| H={item['entropia']:.4f}"
    )

print("Grafico da entropia: OK")


# ============================================================
# 9 - SCORE FUNCTION
# ============================================================

print("\n7 - CRIANDO SCORE FUNCTION")


def log_p(x, y):

    return -0.5 * (
        x ** 2 + y ** 2
    )


def score(x, y):

    return -x, -y


x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

X, Y = np.meshgrid(x, y)

Z = np.exp(
    log_p(X, Y)
)

print("Grade da distribuicao: OK")
print("Score analitico: OK")


# ============================================================
# 10 - MOVIMENTO FORWARD EM 2D
# ============================================================

print("\n8 - SIMULANDO TRAJETORIA FORWARD")

steps = 30

point = np.array([
    0.0,
    0.0
])

trajectory_forward = [
    point.copy()
]

for _ in range(steps):

    noise = rng.normal(
        0,
        0.5,
        size=2
    )

    point = point + noise

    trajectory_forward.append(
        point.copy()
    )

assert len(trajectory_forward) == 31

print("Trajetoria forward: 31 pontos")


# ============================================================
# 11 - MOVIMENTO REVERSE COM SCORE
# ============================================================

print("\n9 - SIMULANDO TRAJETORIA REVERSE")

point = np.array([
    2.5,
    -2.0
])

trajectory_reverse = [
    point.copy()
]

eta = 0.1

for _ in range(steps):

    dx, dy = score(
        point[0],
        point[1]
    )

    point = point + eta * np.array([
        dx,
        dy
    ])

    trajectory_reverse.append(
        point.copy()
    )

assert len(trajectory_reverse) == 31

print("Trajetoria reverse: 31 pontos")
print("Ponto final:", trajectory_reverse[-1])


# ============================================================
# 12 - GERADOR DE GIF 3D
# ============================================================

print("\n10 - GERANDO ANIMACOES 3D")


def gerar_gif_3d(
    trajetoria,
    nome,
    cor,
    titulo
):

    pasta_quadros = (
        RESULTADOS
        / f"aula06_quadros_{nome}"
    )

    pasta_quadros.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_gif = (
        RESULTADOS
        / f"aula06_{nome}_3d.gif"
    )

    with imageio.get_writer(
        arquivo_gif,
        mode="I",
        duration=0.17,
        loop=0
    ) as writer:

        for i in range(len(trajetoria)):

            fig = plt.figure(
                figsize=(8, 6)
            )

            ax = fig.add_subplot(
                111,
                projection="3d"
            )

            ax.plot_surface(
                X,
                Y,
                Z,
                cmap="Blues",
                alpha=0.6,
                rstride=3,
                cstride=3,
                linewidth=0
            )

            traj = np.array(
                trajetoria[:i + 1]
            )

            alturas = np.exp(
                log_p(
                    traj[:, 0],
                    traj[:, 1]
                )
            )

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                alturas,
                color=cor,
                marker="o",
                linewidth=2
            )

            ax.set_title(
                f"{titulo} - passo {i}"
            )

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_zlim(0, 1)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("Densidade relativa")

            quadro = (
                pasta_quadros
                / f"passo_{i:02d}.png"
            )

            fig.savefig(
                quadro,
                dpi=85
            )

            plt.close(fig)

            imagem_quadro = (
                imageio.imread(quadro)
            )

            writer.append_data(
                imagem_quadro
            )

            if i % 10 == 0 or i == 30:

                print(
                    f"{nome}: "
                    f"{i + 1}/31 quadros"
                )

    assert arquivo_gif.exists()

    return arquivo_gif


gif_forward = gerar_gif_3d(
    trajectory_forward,
    "forward",
    "red",
    "Forward: movimento aleatorio"
)

gif_reverse = gerar_gif_3d(
    trajectory_reverse,
    "reverse",
    "green",
    "Reverse: seguindo o score"
)

print("GIF forward: OK")
print("GIF reverse: OK")


# ============================================================
# 13 - SALVAR ESTATISTICAS
# ============================================================

print("\n11 - SALVANDO RESULTADOS")

arquivo_csv = (
    RESULTADOS
    / "aula06_estatisticas_forward.csv"
)

with arquivo_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    writer = csv.DictWriter(
        arquivo,
        fieldnames=[
            "timestep",
            "beta",
            "alpha_acumulado",
            "media",
            "desvio_padrao",
            "entropia"
        ]
    )

    writer.writeheader()
    writer.writerows(estatisticas)


arquivo_json = (
    RESULTADOS
    / "aula06_configuracoes.json"
)

dados = {
    "aula": 6,
    "tema": "Forward Process",
    "seed": 42,
    "timesteps": timesteps,
    "scheduler_utilizado": "linear",
    "scheduler_cosine_implementado": True,
    "dimensoes_imagem": [128, 128, 3],
    "origem_imagem": origem,
    "passos_visualizados": steps_visualizacao,
    "passos_score": steps,
    "eta_reverse": eta,
    "ponto_final_reverse": (
        trajectory_reverse[-1].tolist()
    ),
    "estatisticas": estatisticas
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        dados,
        arquivo,
        indent=4,
        ensure_ascii=False
    )

print("CSV: OK")
print("JSON: OK")


# ============================================================
# 14 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert image_array.shape == (128, 128, 3)
assert len(betas) == 1000
assert len(imagens_selecionadas) == 5
assert len(images) == 20
assert len(trajectory_forward) == 31
assert len(trajectory_reverse) == 31

assert np.isfinite(entropias).all()

for arquivo in [
    arquivo_original,
    arquivo_painel,
    arquivo_aleatorias,
    arquivo_entropia,
    gif_forward,
    gif_reverse,
    arquivo_csv,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("PyTorch: OK")
print("NumPy: OK")
print("Scheduler linear: OK")
print("Scheduler cosseno: OK")
print("Imagem RGB: OK")
print("Mil etapas de difusao: OK")
print("Cinco histogramas: OK")
print("Vinte imagens aleatorias: OK")
print("Entropia estimada: OK")
print("Score Function: OK")
print("Trajetoria forward: OK")
print("Trajetoria reverse: OK")
print("GIF forward 3D: OK")
print("GIF reverse 3D: OK")
print("CSV e JSON: OK")

print("\nAULA 06 EXECUTADA COM SUCESSO!")
