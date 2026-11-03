# ============================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 10 - MINI U-NET
# ============================================================

from pathlib import Path
import json
import math
import re
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchview import draw_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "pixel-art"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("AULA 10 - CONSTRUINDO A MINI U-NET")
print("=" * 65)
print("PyTorch:", torch.__version__)
print("Dispositivo:", device)


# ============================================================
# 2 - SINUSOIDAL EMBEDDING
# ============================================================

print("\n1 - CRIANDO SINUSOIDAL EMBEDDING")


def sinusoidal_embedding(t, dim):

    half = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=t.device,
            dtype=torch.float32
        )
        / (half - 1)
    )

    emb = (
        t[:, None].float()
        * freqs[None, :]
    )

    return torch.cat(
        [emb.sin(), emb.cos()],
        dim=1
    )


teste_t = torch.tensor([0, 100, 499], device=device)

emb = sinusoidal_embedding(
    teste_t,
    128
)

assert emb.shape == (3, 128)

print("Embedding temporal:", tuple(emb.shape))
print("Sinusoidal embedding: OK")


# ============================================================
# 3 - BLOCO CONVOLUCIONAL
# ============================================================

print("\n2 - CRIANDO CONVBLOCK")


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        t_dim,
        groups=8
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.GroupNorm(
            groups,
            out_ch
        )

        self.act = nn.SiLU()

        self.time = nn.Linear(
            t_dim,
            out_ch
        )

        self.skip = (
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):

        residual = self.skip(x)

        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)

        t_proj = self.time(t_emb)[
            :, :, None, None
        ]

        h = h + t_proj

        return h + residual


print("ConvBlock: OK")


# ============================================================
# 4 - DOWNSAMPLING
# ============================================================

print("\n3 - CRIANDO DOWN")


class Down(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        t_dim
    ):

        super().__init__()

        self.block1 = ConvBlock(
            in_ch,
            out_ch,
            t_dim
        )

        self.block2 = ConvBlock(
            out_ch,
            out_ch,
            t_dim
        )

        self.pool = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x, t_emb):

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)

        p = self.pool(x)

        return x, p


print("Down: OK")


# ============================================================
# 5 - UPSAMPLING
# ============================================================

print("\n4 - CRIANDO UP")


class Up(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        t_dim
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block1 = ConvBlock(
            out_ch + skip_ch,
            out_ch,
            t_dim
        )

        self.block2 = ConvBlock(
            out_ch,
            out_ch,
            t_dim
        )

    def forward(self, x, skip, t_emb):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)

        return x


print("Up: OK")


# ============================================================
# 6 - ARQUITETURA MINI U-NET
# ============================================================

print("\n5 - CONSTRUINDO MINI U-NET")


class MiniUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        time_dim=128
    ):

        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_dim,
                time_dim * 4
            ),
            nn.SiLU(),
            nn.Linear(
                time_dim * 4,
                time_dim
            )
        )

        self.down1 = Down(
            in_channels,
            base_channels,
            time_dim
        )

        self.down2 = Down(
            base_channels,
            base_channels * 2,
            time_dim
        )

        self.bot1 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim
        )

        self.bot2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
            time_dim
        )

        self.up2 = Up(
            base_channels * 2,
            base_channels * 2,
            base_channels,
            time_dim
        )

        self.up1 = Up(
            base_channels,
            base_channels,
            base_channels,
            time_dim
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                base_channels,
                in_channels,
                kernel_size=1
            )
        )

    def forward(self, x, t):

        t_emb = sinusoidal_embedding(
            t,
            self.time_mlp[0].in_features
        )

        t_emb = self.time_mlp(t_emb)

        s1, p1 = self.down1(
            x,
            t_emb
        )

        s2, p2 = self.down2(
            p1,
            t_emb
        )

        b = self.bot1(
            p2,
            t_emb
        )

        b = self.bot2(
            b,
            t_emb
        )

        u2 = self.up2(
            b,
            s2,
            t_emb
        )

        u1 = self.up1(
            u2,
            s1,
            t_emb
        )

        return self.out(u1)


model = MiniUNet().to(device)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Modelo criado: OK")
print("Parametros totais:", parametros)


# ============================================================
# 7 - PREPARAR Q_SAMPLE
# ============================================================

print("\n6 - CONFIGURANDO Q_SAMPLE")

timesteps = 500

betas = torch.linspace(
    1e-4,
    0.02,
    timesteps,
    device=device
)

alphas = 1.0 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)


def q_sample(x0, t, noise):

    t = t.to(
        device=x0.device,
        dtype=torch.long
    )

    sqrt_alpha_bar_t = torch.sqrt(
        alpha_bar[t]
    ).view(-1, 1, 1, 1)

    sqrt_one_minus_alpha_bar_t = torch.sqrt(
        1 - alpha_bar[t]
    ).view(-1, 1, 1, 1)

    return (
        sqrt_alpha_bar_t * x0
        + sqrt_one_minus_alpha_bar_t * noise
    )


print("Timesteps:", timesteps)
print("Schedule: linear")
print("q_sample: OK")


# ============================================================
# 8 - DATASET PIXELART FILTRADO
# ============================================================

print("\n7 - CARREGANDO DATASET DA AULA 09")


class PixelArtDataset(Dataset):

    def __init__(self, pasta, csv_labels):

        extensoes = (
            ".png", ".jpg", ".jpeg", ".bmp"
        )

        self.image_files = sorted(
            p
            for p in pasta.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensoes
        )

        df = pd.read_csv(csv_labels)

        self.captions = {}

        for path, label in zip(
            df["Image Path"],
            df["Label"]
        ):

            nome = (
                Path(
                    str(path).replace("\\", "/")
                ).stem.lower() + ".png"
            )

            texto = re.sub(
                r"[\[\],]",
                " ",
                str(label)
            )

            valores = np.fromstring(
                texto,
                sep=" ",
                dtype=np.float32
            )

            if len(valores) == 5:
                self.captions[nome] = (
                    valores.tolist()
                )

        filtered = []

        for caminho in self.image_files:

            nome = caminho.stem.lower() + ".png"
            label = self.captions.get(nome)

            if (
                label is not None
                and np.isclose(label[3], 1.0)
            ):
                filtered.append(caminho)

        self.image_files = filtered

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        caminho = self.image_files[idx]

        with Image.open(caminho) as arq:

            img = arq.convert("RGB").resize(
                (16, 16),
                Image.Resampling.BILINEAR
            )

        valores = (
            np.asarray(
                img,
                dtype=np.float32
            ).copy() / 255.0
        )

        tensor = (
            torch.from_numpy(valores)
            .permute(2, 0, 1)
        )

        tensor = tensor * 2.0 - 1.0

        nome = caminho.stem.lower() + ".png"

        return {
            "image": tensor,
            "label": torch.tensor(
                self.captions[nome],
                dtype=torch.float32
            )
        }


dataset = PixelArtDataset(
    DADOS,
    DADOS / "labels.csv"
)

print("Imagens filtradas:", len(dataset))

assert len(dataset) > 0

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

x_batch = batch["image"].to(device)

print("Batch:", tuple(x_batch.shape))

assert x_batch.shape == (8, 3, 16, 16)


# ============================================================
# 9 - MODELO E OTIMIZADOR
# ============================================================

print("\n8 - CONFIGURANDO OTIMIZADOR")

opt = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4
)

epochs = 10

print("Otimizador: AdamW")
print("Learning rate: 0.0002")
print("Epocas configuradas:", epochs)
print("Treinamento iniciado: NAO")


# ============================================================
# 10 - VALIDAR FORWARD DA MINI U-NET
# ============================================================

print("\n9 - TESTANDO MINI U-NET")

x_sample = x_batch[:1]

# Com 500 timesteps os indices validos
# vao de 0 ate 499.
t_sample = torch.tensor(
    [499],
    dtype=torch.long,
    device=device
)

noise = torch.randn_like(
    x_sample
)

xt = q_sample(
    x_sample,
    t_sample,
    noise
)

model.eval()

with torch.no_grad():

    ruido_previsto = model(
        xt,
        t_sample
    )

    erro_inicial = F.mse_loss(
        ruido_previsto,
        noise
    ).item()

assert xt.shape == x_sample.shape

assert (
    ruido_previsto.shape
    == x_sample.shape
)

assert torch.isfinite(
    ruido_previsto
).all()

print(
    "Entrada:",
    tuple(x_sample.shape)
)

print(
    "Imagem com ruido:",
    tuple(xt.shape)
)

print(
    "Saida Mini U-Net:",
    tuple(ruido_previsto.shape)
)

print(
    "MSE inicial, sem treinamento:",
    round(erro_inicial, 6)
)

print("Forward da rede neural: OK")


# ============================================================
# 11 - SALVAR VISUALIZACAO DO TESTE
# ============================================================

print("\n10 - VISUALIZANDO TESTE")

def para_imagem(tensor):

    return (
        ((tensor.detach().cpu() + 1) / 2)
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )


fig, axes = plt.subplots(
    1, 3,
    figsize=(10, 3)
)

imagens = [
    para_imagem(x_sample),
    para_imagem(xt),
    para_imagem(ruido_previsto)
]

titulos = [
    "Imagem original",
    "Imagem com ruido",
    "Saida sem treinamento"
]

for ax, imagem, titulo in zip(
    axes,
    imagens,
    titulos
):

    ax.imshow(
        imagem,
        interpolation="nearest"
    )

    ax.set_title(titulo)
    ax.axis("off")

fig.tight_layout()

arquivo_teste = (
    RESULTADOS
    / "aula10_mini_unet_teste.png"
)

fig.savefig(
    arquivo_teste,
    dpi=160
)

plt.close(fig)

print("Visualizacao do teste: OK")


# ============================================================
# 12 - VISUALIZACAO DA ARQUITETURA
# ============================================================

print("\n11 - GERANDO DIAGRAMA DA U-NET")

# Diagrama simplificado: sempre gerado
fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")

blocos = [
    (1.5, 8.0, "Entrada\n3x16x16"),
    (1.5, 6.5, "Down 1\n32x8x8"),
    (1.5, 5.0, "Down 2\n64x4x4"),
    (5.0, 3.0, "Bottleneck\n128 -> 64"),
    (8.5, 5.0, "Up 2\n32x8x8"),
    (8.5, 6.5, "Up 1\n32x16x16"),
    (8.5, 8.0, "Saida\n3x16x16")
]

for x, y, titulo in blocos:

    ax.text(
        x, y, titulo,
        ha="center",
        va="center",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "aliceblue",
            "edgecolor": "steelblue"
        }
    )

setas = [
    ((1.5, 7.65), (1.5, 6.85)),
    ((1.5, 6.15), (1.5, 5.35)),
    ((1.7, 4.7), (4.6, 3.2)),
    ((5.4, 3.2), (8.3, 4.7)),
    ((8.5, 5.35), (8.5, 6.15)),
    ((8.5, 6.85), (8.5, 7.65))
]

for inicio, fim in setas:

    ax.annotate(
        "",
        xy=fim,
        xytext=inicio,
        arrowprops={
            "arrowstyle": "->",
            "color": "gray",
            "lw": 2
        }
    )

for altura in [5.0, 6.5]:

    ax.annotate(
        "",
        xy=(7.8, altura),
        xytext=(2.2, altura),
        arrowprops={
            "arrowstyle": "->",
            "linestyle": "dashed",
            "color": "seagreen",
            "lw": 2
        }
    )

ax.set_title(
    "Mini U-Net - arquitetura simplificada",
    fontsize=16
)

arquivo_arquitetura = (
    RESULTADOS
    / "aula10_mini_unet_arquitetura.png"
)

fig.savefig(
    arquivo_arquitetura,
    dpi=160,
    bbox_inches="tight"
)

plt.close(fig)

print("Diagrama simplificado: OK")


# ============================================================
# 13 - TORCHVIEW
# ============================================================

print("\n12 - TESTANDO TORCHVIEW")


class WrappedUNet(nn.Module):

    def __init__(self, unet):

        super().__init__()
        self.unet = unet

    def forward(self, x):

        t = torch.full(
            (x.shape[0],),
            499,
            dtype=torch.long,
            device=x.device
        )

        return self.unet(x, t)


wrapped_model = WrappedUNet(model)

torchview_status = "nao executado"

try:

    graph = draw_graph(
        wrapped_model,
        input_data=x_sample,
        graph_name="U-Net Pixel Art",
        roll=True
    )

    arquivo_dot = (
        RESULTADOS
        / "aula10_mini_unet_torchview.dot"
    )

    arquivo_dot.write_text(
        graph.visual_graph.source,
        encoding="utf-8"
    )

    torchview_status = "grafo DOT gerado"

    print("Grafo TorchView DOT: OK")

    if shutil.which("dot"):

        caminho_png = graph.visual_graph.render(
            filename=str(
                RESULTADOS
                / "aula10_mini_unet_torchview"
            ),
            format="png",
            cleanup=True
        )

        print("Grafo TorchView PNG:", caminho_png)

        torchview_status = "PNG renderizado"

    else:

        print(
            "Graphviz executavel 'dot' indisponivel."
        )

        print(
            "O DOT foi salvo; o diagrama "
            "simplificado PNG esta disponivel."
        )

except Exception as erro:

    torchview_status = (
        "falha na visualizacao: "
        + str(erro)
    )

    print(
        "TorchView nao renderizou:",
        type(erro).__name__
    )

    print(
        "O teste da Mini U-Net continua valido."
    )


# ============================================================
# 14 - SALVAR PESOS INICIAIS E METADADOS
# ============================================================

print("\n13 - SALVANDO RESULTADOS")

arquivo_pesos = (
    RESULTADOS
    / "aula10_pesos_iniciais_nao_treinados.pth"
)

torch.save(
    model.state_dict(),
    arquivo_pesos
)

arquivo_json = (
    RESULTADOS
    / "aula10_mini_unet_configuracoes.json"
)

informacoes = {
    "aula": 10,
    "modelo": "MiniUNet",
    "dataset_filtrado": len(dataset),
    "in_channels": 3,
    "base_channels": 32,
    "time_dim": 128,
    "timesteps": timesteps,
    "scheduler": "linear",
    "otimizador": "AdamW",
    "learning_rate": 0.0002,
    "epocas_configuradas": epochs,
    "treinamento_executado": False,
    "parametros": parametros,
    "entrada": list(x_sample.shape),
    "saida": list(ruido_previsto.shape),
    "mse_inicial_sem_treino": erro_inicial,
    "torchview_status": torchview_status,
    "device": device
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        informacoes,
        f,
        indent=4,
        ensure_ascii=False
    )

print("Pesos iniciais: OK")
print("JSON: OK")


# ============================================================
# 15 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert emb.shape == (3, 128)
assert len(dataset) > 0
assert x_batch.shape == (8, 3, 16, 16)

assert x_sample.shape == (
    1, 3, 16, 16
)

assert ruido_previsto.shape == (
    1, 3, 16, 16
)

assert parametros > 0
assert math.isfinite(erro_inicial)

for arquivo in [
    arquivo_teste,
    arquivo_arquitetura,
    arquivo_pesos,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("Sinusoidal embedding: OK")
print("ConvBlock: OK")
print("Downsampling: OK")
print("Upsampling: OK")
print("Skip connections: OK")
print("MiniUNet: OK")
print("q_sample: OK")
print("Dataset filtrado: OK")
print("DataLoader: OK")
print("AdamW configurado: OK")
print("Forward da rede: OK")
print("Formato de saida: OK")
print("Diagrama simplificado: OK")
print("Pesos iniciais: OK")
print("Metadados JSON: OK")
print("TorchView:", torchview_status)

print("\nTREINAMENTO EXECUTADO: NAO")
print("AULA 10 - ARQUITETURA VALIDADA!")
