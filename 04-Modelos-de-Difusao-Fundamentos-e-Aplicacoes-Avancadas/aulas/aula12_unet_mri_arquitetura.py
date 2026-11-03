# ============================================================
# PROJETO 04 - AULA 12
# ARQUITETURA RESIDUAL U-NET / BRAIN MRI
# ============================================================

from pathlib import Path
import json
import math
import os
import random
import shutil

import kagglehub
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchview import draw_graph


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "brain-mri-images"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp"}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))

print("=" * 65)
print("AULA 12 - ARQUITETURA RESIDUAL U-NET / MRI")
print("=" * 65)
print("PyTorch:", torch.__version__)
print("Dispositivo:", device)


# ============================================================
# 2 - BAIXAR DATASET
# ============================================================

print("\n1 - LOCALIZANDO DATASET BRAIN MRI")

arquivos = [
    p for p in DADOS.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXTENSOES
]

if not arquivos:

    print("Baixando dataset de ressonancia...")

    caminho = kagglehub.dataset_download(
        "ashfakyeafi/brain-mri-images",
        output_dir=str(DADOS)
    )

    print("Download:", caminho)

else:

    print("Dataset ja encontrado no projeto.")

print("Pasta:", DADOS)


# ============================================================
# 3 - DATASET CUSTOMIZADO
# ============================================================

class BrainMRIDataset(Dataset):

    def __init__(
        self,
        images_dir,
        transform=None,
        slice_plane="z",
        side=None
    ):

        self.transform = transform
        self.slice_plane = slice_plane
        self.side = side
        self.image_files = []
        self.plane_counts = {}

        for caminho in sorted(Path(images_dir).rglob("*")):

            if not caminho.is_file():
                continue

            if caminho.suffix.lower() not in EXTENSOES:
                continue

            parts = caminho.stem.lower().split("_")

            if len(parts) <= 3:
                continue

            plane = parts[3]

            self.plane_counts[plane] = (
                self.plane_counts.get(plane, 0) + 1
            )

            if plane == self.slice_plane:
                self.image_files.append(caminho)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        caminho = self.image_files[idx]

        with Image.open(caminho) as arq:
            image = arq.convert("L")

        if self.transform:
            image = self.transform(image)

        slice_plane = caminho.stem.lower().split("_")[3]

        return {
            "image": image,
            "slice": slice_plane
        }


# ============================================================
# 4 - TRANSFORMACOES
# ============================================================

print("\n2 - CONFIGURANDO TRANSFORMACOES")

img_size = 64


def transform_mri(image):

    image = image.resize(
        (img_size, img_size),
        Image.Resampling.BILINEAR
    )

    valores = (
        np.asarray(image, dtype=np.float32)
        .copy() / 255.0
    )

    tensor = torch.from_numpy(
        valores
    ).unsqueeze(0)

    return tensor * 2.0 - 1.0


dataset = BrainMRIDataset(
    images_dir=DADOS,
    transform=transform_mri,
    slice_plane="z"
)

print("Planos encontrados:", dataset.plane_counts)
print("Imagens do plano Z:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError(
        "Nenhuma imagem do plano Z encontrada. "
        "Confira os nomes dos arquivos baixados."
    )

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

print("Batch:", tuple(batch["image"].shape))

assert batch["image"].shape[1:] == (1, 64, 64)

print("Dataset e DataLoader: OK")


# ============================================================
# 5 - VISUALIZAR 16 RESSONANCIAS
# ============================================================

print("\n3 - VISUALIZANDO AMOSTRAS MRI")

indices = np.random.choice(
    len(dataset),
    size=min(16, len(dataset)),
    replace=False
)

fig, axes = plt.subplots(
    4, 4,
    figsize=(9, 9)
)

for ax in axes.flat:
    ax.axis("off")

for ax, idx in zip(axes.flat, indices):

    sample = dataset[int(idx)]

    img = (
        (sample["image"].squeeze(0) + 1) / 2
    ).clamp(0, 1).numpy()

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"#{idx}: {sample['slice']}",
        fontsize=9
    )

fig.suptitle(
    "Brain MRI - Plano Z - 64x64"
)

fig.tight_layout()

arquivo_grid = (
    RESULTADOS / "aula12_amostras_mri_z.png"
)

fig.savefig(
    arquivo_grid,
    dpi=150
)

plt.close(fig)

print("Grid de ressonancias: OK")


# ============================================================
# 6 - EMBEDDING TEMPORAL
# ============================================================

print("\n4 - CRIANDO SINUSOIDAL EMBEDDING")


def sinusoidal_embedding(t, dim):

    half = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=t.device,
            dtype=torch.float32
        ) / (half - 1)
    )

    emb = t[:, None].float() * freqs[None, :]

    return torch.cat(
        (emb.sin(), emb.cos()),
        dim=1
    )


print("Sinusoidal embedding: OK")


# ============================================================
# 7 - BLOCO RESIDUAL
# ============================================================

class ResidualTimeBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        t_dim,
        groups=8
    ):

        super().__init__()

        self.same_channels = in_ch == out_ch

        self.conv1 = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            groups, out_ch
        )

        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(
            t_dim, out_ch
        )

        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            groups, out_ch
        )

        self.act2 = nn.SiLU()

        self.skip = (
            nn.Identity()
            if self.same_channels
            else nn.Conv2d(
                in_ch, out_ch,
                kernel_size=1
            )
        )

    def forward(self, x, t_emb):

        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        h = h + self.time_proj(
            t_emb
        )[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return (h + residual) / 1.414


# ============================================================
# 8 - DOWN E UP
# ============================================================

class Down(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        t_dim
    ):

        super().__init__()

        self.block1 = ResidualTimeBlock(
            in_ch, out_ch, t_dim
        )

        self.block2 = ResidualTimeBlock(
            out_ch, out_ch, t_dim
        )

        self.pool = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x, t_emb):

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)

        return x, self.pool(x)


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
            in_ch, out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block1 = ResidualTimeBlock(
            out_ch + skip_ch,
            out_ch,
            t_dim
        )

        self.block2 = ResidualTimeBlock(
            out_ch, out_ch,
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


# ============================================================
# 9 - U-NET RESIDUAL
# ============================================================

print("\n5 - CONSTRUINDO U-NET")


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        time_dim=256
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

        self.down3 = Down(
            base_channels * 2,
            base_channels * 4,
            time_dim
        )

        self.down4 = Down(
            base_channels * 4,
            base_channels * 8,
            time_dim
        )

        self.bot1 = ResidualTimeBlock(
            base_channels * 8,
            base_channels * 16,
            time_dim
        )

        self.bot2 = ResidualTimeBlock(
            base_channels * 16,
            base_channels * 8,
            time_dim
        )

        self.up4 = Up(
            base_channels * 8,
            base_channels * 8,
            base_channels * 4,
            time_dim
        )

        self.up3 = Up(
            base_channels * 4,
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

        self.out = nn.Conv2d(
            base_channels,
            in_channels,
            kernel_size=1
        )

    def forward(self, x, t):

        t_emb = sinusoidal_embedding(
            t,
            self.time_mlp[0].in_features
        )

        t_emb = self.time_mlp(t_emb)

        s1, p1 = self.down1(x, t_emb)
        s2, p2 = self.down2(p1, t_emb)
        s3, p3 = self.down3(p2, t_emb)
        s4, p4 = self.down4(p3, t_emb)

        b = self.bot1(p4, t_emb)
        b = self.bot2(b, t_emb)

        u4 = self.up4(b, s4, t_emb)
        u3 = self.up3(u4, s3, t_emb)
        u2 = self.up2(u3, s2, t_emb)
        u1 = self.up1(u2, s1, t_emb)

        return self.out(u1)


model = UNet().to(device)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("UNet criada: OK")
print("Parametros:", parametros)
print("AdamW: 0.0001")
print("Treinamento executado: NAO")


# ============================================================
# 10 - TESTAR FORWARD
# ============================================================

print("\n6 - TESTANDO FORWARD DA U-NET")

x_sample = batch["image"][:1].to(device)

t_sample = torch.tensor(
    [500],
    dtype=torch.long,
    device=device
)

model.eval()

with torch.no_grad():

    saida = model(
        x_sample,
        t_sample
    )

print(
    "Entrada:",
    tuple(x_sample.shape)
)

print(
    "Saida:",
    tuple(saida.shape)
)

assert saida.shape == (1, 1, 64, 64)

assert torch.isfinite(saida).all()

print("Forward da U-Net: OK")


# ============================================================
# 11 - TORCHVIEW
# ============================================================

print("\n7 - TESTANDO TORCHVIEW")


class WrappedUNet(nn.Module):

    def __init__(self, unet):

        super().__init__()

        self.unet = unet

    def forward(self, x):

        t = torch.full(
            (x.size(0),),
            500,
            dtype=torch.long,
            device=x.device
        )

        return self.unet(x, t)


status_torchview = "nao executado"

try:

    graph = draw_graph(
        WrappedUNet(model),
        input_data=x_sample,
        graph_name="UNet Brain MRI",
        roll=True
    )

    arquivo_dot = (
        RESULTADOS / "aula12_unet_torchview.dot"
    )

    arquivo_dot.write_text(
        graph.visual_graph.source,
        encoding="utf-8"
    )

    print("Grafo DOT: OK")

    status_torchview = "DOT gerado"

    if shutil.which("dot"):

        graph.visual_graph.render(
            filename=str(
                RESULTADOS / "aula12_unet_torchview"
            ),
            format="png",
            cleanup=True
        )

        status_torchview = "PNG renderizado"

        print("Diagrama TorchView PNG: OK")

    else:

        print(
            "Executavel Graphviz dot indisponivel."
        )

        print(
            "Arquivo DOT salvo para visualizacao."
        )

except Exception as erro:

    status_torchview = (
        f"Erro de visualizacao: {erro}"
    )

    print(
        "TorchView:",
        type(erro).__name__,
        str(erro)
    )

    print(
        "Teste da arquitetura continua valido."
    )


# ============================================================
# 12 - METADADOS
# ============================================================

print("\n8 - SALVANDO METADADOS")

arquivo_json = (
    RESULTADOS / "aula12_unet_mri_configuracoes.json"
)

dados = {
    "aula": 12,
    "dataset": "ashfakyeafi/brain-mri-images",
    "total_imagens_z": len(dataset),
    "image_shape": [1, 64, 64],
    "batch_size": 128,
    "base_channels": 64,
    "time_dim": 256,
    "parametros": parametros,
    "learning_rate": 0.0001,
    "otimizador": "AdamW",
    "timestep_teste": 500,
    "entrada": list(x_sample.shape),
    "saida": list(saida.shape),
    "torchview": status_torchview,
    "device": device,
    "treinamento_executado": False
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dados,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 13 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert len(dataset) > 0
assert x_sample.shape == (1, 1, 64, 64)
assert saida.shape == x_sample.shape
assert parametros > 0

for arquivo in [
    arquivo_grid,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("Dataset Brain MRI: OK")
print("Filtro plano Z: OK")
print("Transformacoes 64x64: OK")
print("DataLoader: OK")
print("Sinusoidal embedding: OK")
print("ResidualTimeBlock: OK")
print("Downsampling: OK")
print("Upsampling: OK")
print("Skip connections: OK")
print("UNet residual: OK")
print("Forward da rede: OK")
print("AdamW configurado: OK")
print("TorchView:", status_torchview)

print("\nTREINAMENTO EXECUTADO: NAO")
print("AULA 12 - ARQUITETURA VALIDADA!")
