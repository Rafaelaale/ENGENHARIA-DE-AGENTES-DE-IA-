# ============================================================
# PROJETO 04 - AULA 15
# DATASET E ARQUITETURA U-NET CONDICIONAL
# ============================================================

from pathlib import Path
from collections import Counter
import csv
import json
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchview import draw_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "brain-mri-images"
SAIDA = BASE / "resultados"

SAIDA.mkdir(parents=True, exist_ok=True)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

print("=" * 65)
print("AULA 15 - DATASET E U-NET CONDICIONAL")
print("=" * 65)
print("Dispositivo:", device)


# ============================================================
# 2 - DATASET COM LABELS X, Y, Z
# ============================================================

class BrainMRIConditionalDataset(Dataset):

    def __init__(
        self,
        images_dir,
        transform=None,
        slice_plane=None
    ):

        self.transform = transform

        self.image_files = []
        self.slice_planes = []

        for path in sorted(
            Path(images_dir).rglob("*")
        ):

            if (
                not path.is_file()
                or path.suffix.lower() not in EXT
            ):
                continue

            partes = path.stem.lower().split("_")

            if len(partes) <= 3:
                continue

            plano = partes[3]

            if plano not in ("x", "y", "z"):
                continue

            if (
                slice_plane is None
                or plano == slice_plane
            ):

                self.image_files.append(path)
                self.slice_planes.append(plano)

        # Mapeamento fixo e reproduzivel
        classes = sorted(
            set(self.slice_planes)
        )

        self.label_to_idx = {
            nome: i
            for i, nome in enumerate(classes)
        }

        self.idx_to_label = {
            i: nome
            for nome, i in self.label_to_idx.items()
        }

        self.labels = [
            self.label_to_idx[plano]
            for plano in self.slice_planes
        ]

        self.num_classes = len(classes)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        caminho = self.image_files[idx]

        with Image.open(caminho) as arquivo:
            image = arquivo.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "slice": self.slice_planes[idx],
            "label": self.labels[idx]
        }


# ============================================================
# 3 - TRANSFORMACOES
# ============================================================

def transform_mri(image):

    image = image.resize(
        (64, 64),
        Image.Resampling.BILINEAR
    )

    valores = (
        np.asarray(image, dtype=np.float32)
        .copy() / 255.0
    )

    return (
        torch.from_numpy(valores)
        .unsqueeze(0)
        * 2.0 - 1.0
    )


print("\n1 - CARREGANDO DATASET CONDICIONAL")

dataset = BrainMRIConditionalDataset(
    DADOS,
    transform=transform_mri,
    slice_plane=None
)

assert dataset.num_classes == 3, (
    "Esperadas classes X, Y e Z. "
    f"Encontradas: {dataset.label_to_idx}"
)

counts = Counter(dataset.slice_planes)

print("Total de imagens:", len(dataset))
print("Classes:", dataset.label_to_idx)

for classe in ("x", "y", "z"):

    print(
        f"Classe {classe}: {counts[classe]}"
    )

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

assert batch["image"].shape[1:] == (
    1, 64, 64
)

assert batch["label"].dtype == torch.int64

print(
    "Batch:",
    tuple(batch["image"].shape)
)

print(
    "Labels:",
    tuple(batch["label"].shape)
)


# ============================================================
# 4 - VISUALIZAR AS TRES CLASSES
# ============================================================

print("\n2 - VISUALIZANDO CLASSES")

fig, axes = plt.subplots(
    3, 4,
    figsize=(8, 7)
)

for linha, classe in enumerate(
    ("x", "y", "z")
):

    indices = [
        i
        for i, plano in enumerate(
            dataset.slice_planes
        )
        if plano == classe
    ][:4]

    for coluna, idx in enumerate(indices):

        imagem = dataset[idx]["image"][0]

        axes[linha, coluna].imshow(
            ((imagem + 1) / 2).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[linha, coluna].set_title(
            f"{classe.upper()} | "
            f"label={dataset.labels[idx]}",
            fontsize=9
        )

        axes[linha, coluna].axis("off")

fig.suptitle(
    "Brain MRI - Planos X, Y e Z"
)

fig.tight_layout()

arquivo_grid = (
    SAIDA / "aula15_amostras_x_y_z.png"
)

fig.savefig(
    arquivo_grid,
    dpi=150
)

plt.close(fig)

print("Grid X/Y/Z: OK")


# ============================================================
# 5 - SINUSOIDAL EMBEDDING
# ============================================================

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
        (emb.sin(), emb.cos()),
        dim=1
    )


# ============================================================
# 6 - LABEL EMBEDDING
# ============================================================

class LabelEmbedding(nn.Module):

    def __init__(
        self,
        num_classes,
        embed_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_classes,
            embed_dim
        )

    def forward(self, labels):

        return self.embedding(labels)


# ============================================================
# 7 - CONDITIONAL TIME EMBEDDING
# ============================================================

class ConditionalTimeEmbedding(nn.Module):

    def __init__(
        self,
        time_dim,
        label_dim
    ):

        super().__init__()

        self.time_dim = time_dim

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

        self.label_mlp = nn.Sequential(
            nn.Linear(
                label_dim,
                label_dim * 2
            ),
            nn.SiLU(),
            nn.Linear(
                label_dim * 2,
                time_dim
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                time_dim * 2,
                time_dim
            ),
            nn.SiLU()
        )

    def forward(self, t, label_emb):

        t_emb = sinusoidal_embedding(
            t,
            self.time_dim
        )

        t_emb = self.time_mlp(t_emb)

        l_emb = self.label_mlp(label_emb)

        combined = torch.cat(
            (t_emb, l_emb),
            dim=1
        )

        return self.fusion(combined)


# ============================================================
# 8 - BLOCO RESIDUAL CONDICIONAL
# ============================================================

class ConditionalResidualBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        time_dim,
        groups=8,
        dropout=0.1
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            groups,
            out_ch
        )

        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(
            time_dim,
            out_ch
        )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            groups,
            out_ch
        )

        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1
            )
        )

    def forward(self, x, cond_emb):

        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        cond = self.time_proj(
            cond_emb
        )[:, :, None, None]

        h = h + cond

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.act2(h)

        return (
            h + residual
        ) / math.sqrt(2)


# ============================================================
# 9 - CONDITIONAL DOWN
# ============================================================

class ConditionalDown(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        time_dim
    ):

        super().__init__()

        self.block1 = ConditionalResidualBlock(
            in_ch,
            out_ch,
            time_dim
        )

        self.block2 = ConditionalResidualBlock(
            out_ch,
            out_ch,
            time_dim
        )

        self.pool = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x, cond_emb):

        x = self.block1(x, cond_emb)
        x = self.block2(x, cond_emb)

        return x, self.pool(x)


# ============================================================
# 10 - CONDITIONAL UP
# ============================================================

class ConditionalUp(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        time_dim
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block1 = ConditionalResidualBlock(
            out_ch + skip_ch,
            out_ch,
            time_dim
        )

        self.block2 = ConditionalResidualBlock(
            out_ch,
            out_ch,
            time_dim
        )

    def forward(self, x, skip, cond_emb):

        x = self.up(x)

        x = torch.cat(
            (x, skip),
            dim=1
        )

        x = self.block1(x, cond_emb)
        x = self.block2(x, cond_emb)

        return x


# ============================================================
# 11 - U-NET CONDICIONAL
# ============================================================

class ConditionalUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        num_classes=3,
        time_dim=256,
        label_dim=128,
        dropout_prob=0.1
    ):

        super().__init__()

        # Classes reais: X, Y, Z
        self.num_classes = num_classes

        # Classe especial sem condicao
        self.null_label = num_classes

        self.dropout_prob = dropout_prob

        # 3 classes + 1 classe nula
        self.label_embedding = LabelEmbedding(
            num_classes + 1,
            label_dim
        )

        self.time_embedding = ConditionalTimeEmbedding(
            time_dim,
            label_dim
        )

        # Encoder
        self.down1 = ConditionalDown(
            in_channels,
            base_channels,
            time_dim
        )

        self.down2 = ConditionalDown(
            base_channels,
            base_channels * 2,
            time_dim
        )

        self.down3 = ConditionalDown(
            base_channels * 2,
            base_channels * 4,
            time_dim
        )

        self.down4 = ConditionalDown(
            base_channels * 4,
            base_channels * 8,
            time_dim
        )

        # Bottleneck
        self.bot1 = ConditionalResidualBlock(
            base_channels * 8,
            base_channels * 16,
            time_dim
        )

        self.bot2 = ConditionalResidualBlock(
            base_channels * 16,
            base_channels * 8,
            time_dim
        )

        # Decoder
        self.up4 = ConditionalUp(
            base_channels * 8,
            base_channels * 8,
            base_channels * 4,
            time_dim
        )

        self.up3 = ConditionalUp(
            base_channels * 4,
            base_channels * 4,
            base_channels * 2,
            time_dim
        )

        self.up2 = ConditionalUp(
            base_channels * 2,
            base_channels * 2,
            base_channels,
            time_dim
        )

        # Precisa retornar 64 canais,
        # pois self.out recebe base_channels.
        self.up1 = ConditionalUp(
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

    def forward(self, x, t, labels=None):

        if labels is None:

            # Inferencia incondicional:
            # utiliza a classe especial 3
            ids = torch.full(
                (x.size(0),),
                self.null_label,
                dtype=torch.long,
                device=x.device
            )

        else:

            ids = labels.to(
                device=x.device,
                dtype=torch.long
            ).clone()

            if self.training:

                # Classifier-Free Guidance:
                # remove a condicao em 10%
                # das amostras do treinamento.
                mask = (
                    torch.rand(
                        ids.size(0),
                        device=x.device
                    )
                    < self.dropout_prob
                )

                ids[mask] = self.null_label

        label_emb = self.label_embedding(ids)

        cond_emb = self.time_embedding(
            t,
            label_emb
        )

        # Encoder
        s1, p1 = self.down1(x, cond_emb)
        s2, p2 = self.down2(p1, cond_emb)
        s3, p3 = self.down3(p2, cond_emb)
        s4, p4 = self.down4(p3, cond_emb)

        # Bottleneck
        b = self.bot1(p4, cond_emb)
        b = self.bot2(b, cond_emb)

        # Decoder
        u4 = self.up4(b, s4, cond_emb)
        u3 = self.up3(u4, s3, cond_emb)
        u2 = self.up2(u3, s2, cond_emb)
        u1 = self.up1(u2, s1, cond_emb)

        return self.out(u1)


# ============================================================
# 12 - MODELO E OTIMIZADOR
# ============================================================

print("\n3 - CONSTRUINDO U-NET CONDICIONAL")

model = ConditionalUNet(
    in_channels=1,
    base_channels=64,
    num_classes=dataset.num_classes,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Parametros:", parametros)
print("Classes reais:", dataset.num_classes)
print("Classe nula:", model.null_label)
print("AdamW:", 0.0001)


# ============================================================
# 13 - TESTAR X, Y, Z E SEM CONDICAO
# ============================================================

print("\n4 - TESTANDO FORWARD CONDICIONAL")

indices_teste = [
    dataset.slice_planes.index(classe)
    for classe in ("x", "y", "z")
]

x_sample = torch.stack([
    dataset[i]["image"]
    for i in indices_teste
]).to(device)

labels = torch.tensor(
    [
        dataset.label_to_idx[classe]
        for classe in ("x", "y", "z")
    ],
    device=device,
    dtype=torch.long
)

t = torch.full(
    (3,),
    500,
    device=device,
    dtype=torch.long
)

model.eval()

with torch.no_grad():

    pred_cond = model(
        x_sample,
        t,
        labels
    )

    pred_sem = model(
        x_sample[:1],
        t[:1],
        labels=None
    )

assert pred_cond.shape == (
    3, 1, 64, 64
)

assert pred_sem.shape == (
    1, 1, 64, 64
)

assert torch.isfinite(pred_cond).all()
assert torch.isfinite(pred_sem).all()

assert (
    model.label_embedding.embedding.num_embeddings
    == 4
)

print(
    "Condicional X/Y/Z:",
    tuple(pred_cond.shape),
    "OK"
)

print(
    "Sem condicao:",
    tuple(pred_sem.shape),
    "OK"
)

print("Classe nula reservada: OK")


# ============================================================
# 14 - TORCHVIEW
# ============================================================

print("\n5 - VISUALIZANDO ARQUITETURA")


class WrappedUNet(nn.Module):

    def __init__(self, unet, label_idx):

        super().__init__()

        self.unet = unet
        self.label_idx = label_idx

    def forward(self, x):

        tempo = torch.full(
            (x.size(0),),
            500,
            dtype=torch.long,
            device=x.device
        )

        rotulos = torch.full(
            (x.size(0),),
            self.label_idx,
            dtype=torch.long,
            device=x.device
        )

        return self.unet(
            x,
            tempo,
            rotulos
        )


torchview_status = "nao executado"

try:

    wrapped_model = WrappedUNet(
        model,
        dataset.label_to_idx["z"]
    )

    graph = draw_graph(
        wrapped_model,
        input_data=x_sample[:1],
        graph_name="UNet Condicional Brain MRI",
        roll=True
    )

    arquivo_dot = (
        SAIDA
        / "aula15_conditional_unet_torchview.dot"
    )

    arquivo_dot.write_text(
        graph.visual_graph.source,
        encoding="utf-8"
    )

    torchview_status = "DOT gerado"

    print("TorchView DOT: OK")

    if shutil.which("dot"):

        graph.visual_graph.render(
            filename=str(
                SAIDA
                / "aula15_conditional_unet_torchview"
            ),
            format="png",
            cleanup=True
        )

        torchview_status = "PNG renderizado"
        print("TorchView PNG: OK")

    else:

        print(
            "Graphviz dot indisponivel. "
            "Arquivo DOT salvo."
        )

except Exception as erro:

    torchview_status = (
        f"Falha: {type(erro).__name__}: {erro}"
    )

    print(
        "TorchView nao renderizou:",
        type(erro).__name__,
        str(erro)
    )


# ============================================================
# 15 - SALVAR RESULTADOS
# ============================================================

print("\n6 - SALVANDO RESULTADOS")

arquivo_csv = (
    SAIDA
    / "aula15_distribuicao_classes.csv"
)

with arquivo_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    writer = csv.writer(arquivo)

    writer.writerow([
        "plano",
        "label",
        "quantidade"
    ])

    for classe in ("x", "y", "z"):

        writer.writerow([
            classe,
            dataset.label_to_idx[classe],
            counts[classe]
        ])


arquivo_json = (
    SAIDA
    / "aula15_unet_condicional_config.json"
)

dados = {
    "aula": 15,
    "dataset": "ashfakyeafi/brain-mri-images",
    "imagens": len(dataset),
    "label_to_idx": dataset.label_to_idx,
    "quantidade_por_classe": dict(counts),
    "classe_sem_condicao": model.null_label,
    "parametros": parametros,
    "image_shape": [1, 64, 64],
    "learning_rate": 0.0001,
    "dropout_prob": 0.1,
    "torchview": torchview_status,
    "treinamento_executado": False
}

arquivo_json.write_text(
    json.dumps(
        dados,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 16 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

for arquivo in [
    arquivo_grid,
    arquivo_csv,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("Dataset completo X/Y/Z: OK")
print("Labels numericas: OK")
print("Transformacoes 64x64: OK")
print("DataLoader: OK")
print("LabelEmbedding: OK")
print("ConditionalTimeEmbedding: OK")
print("ConditionalResidualBlock: OK")
print("ConditionalDown: OK")
print("ConditionalUp: OK")
print("U-Net condicional: OK")
print("Classe nula para CFG: OK")
print("Forward condicional: OK")
print("Forward incondicional: OK")
print("AdamW: OK")
print("TorchView:", torchview_status)
print("CSV e JSON: OK")

print("\nTREINAMENTO EXECUTADO: NAO")
print("AULA 15 - ARQUITETURA VALIDADA!")
