# ============================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 09 - PREPARACAO DO DATASET PARA U-NET
# ============================================================

from pathlib import Path
import csv
import json
import random
import re

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchview import draw_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "pixel-art"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("AULA 09 - U-NET: PREPARACAO DO DATASET")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("Dispositivo:", device)
print("Torchview: importado")


# ============================================================
# 2 - TRANSFORMACOES E DATA AUGMENTATION
# ============================================================

print("\n1 - CONFIGURANDO DATA AUGMENTATION")


def random_horizontal_flip(image, p=0.5):

    if random.random() < p:
        return image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    return image


def ajustar_hue(image, fator):

    hsv = np.array(
        image.convert("HSV"),
        dtype=np.uint8
    )

    deslocamento = int(
        round(fator * 255)
    )

    hsv[:, :, 0] = (
        (
            hsv[:, :, 0].astype(np.int16)
            + deslocamento
        ) % 256
    ).astype(np.uint8)

    return Image.fromarray(
        hsv
    ).convert("RGB")


def color_jitter(image):

    # Valores equivalentes aos da transcricao:
    # brightness=0.2
    # contrast=0.2
    # saturation=0.2
    # hue=0.1

    brilho = random.uniform(0.8, 1.2)
    contraste = random.uniform(0.8, 1.2)
    saturacao = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.1, 0.1)

    operacoes = [
        lambda img: ImageEnhance.Brightness(
            img
        ).enhance(brilho),

        lambda img: ImageEnhance.Contrast(
            img
        ).enhance(contraste),

        lambda img: ImageEnhance.Color(
            img
        ).enhance(saturacao),

        lambda img: ajustar_hue(img, hue)
    ]

    random.shuffle(operacoes)

    for operacao in operacoes:
        image = operacao(image)

    return image


def transform(image):

    # RandomHorizontalFlip
    image = random_horizontal_flip(image)

    # ColorJitter
    image = color_jitter(image)

    # Resize
    image = image.resize(
        (16, 16),
        Image.Resampling.BILINEAR
    )

    # ToTensor
    valores = (
        np.asarray(
            image,
            dtype=np.float32
        ).copy() / 255.0
    )

    tensor = torch.from_numpy(
        valores
    ).permute(2, 0, 1)

    # Normalize([0.5]*3, [0.5]*3)
    tensor = (
        tensor - 0.5
    ) / 0.5

    return tensor


print("RandomHorizontalFlip: OK")
print("ColorJitter: OK")
print("Resize 16x16: OK")
print("Normalize [-1,1]: OK")


# ============================================================
# 3 - DATASET COM FILTRO DE LABEL
# ============================================================

print("\n2 - CRIANDO PIXELARTDATASET")


class PixelArtDataset(Dataset):

    def __init__(
        self,
        images_dir,
        captions_file=None,
        transform=None,
        filter_label_value=None,
        filter_index=0
    ):

        self.transform = transform
        self.captions = {}

        self.filter_label_value = filter_label_value
        self.filter_index = filter_index

        extensoes = (
            ".png", ".jpg", ".jpeg", ".bmp"
        )

        self.image_files = sorted(
            caminho
            for caminho in Path(images_dir).rglob("*")
            if caminho.is_file()
            and caminho.suffix.lower() in extensoes
        )

        self.total_original = len(
            self.image_files
        )

        if captions_file:

            df = pd.read_csv(
                captions_file
            )

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

        # Filtrar apenas imagens da classe escolhida
        if filter_label_value is not None:

            filtered = []

            for img_path in self.image_files:

                img_name = self.force_extension(
                    img_path.name.lower(),
                    ".png"
                )

                label = self.captions.get(
                    img_name
                )

                if (
                    label is not None
                    and len(label) > filter_index
                    and np.isclose(
                        label[filter_index],
                        filter_label_value
                    )
                ):

                    filtered.append(img_path)

            self.image_files = filtered

    def __len__(self):

        return len(self.image_files)

    @staticmethod
    def force_extension(
        filename,
        ext=".png"
    ):

        return Path(filename).stem + ext

    def __getitem__(self, idx):

        img_path = self.image_files[idx]

        img_name = self.force_extension(
            img_path.name.lower(),
            ".png"
        )

        with Image.open(img_path) as arquivo:

            image = arquivo.convert("RGB")

        if self.transform:

            image = self.transform(image)

        label = self.captions.get(
            img_name,
            [0.0] * 5
        )

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.float32
            )
        }


# ============================================================
# 4 - CARREGAR DATASET REAL
# ============================================================

print("\n3 - CARREGANDO DATASET PIXELART")

dataset = PixelArtDataset(
    images_dir=DADOS,
    captions_file=DADOS / "labels.csv",
    transform=transform,
    filter_label_value=1.0,
    filter_index=3
)

print(
    "Imagens originais:",
    dataset.total_original
)

print(
    "Rotulos identificados:",
    len(dataset.captions)
)

print(
    "Imagens apos filtro:",
    len(dataset)
)

print(
    "Filtro: label[3] == 1.0"
)

if len(dataset) == 0:

    raise SystemExit(
        "Nenhuma imagem corresponde ao filtro. "
        "Verifique os valores do labels.csv."
    )

# Validar o filtro em todas as imagens selecionadas
for caminho in dataset.image_files:

    nome = dataset.force_extension(
        caminho.name.lower()
    )

    label = dataset.captions[nome]

    assert np.isclose(
        label[3],
        1.0
    )

print("Filtro de labels: OK")


# ============================================================
# 5 - DATALOADER
# ============================================================

print("\n4 - CRIANDO DATALOADER")

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

imagens = batch["image"]
rotulos = batch["label"]

print(
    "Tensor de imagens:",
    tuple(imagens.shape)
)

print(
    "Tensor de labels:",
    tuple(rotulos.shape)
)

assert imagens.ndim == 4
assert imagens.shape[1:] == (3, 16, 16)
assert rotulos.shape[1] == 5

assert torch.isfinite(
    imagens
).all()

assert imagens.min().item() >= -1.00001
assert imagens.max().item() <= 1.00001

assert torch.allclose(
    rotulos[:, 3],
    torch.ones(
        len(rotulos)
    )
)

print("DataLoader: OK")


# ============================================================
# 6 - VISUALIZAR AMOSTRAS FILTRADAS
# ============================================================

print("\n5 - VISUALIZANDO AMOSTRAS FILTRADAS")

quantidade = min(
    16,
    len(dataset)
)

indices = np.random.choice(
    len(dataset),
    size=quantidade,
    replace=False
)

cols = 8
rows = (
    quantidade + cols - 1
) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, rows * 3),
    squeeze=False
)

for i, idx in enumerate(indices):

    row, col = divmod(i, cols)

    sample = dataset[int(idx)]

    imagem = sample["image"]

    vis = torch.clamp(
        imagem * 0.5 + 0.5,
        0,
        1
    )

    vis = vis.permute(
        1, 2, 0
    ).cpu().numpy()

    ax = axes[row, col]

    ax.imshow(
        vis,
        interpolation="nearest"
    )

    ax.axis("off")

    ax.set_title(
        f"Indice #{idx}\n"
        f"label[3]={sample['label'][3].item():.1f}",
        fontsize=8
    )

for i in range(quantidade, rows * cols):

    row, col = divmod(i, cols)

    axes[row, col].axis("off")

fig.suptitle(
    "PixelArt - Amostras filtradas e aumentadas",
    fontsize=15
)

fig.tight_layout()

arquivo_grid = (
    RESULTADOS
    / "aula09_pixelart_filtrado.png"
)

fig.savefig(
    arquivo_grid,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("Grid de amostras: OK")


# ============================================================
# 7 - COMPARAR ORIGINAL E AUGMENTATIONS
# ============================================================

print("\n6 - COMPARANDO DATA AUGMENTATION")

caminho = dataset.image_files[0]

with Image.open(caminho) as arq:

    original = arq.convert(
        "RGB"
    ).resize(
        (16, 16),
        Image.Resampling.BILINEAR
    )

fig, axes = plt.subplots(
    1,
    5,
    figsize=(13, 3)
)

axes[0].imshow(
    original,
    interpolation="nearest"
)

axes[0].set_title("Original")
axes[0].axis("off")

for i in range(1, 5):

    tensor = transform(
        original.copy()
    )

    imagem = (
        (tensor * 0.5 + 0.5)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    axes[i].imshow(
        imagem,
        interpolation="nearest"
    )

    axes[i].set_title(
        f"Augmentation {i}"
    )

    axes[i].axis("off")

fig.tight_layout()

arquivo_augmentation = (
    RESULTADOS
    / "aula09_data_augmentation.png"
)

fig.savefig(
    arquivo_augmentation,
    dpi=150
)

plt.close(fig)

print("Comparacao de augmentation: OK")


# ============================================================
# 8 - SALVAR RESULTADOS
# ============================================================

print("\n7 - SALVANDO METADADOS")

arquivo_json = (
    RESULTADOS
    / "aula09_dataset_filtrado.json"
)

dados = {
    "aula": 9,
    "tema": "UNet - preparacao do dataset",
    "seed": SEED,
    "device": device,
    "total_original": dataset.total_original,
    "total_filtrado": len(dataset),
    "filter_label_value": 1.0,
    "filter_index": 3,
    "quantidade_rotulos": len(dataset.captions),
    "batch_size": 8,
    "shuffle": True,
    "image_shape": [3, 16, 16],
    "normalizacao": [-1, 1],
    "augmentations": [
        "RandomHorizontalFlip",
        "ColorJitter",
        "Resize",
        "ToTensor",
        "Normalize"
    ]
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arq:

    json.dump(
        dados,
        arq,
        indent=4,
        ensure_ascii=False
    )


arquivo_csv = (
    RESULTADOS
    / "aula09_amostras_filtradas.csv"
)

with arquivo_csv.open(
    "w",
    encoding="utf-8",
    newline=""
) as arq:

    writer = csv.writer(arq)

    writer.writerow([
        "indice_dataset",
        "nome_arquivo",
        "label"
    ])

    for i, caminho in enumerate(
        dataset.image_files[:100]
    ):

        nome = dataset.force_extension(
            caminho.name.lower()
        )

        writer.writerow([
            i,
            caminho.name,
            json.dumps(
                dataset.captions[nome]
            )
        ])


# ============================================================
# 9 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert dataset.total_original > 0
assert len(dataset) > 0
assert len(dataset) <= dataset.total_original

assert imagens.shape[1:] == (3, 16, 16)
assert rotulos.shape[1] == 5

for arquivo in [
    arquivo_grid,
    arquivo_augmentation,
    arquivo_json,
    arquivo_csv
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("PyTorch: OK")
print("Torchview: OK")
print("Dataset PixelArt: OK")
print("Filtro label[3] == 1.0: OK")
print("PixelArtDataset: OK")
print("RandomHorizontalFlip: OK")
print("ColorJitter: OK")
print("Resize 16x16: OK")
print("Normalizacao [-1,1]: OK")
print("DataLoader: OK")
print("Grid de amostras: OK")
print("Comparacao de augmentation: OK")
print("CSV e JSON: OK")

print("\nAULA 09 EXECUTADA COM SUCESSO!")
