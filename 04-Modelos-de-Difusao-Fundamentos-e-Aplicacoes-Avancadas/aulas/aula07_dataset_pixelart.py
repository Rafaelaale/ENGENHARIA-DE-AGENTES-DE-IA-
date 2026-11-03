# ============================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 07 - DATASET E DATALOADER PIXELART
# ============================================================

from pathlib import Path
import argparse
import json
import re
import textwrap

import kagglehub
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "pixel-art"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--baixar",
    action="store_true",
    help="Baixar o dataset real do Kaggle, se necessario."
)

args = parser.parse_args()

print("=" * 65)
print("AULA 07 - DATASET PIXELART")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("Dispositivo:", device)
print("Projeto:", BASE)


# ============================================================
# 2 - LOCALIZAR OU BAIXAR O DATASET
# ============================================================

print("\n1 - LOCALIZANDO DATASET")


def encontrar_imagens(pasta):

    return sorted(
        p for p in pasta.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSOES
    )


arquivos_existentes = encontrar_imagens(DADOS)

if not arquivos_existentes:

    if not args.baixar:
        raise SystemExit(
            "Dataset ainda nao existe. "
            "Execute este script com --baixar."
        )

    print("Baixando dataset do Kaggle...")
    print("Origem: ebrahimelgazar/pixel-art")

    try:

        caminho_download = kagglehub.dataset_download(
            "ebrahimelgazar/pixel-art",
            output_dir=str(DADOS)
        )

        print(
            "Download retornou:",
            caminho_download
        )

    except Exception as erro:

        print(
            "ERRO NO DOWNLOAD:",
            type(erro).__name__,
            str(erro)
        )

        print(
            "Verifique o acesso ao dataset em:"
        )

        print(
            "https://www.kaggle.com/datasets/"
            "ebrahimelgazar/pixel-art"
        )

        raise SystemExit(2)

arquivos_existentes = encontrar_imagens(DADOS)

if not arquivos_existentes:

    raise SystemExit(
        "Nenhuma imagem foi encontrada em dados/pixel-art. "
        "Verifique a estrutura baixada."
    )

print("Total de imagens encontradas:", len(arquivos_existentes))
print("Pasta dos dados:", DADOS)


# ============================================================
# 3 - LOCALIZAR CSV DE ROTULOS
# ============================================================

print("\n2 - LOCALIZANDO ROTULOS")

arquivos_csv = sorted(DADOS.rglob("*.csv"))

captions_file = None

for arquivo in arquivos_csv:

    try:
        colunas = pd.read_csv(
            arquivo,
            nrows=0
        ).columns

        if (
            "Image Path" in colunas
            and "Label" in colunas
        ):
            captions_file = arquivo
            break

    except Exception:
        continue

if captions_file:

    print("CSV encontrado:", captions_file)

else:

    print(
        "CSV com colunas Image Path e Label "
        "nao encontrado."
    )

    print(
        "Rotulos ausentes serao identificados "
        "com vetor de zeros."
    )


# ============================================================
# 4 - CRIAR DATASET CUSTOMIZADO
# ============================================================

print("\n3 - CRIANDO PIXELARTDATASET")


class PixelArtDataset(Dataset):

    def __init__(
        self,
        images_dir,
        captions_file=None,
        transform=None
    ):

        self.transform = transform
        self.captions = {}
        self.rotulos_invalidos = 0

        self.image_files = encontrar_imagens(
            Path(images_dir)
        )

        if captions_file:

            df = pd.read_csv(captions_file)

            for caminho, rotulo in zip(
                df["Image Path"],
                df["Label"]
            ):

                nome = self.force_extension(
                    Path(str(caminho)).name.lower(),
                    ".png"
                )

                texto = re.sub(
                    r"[\[\],]",
                    " ",
                    str(rotulo)
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

                else:

                    self.rotulos_invalidos += 1

    def __len__(self):

        return len(self.image_files)

    @staticmethod
    def force_extension(filename, ext=".png"):

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
# 5 - TRANSFORMACOES: 16x16 E [-1,1]
# ============================================================

print("\n4 - CONFIGURANDO TRANSFORMACOES")


def transformar_imagem(image):

    # Equivalente ao Resize((16,16))
    image = image.resize(
        (16, 16),
        resample=Image.Resampling.BILINEAR
    )

    # Equivalente ao ToTensor()
    imagem_np = np.asarray(
        image,
        dtype=np.float32
    ).copy() / 255.0

    imagem_tensor = torch.from_numpy(
        imagem_np
    ).permute(2, 0, 1)

    # Equivalente ao Normalize([0.5]*3,[0.5]*3)
    imagem_tensor = (
        imagem_tensor - 0.5
    ) / 0.5

    return imagem_tensor


dataset = PixelArtDataset(
    images_dir=DADOS,
    captions_file=captions_file,
    transform=transformar_imagem
)

print("Dataset criado: OK")
print("Total de imagens:", len(dataset))
print("Rotulos no CSV:", len(dataset.captions))
print("Rotulos invalidos:", dataset.rotulos_invalidos)

assert len(dataset) > 0


# ============================================================
# 6 - CRIAR DATALOADER
# ============================================================

print("\n5 - CRIANDO DATALOADER")

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

imagens_batch = batch["image"]
labels_batch = batch["label"]

print(
    "Formato das imagens:",
    tuple(imagens_batch.shape)
)

print(
    "Formato dos rotulos:",
    tuple(labels_batch.shape)
)

assert imagens_batch.ndim == 4
assert imagens_batch.shape[1:] == (3, 16, 16)
assert labels_batch.shape[1] == 5

assert torch.isfinite(
    imagens_batch
).all()

assert imagens_batch.min() >= -1.00001
assert imagens_batch.max() <= 1.00001

print("DataLoader validado: OK")


# ============================================================
# 7 - VISUALIZAR 16 AMOSTRAS
# ============================================================

print("\n6 - VISUALIZANDO 16 AMOSTRAS")

indices = np.random.choice(
    len(dataset),
    size=min(16, len(dataset)),
    replace=False
)

cols = 8
rows = (len(indices) + cols - 1) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 4 * rows),
    squeeze=False
)

for i, idx in enumerate(indices):

    row, col = divmod(i, cols)

    ax = axes[row, col]

    sample = dataset[int(idx)]

    image = sample["image"]
    label = sample["label"]

    img_vis = torch.clamp(
        image * 0.5 + 0.5,
        0,
        1
    )

    img_np = img_vis.permute(
        1,
        2,
        0
    ).cpu().numpy()

    ax.imshow(
        img_np,
        interpolation="nearest"
    )

    ax.axis("off")

    texto_label = textwrap.fill(
        str(label.tolist()),
        width=24
    )

    ax.set_title(
        f"#{idx}\n{texto_label}",
        fontsize=7
    )

for i in range(len(indices), rows * cols):

    row, col = divmod(i, cols)

    axes[row, col].axis("off")

fig.suptitle(
    "Amostras do Dataset PixelArt",
    fontsize=16
)

fig.tight_layout()

arquivo_grid = (
    RESULTADOS
    / "aula07_amostras_pixelart.png"
)

fig.savefig(
    arquivo_grid,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("Grid de imagens salvo: OK")


# ============================================================
# 8 - VISUALIZAR PRIMEIRO BATCH
# ============================================================

print("\n7 - VISUALIZANDO BATCH")

n = imagens_batch.shape[0]

fig, axes = plt.subplots(
    1,
    n,
    figsize=(2 * n, 3),
    squeeze=False
)

for i in range(n):

    img = torch.clamp(
        imagens_batch[i] * 0.5 + 0.5,
        0,
        1
    )

    img = img.permute(
        1,
        2,
        0
    ).numpy()

    axes[0, i].imshow(
        img,
        interpolation="nearest"
    )

    axes[0, i].set_title(
        f"Imagem {i + 1}"
    )

    axes[0, i].axis("off")

fig.tight_layout()

arquivo_batch = (
    RESULTADOS
    / "aula07_batch_pixelart.png"
)

fig.savefig(
    arquivo_batch,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("Batch salvo: OK")


# ============================================================
# 9 - SALVAR METADADOS
# ============================================================

print("\n8 - SALVANDO METADADOS")

arquivo_json = (
    RESULTADOS
    / "aula07_dataset_configuracoes.json"
)

rotulos_associados = sum(
    dataset.force_extension(
        p.name.lower()
    ) in dataset.captions
    for p in dataset.image_files
)

dados = {
    "aula": 7,
    "tema": "Dataset e DataLoader PixelArt",
    "origem": "ebrahimelgazar/pixel-art",
    "total_imagens": len(dataset),
    "imagem_dimensoes": [3, 16, 16],
    "normalizacao": [-1, 1],
    "batch_size": 8,
    "shuffle": True,
    "rotulos_csv_encontrado": bool(captions_file),
    "rotulos_associados": rotulos_associados,
    "rotulos_invalidos": dataset.rotulos_invalidos,
    "amostras_visualizadas": len(indices),
    "dispositivo": device
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

print("JSON salvo: OK")


# ============================================================
# 10 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert len(dataset) > 0
assert imagens_batch.shape[1:] == (3, 16, 16)
assert labels_batch.shape[1] == 5

for arquivo in [
    arquivo_grid,
    arquivo_batch,
    arquivo_json
]:
    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("KaggleHub: OK")
print("Dataset real localizado: OK")
print("PixelArtDataset: OK")
print("__len__: OK")
print("__getitem__: OK")
print("Transformacoes 16x16: OK")
print("Normalizacao [-1,1]: OK")
print("DataLoader: OK")
print("Grid de amostras: OK")
print("Batch: OK")
print("Metadados JSON: OK")

print("\nAULA 07 EXECUTADA COM SUCESSO!")
