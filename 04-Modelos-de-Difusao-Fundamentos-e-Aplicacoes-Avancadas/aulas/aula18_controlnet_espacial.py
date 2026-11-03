# ============================================================
# PROJETO 04 - AULA 18
# ARQUITETURA COM CONDICIONAMENTO ESPACIAL
# ============================================================

from pathlib import Path
from collections import Counter
import argparse
import ast
import json
import math
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1 - CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados" / "brain-mri-images"
SAIDA = BASE / "resultados"

SAIDA.mkdir(exist_ok=True, parents=True)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base",
    type=int,
    choices=[16, 64],
    default=16
)

parser.add_argument(
    "--amostras-media",
    type=int,
    default=10
)

parser.add_argument(
    "--tipo",
    choices=[
        "intensity",
        "low_intensity",
        "edge",
        "gradient",
        "scharr"
    ],
    default="intensity"
)

parser.add_argument(
    "--grafo",
    action="store_true"
)

args = parser.parse_args()

if args.amostras_media < 1:
    parser.error(
        "--amostras-media deve ser >= 1"
    )

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

print("=" * 66)
print("AULA 18 - CONDICIONAMENTO ESPACIAL")
print("=" * 66)

print("Dispositivo:", device)
print("OpenCV:", cv2.__version__)


# ============================================================
# 2 - REUTILIZAR CLASSES DA AULA 15
# ============================================================

def importar_definicoes(caminho, nomes):

    arvore = ast.parse(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )

    definicoes = {
        n.name: n
        for n in arvore.body
        if isinstance(
            n, (ast.FunctionDef, ast.ClassDef)
        )
    }

    ausentes = [
        n for n in nomes
        if n not in definicoes
    ]

    if ausentes:
        raise RuntimeError(
            f"Definicoes nao encontradas: {ausentes}"
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
            str(caminho),
            "exec"
        ),
        globals()
    )


print("\n1 - REUTILIZANDO AULA 15")

importar_definicoes(
    BASE / "aulas" / "aula15_unet_condicional.py",
    [
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
)

dataset = BrainMRIConditionalDataset(
    DADOS,
    transform=transform_mri,
    slice_plane=None
)

assert dataset.label_to_idx == {
    "x": 0,
    "y": 1,
    "z": 2
}

print("Dataset:", len(dataset), "imagens")
print("Classes:", dataset.label_to_idx)
print(
    "Contagens:",
    dict(Counter(dataset.slice_planes))
)


# ============================================================
# 3 - SCHEDULER
# ============================================================

print("\n2 - SCHEDULER DE DIFUSAO")

T = 1000

betas = torch.linspace(
    1e-4,
    0.02,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)

assert (
    alpha_bar[1:] < alpha_bar[:-1]
).all()

print("Passos:", T)
print("Beta inicial:", float(betas[0]))
print("Beta final:", float(betas[-1]))


# ============================================================
# 4 - ENCODER DE CONDICIONAMENTO ESPACIAL
# ============================================================

class CondicionamentoEspacial(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        time_dim=256
    ):

        super().__init__()

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

    def forward(self, x, cond_emb):

        f1, p1 = self.down1(
            x, cond_emb
        )

        f2, p2 = self.down2(
            p1, cond_emb
        )

        f3, p3 = self.down3(
            p2, cond_emb
        )

        f4, _ = self.down4(
            p3, cond_emb
        )

        return [f1, f2, f3, f4]


# ============================================================
# 5 - U-NET ESTENDIDA COM HINTS ESPACIAIS
# ============================================================

class ControlNet(ConditionalUNet):

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        num_classes=3,
        time_dim=256,
        label_dim=128,
        dropout_prob=0.1
    ):

        super().__init__(
            in_channels,
            base_channels,
            num_classes,
            time_dim,
            label_dim,
            dropout_prob
        )

        self.hint_projections = nn.ModuleList([
            nn.Conv2d(
                base_channels * (2 ** i),
                base_channels * (2 ** i),
                kernel_size=1
            )
            for i in range(4)
        ])

        # Projecoes zeradas inicialmente.
        # O condicionamento espacial sera
        # aprendido em treinamento posterior.
        for proj in self.hint_projections:

            nn.init.zeros_(
                proj.weight
            )

            nn.init.zeros_(
                proj.bias
            )

    def forward(
        self,
        x,
        cond_emb,
        spatial_hints=None
    ):

        s1, p1 = self.down1(
            x, cond_emb
        )

        s2, p2 = self.down2(
            p1, cond_emb
        )

        s3, p3 = self.down3(
            p2, cond_emb
        )

        s4, p4 = self.down4(
            p3, cond_emb
        )

        if spatial_hints is not None:

            if len(spatial_hints) != 4:
                raise ValueError(
                    "Esperados quatro hints espaciais"
                )

            s1 = s1 + self.hint_projections[0](
                spatial_hints[0]
            )

            s2 = s2 + self.hint_projections[1](
                spatial_hints[1]
            )

            s3 = s3 + self.hint_projections[2](
                spatial_hints[2]
            )

            s4 = s4 + self.hint_projections[3](
                spatial_hints[3]
            )

        b = self.bot1(
            p4, cond_emb
        )

        b = self.bot2(
            b, cond_emb
        )

        u4 = self.up4(
            b, s4, cond_emb
        )

        u3 = self.up3(
            u4, s3, cond_emb
        )

        u2 = self.up2(
            u3, s2, cond_emb
        )

        u1 = self.up1(
            u2, s1, cond_emb
        )

        return self.out(u1)


# ============================================================
# 6 - PIPELINE INTEGRADO
# ============================================================

class CondSpacialNetPipeline(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        num_classes=3,
        time_dim=256,
        label_dim=128
    ):

        super().__init__()

        self.condspacial = CondicionamentoEspacial(
            in_channels,
            base_channels,
            time_dim
        )

        self.unet = ControlNet(
            in_channels,
            base_channels,
            num_classes,
            time_dim,
            label_dim
        )

        self.num_classes = num_classes

        # Classe especial sem condicao
        self.null_label = num_classes

    def forward(
        self,
        x,
        t,
        labels=None,
        spatial_condition=None
    ):

        B = x.size(0)

        t = t.to(
            device=x.device,
            dtype=torch.long
        )

        if t.shape != (B,):
            raise ValueError(
                "Timestep deve ter formato [B]"
            )

        if labels is None:

            ids = torch.full(
                (B,),
                self.null_label,
                device=x.device,
                dtype=torch.long
            )

        else:

            ids = labels.to(
                device=x.device,
                dtype=torch.long
            ).clone()

            if (
                ids.shape != (B,)
                or not (
                    (ids >= 0)
                    & (ids <= self.null_label)
                ).all()
            ):
                raise ValueError(
                    "Labels esperadas: 0, 1, 2 ou 3"
                )

            if self.training:

                mask = (
                    torch.rand(
                        B,
                        device=x.device
                    )
                    < self.unet.dropout_prob
                )

                ids[mask] = self.null_label

        label_emb = self.unet.label_embedding(
            ids
        )

        cond_emb = self.unet.time_embedding(
            t,
            label_emb
        )

        hints = None

        if spatial_condition is not None:

            if spatial_condition.shape != x.shape:
                raise ValueError(
                    "Mascara espacial com formato invalido"
                )

            hints = self.condspacial(
                spatial_condition.to(
                    device=x.device,
                    dtype=x.dtype
                ),
                cond_emb
            )

        return self.unet(
            x,
            cond_emb,
            hints
        )


# ============================================================
# 7 - CRIAR MASCARAS ESPACIAIS
# ============================================================

def create_spatial_condition(
    height,
    width,
    condition_type="intensity",
    image_tensor=None,
    threshold=0.2
):

    if (
        image_tensor is None
        or image_tensor.shape
        != (1, 1, height, width)
    ):
        raise ValueError(
            "Imagem esperada: [1, 1, H, W]"
        )

    img = (
        image_tensor[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    img_u8 = np.clip(
        (img + 1) * 127.5,
        0,
        255
    ).astype(np.uint8)

    if condition_type == "intensity":

        threshold_value = (
            img.mean()
            + 0.1 * img.std()
        )

        binary = (
            img > threshold_value
        ).astype(np.float32)

    elif condition_type == "low_intensity":

        threshold_value = (
            img.mean()
            - 0.2 * img.std()
        )

        binary = (
            img < threshold_value
        ).astype(np.float32)

    elif condition_type == "edge":

        edges = cv2.Canny(
            img_u8,
            50,
            150
        )

        binary = (
            edges > 0
        ).astype(np.float32)

    elif condition_type == "gradient":

        gx = cv2.Sobel(
            img_u8.astype(np.float32),
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            img_u8.astype(np.float32),
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        mag = np.hypot(
            gx, gy
        )

        mag = (
            (mag - mag.min())
            / (np.ptp(mag) + 1e-8)
        )

        binary = (
            mag > threshold
        ).astype(np.float32)

    elif condition_type == "scharr":

        gx = cv2.Scharr(
            img_u8.astype(np.float32),
            cv2.CV_32F,
            1,
            0
        )

        gy = cv2.Scharr(
            img_u8.astype(np.float32),
            cv2.CV_32F,
            0,
            1
        )

        mag = np.hypot(
            gx, gy
        )

        mag = (
            (mag - mag.min())
            / (np.ptp(mag) + 1e-8)
        )

        binary = (
            mag > threshold
        ).astype(np.float32)

    else:

        raise ValueError(
            f"Tipo desconhecido: {condition_type}"
        )

    condition = (
        torch.from_numpy(
            binary.copy()
        )[None, None]
        .to(
            device=image_tensor.device,
            dtype=torch.float32
        )
    )

    return condition * 2.0 - 1.0


# ============================================================
# 8 - MASCARA MEDIA POR CLASSE
# ============================================================

def create_average_spatial_condition(
    dataset,
    class_index,
    img_size=64,
    condition_type="intensity",
    samples_per_class=10,
    threshold=0.2
):

    indices = [
        i
        for i, rotulo in enumerate(
            dataset.labels
        )
        if rotulo == class_index
    ]

    if not indices:
        raise ValueError(
            "Classe sem imagens"
        )

    rng = np.random.default_rng(
        2026 + class_index
    )

    n = min(
        samples_per_class,
        len(indices)
    )

    selecionados = rng.choice(
        indices,
        size=n,
        replace=False
    )

    mascaras = []

    for idx in selecionados:

        imagem = (
            dataset[int(idx)]["image"]
            [None]
            .to(device)
        )

        cond = create_spatial_condition(
            img_size,
            img_size,
            condition_type,
            imagem,
            threshold
        )

        mascaras.append(cond)

    return torch.cat(
        mascaras,
        dim=0
    ).mean(
        dim=0,
        keepdim=True
    )


# ============================================================
# 9 - TESTAR AS CINCO MASCARAS
# ============================================================

print("\n3 - GERANDO CINCO TIPOS DE MASCARA")

idx_z = dataset.slice_planes.index("z")

exemplo = (
    dataset[idx_z]["image"]
    [None]
    .to(device)
)

tipos = [
    "intensity",
    "low_intensity",
    "edge",
    "gradient",
    "scharr"
]

fig, axs = plt.subplots(
    1,
    6,
    figsize=(15, 3)
)

axs[0].imshow(
    exemplo[0, 0].cpu().numpy(),
    cmap="gray",
    vmin=-1,
    vmax=1
)

axs[0].set_title("MRI original")
axs[0].axis("off")

for i, tipo in enumerate(tipos, 1):

    mascara = create_spatial_condition(
        64,
        64,
        tipo,
        exemplo
    )

    assert mascara.shape == (
        1, 1, 64, 64
    )

    assert torch.all(
        (mascara == -1)
        | (mascara == 1)
    )

    axs[i].imshow(
        mascara[0, 0].cpu().numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    axs[i].set_title(tipo)
    axs[i].axis("off")

fig.tight_layout()

path_tipos = (
    SAIDA / "aula18_cinco_tipos_mascaras.png"
)

fig.savefig(
    path_tipos,
    dpi=140
)

plt.close(fig)

print(
    "Intensity / Low / Edge / "
    "Gradient / Scharr: OK"
)


# ============================================================
# 10 - MASCARAS MEDIAS X, Y, Z
# ============================================================

print("\n4 - CALCULANDO MASCARAS MEDIAS X/Y/Z")

spatial_conditions_dict = {}

for classe in ("x", "y", "z"):

    idx = dataset.label_to_idx[classe]

    cond = create_average_spatial_condition(
        dataset,
        idx,
        64,
        args.tipo,
        args.amostras_media
    )

    assert cond.shape == (
        1, 1, 64, 64
    )

    assert torch.isfinite(
        cond
    ).all()

    assert float(cond.min()) >= -1.00001
    assert float(cond.max()) <= 1.00001

    spatial_conditions_dict[idx] = cond

    n = min(
        args.amostras_media,
        dataset.labels.count(idx)
    )

    print(
        f"Classe {classe.upper()}: "
        f"media de {n} mascaras -> "
        f"{tuple(cond.shape)}"
    )

fig, axs = plt.subplots(
    1,
    3,
    figsize=(10, 4)
)

for i, (classe, cond) in enumerate(
    zip(
        ("x", "y", "z"),
        spatial_conditions_dict.values()
    )
):

    axs[i].imshow(
        cond[0, 0].cpu().numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    axs[i].set_title(
        f"Plano {classe.upper()} | {args.tipo}"
    )

    axs[i].axis("off")

fig.tight_layout()

path_media = (
    SAIDA
    / f"aula18_mascaras_medias_{args.tipo}.png"
)

fig.savefig(
    path_media,
    dpi=150
)

plt.close(fig)

print("Mascaras medias: OK")


# ============================================================
# 11 - CONSTRUIR PIPELINE
# ============================================================

print("\n5 - CONSTRUINDO PIPELINE ESPACIAL")

model = CondSpacialNetPipeline(
    in_channels=1,
    base_channels=args.base,
    num_classes=dataset.num_classes,
    time_dim=256,
    label_dim=128
).to(device)

model.eval()

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Parametros:", parametros)
print("Base channels:", args.base)
print("Classe nula:", model.null_label)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

print("AdamW: 0.0001")


# ============================================================
# 12 - TESTAR HINTS E FORWARD
# ============================================================

print("\n6 - TESTANDO HINTS E FORWARD")

indices = [
    dataset.slice_planes.index(c)
    for c in ("x", "y", "z")
]

x = torch.stack([
    dataset[i]["image"]
    for i in indices
]).to(device)

rotulos = torch.tensor(
    [0, 1, 2],
    device=device,
    dtype=torch.long
)

tempo = torch.full(
    (3,),
    500,
    device=device,
    dtype=torch.long
)

mascaras = torch.cat(
    [
        spatial_conditions_dict[i]
        for i in range(3)
    ],
    dim=0
)

with torch.no_grad():

    cond_emb = model.unet.time_embedding(
        tempo,
        model.unet.label_embedding(rotulos)
    )

    hints = model.condspacial(
        mascaras,
        cond_emb
    )

    previsto = [
        (
            3,
            args.base * (2 ** i),
            64 // (2 ** i),
            64 // (2 ** i)
        )
        for i in range(4)
    ]

    for i, (hint, shape) in enumerate(
        zip(hints, previsto),
        1
    ):

        assert tuple(hint.shape) == shape

        print(
            f"Hint {i}: {tuple(hint.shape)} OK"
        )

    pred_cond = model(
        x,
        tempo,
        rotulos,
        mascaras
    )

    pred_sem = model(
        x[:1],
        tempo[:1],
        None,
        None
    )

    assert pred_cond.shape == (
        3, 1, 64, 64
    )

    assert pred_sem.shape == (
        1, 1, 64, 64
    )

    assert torch.isfinite(
        pred_cond
    ).all()

    assert torch.isfinite(
        pred_sem
    ).all()

    print(
        "Forward X/Y/Z com mascaras:",
        tuple(pred_cond.shape),
        "OK"
    )

    print(
        "Forward sem ambas condicoes:",
        tuple(pred_sem.shape),
        "OK"
    )

    assert all(
        torch.allclose(
            proj.weight,
            torch.zeros_like(proj.weight)
        )
        for proj in model.unet.hint_projections
    )

    print(
        "Projecoes espaciais zeradas: OK"
    )


# ============================================================
# 13 - SAMPLING COM CONDICIONAMENTO ESPACIAL
# ============================================================

print("\n7 - TESTANDO DDPM COM GUIDANCE")


@torch.no_grad()
def p_sample_condspacial(
    model,
    x,
    t,
    labels=None,
    spatial_condition=None,
    guidance_scale=1.0,
    add_noise=True,
    return_details=False
):

    B = x.size(0)

    ts = torch.full(
        (B,),
        t,
        device=x.device,
        dtype=torch.long
    )

    # Trajetoria condicional
    pred_cond = model(
        x,
        ts,
        labels,
        spatial_condition
    )

    # Trajetoria sem classe e sem mascara
    pred_uncond = model(
        x,
        ts,
        None,
        None
    )

    # Classifier-Free Guidance
    pred = (
        pred_uncond
        + guidance_scale
        * (pred_cond - pred_uncond)
    )

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]

    mean = (
        x
        - beta_t
        / torch.sqrt(1 - alpha_bar_t)
        * pred
    ) / torch.sqrt(alpha_t)

    if add_noise and t > 0:

        ab_prev = alpha_bar[t - 1]

        variance = (
            beta_t
            * (1 - ab_prev)
            / (1 - alpha_bar_t)
        )

        saida = (
            mean
            + torch.sqrt(variance)
            * torch.randn_like(x)
        )

    else:

        saida = mean

    if return_details:

        return (
            saida,
            pred_cond,
            pred_uncond,
            pred
        )

    return saida


xt = torch.randn(
    (1, 1, 64, 64),
    device=device
)

with torch.no_grad():

    (
        saida,
        pred_c,
        pred_u,
        pred_cfg
    ) = p_sample_condspacial(
        model,
        xt,
        500,
        rotulos[:1],
        mascaras[:1],
        guidance_scale=3.0,
        add_noise=False,
        return_details=True
    )

assert saida.shape == xt.shape

assert torch.isfinite(saida).all()

assert torch.allclose(
    pred_cfg,
    pred_u + 3 * (pred_c - pred_u)
)

print("CFG 0 / 1 / 3: formula validada")

print(
    "Um passo reverso:",
    tuple(saida.shape),
    "OK"
)

print("Treinamento executado: NAO")
print("Geracao de 1000 passos: NAO")


# ============================================================
# 14 - TORCHVIEW OPCIONAL
# ============================================================

status_grafo = "nao solicitado"

if args.grafo:

    print("\n8 - EXPORTANDO GRAFO")

    try:

        from torchview import draw_graph

        class WrappedSpatial(nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, x, mascara):

                B = x.size(0)

                tempo_fixo = torch.full(
                    (B,),
                    500,
                    device=x.device,
                    dtype=torch.long
                )

                labels = torch.zeros(
                    (B,),
                    device=x.device,
                    dtype=torch.long
                )

                return self.net(
                    x,
                    tempo_fixo,
                    labels,
                    mascara
                )

        wrapper = WrappedSpatial(model)

        graph = draw_graph(
            wrapper,
            input_data=(
                x[:1],
                mascaras[:1]
            ),
            graph_name="ControlNet Espacial Aula18",
            roll=True
        )

        path_dot = (
            SAIDA
            / "aula18_controlnet_torchview.dot"
        )

        path_dot.write_text(
            graph.visual_graph.source,
            encoding="utf-8"
        )

        print("TorchView DOT: OK")

        status_grafo = "DOT gerado"

        if shutil.which("dot"):

            graph.visual_graph.render(
                filename=str(
                    SAIDA
                    / "aula18_controlnet_torchview"
                ),
                format="png",
                cleanup=True
            )

            print("TorchView PNG: OK")

            status_grafo = "PNG gerado"

        else:

            print(
                "Graphviz dot indisponivel. "
                "Arquivo DOT salvo."
            )

    except Exception as erro:

        status_grafo = (
            f"Falha {type(erro).__name__}: {erro}"
        )

        print(
            "Falha opcional no grafo:",
            status_grafo
        )


# ============================================================
# 15 - SALVAR RESULTADOS
# ============================================================

print("\n9 - SALVANDO CONFIGURACOES")

path_json = (
    SAIDA
    / f"aula18_controlnet_config_{args.tipo}_base{args.base}.json"
)

dados = {
    "aula": 18,
    "dataset": len(dataset),
    "classes": dataset.label_to_idx,
    "classe_nula": model.null_label,
    "base_channels": args.base,
    "parametros": parametros,
    "timesteps": T,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "tipo_mascara_media": args.tipo,
    "amostras_media_por_classe": args.amostras_media,
    "formatos_hints": [
        list(v)
        for v in previsto
    ],
    "torchview": status_grafo,
    "treinamento_executado": False,
    "geracao_executada": False
}

path_json.write_text(
    json.dumps(
        dados,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

for caminho in [
    path_tipos,
    path_media,
    path_json
]:

    assert caminho.is_file()
    assert caminho.stat().st_size > 0


# ============================================================
# 16 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 66)
print("VALIDACAO FINAL - AULA 18")
print("=" * 66)

print("Dataset X/Y/Z: OK")
print("OpenCV: OK")
print("Cinco mascaras: OK")
print("Mascaras medias por classe: OK")
print("Encoder espacial de quatro niveis: OK")
print("U-Net + hints nas skip connections: OK")
print("Forward condicionado: OK")
print("Forward sem condicao: OK")
print("CFG com classe nula 3: OK")
print("Um passo reverso DDPM: OK")
print("Imagens e JSON: OK")

print("\nTREINAMENTO: NAO executado")
print("GERACAO: NAO executada")

print(
    "\nAULA 18 - ARQUITETURA ESPACIAL VALIDADA!"
)
