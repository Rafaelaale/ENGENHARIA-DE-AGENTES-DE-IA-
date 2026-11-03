# ============================================================
# PROJETO 04 - AULA 13
# FEATURE MAPS E PROCESSO DE DIFUSAO
# ============================================================

from pathlib import Path
from collections import defaultdict
import ast
import csv
import json
import math
import os
import random

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

SAIDA.mkdir(parents=True, exist_ok=True)

EXTENSOES = {
    ".png", ".jpg", ".jpeg", ".bmp"
}

img_size = 64

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
print("AULA 13 - FEATURE MAPS E PROCESSO DE DIFUSAO")
print("=" * 66)
print("Dispositivo:", device)


# ============================================================
# 2 - REUTILIZAR A ARQUITETURA DA AULA 12
# ============================================================

def importar_definicoes(arquivo, nomes):

    raiz = ast.parse(
        arquivo.read_text(encoding="utf-8-sig")
    )

    candidatos = {
        n.name: n
        for n in raiz.body
        if isinstance(
            n, (ast.FunctionDef, ast.ClassDef)
        )
    }

    ausentes = [
        nome for nome in nomes
        if nome not in candidatos
    ]

    if ausentes:
        raise RuntimeError(
            f"Definicoes ausentes: {ausentes}"
        )

    modulo = ast.Module(
        body=[candidatos[n] for n in nomes],
        type_ignores=[]
    )

    ast.fix_missing_locations(modulo)

    exec(
        compile(modulo, str(arquivo), "exec"),
        globals()
    )


print("\n1 - RECUPERANDO CLASSES DA AULA 12")

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

dataset = BrainMRIDataset(
    DADOS,
    transform=transform_mri,
    slice_plane="z"
)

assert len(dataset) >= 3

# Usa apenas 3 imagens para evitar consumo
# excessivo de memoria na CPU.
dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0
)

x_sample = next(
    iter(dataloader)
)["image"].to(device)

assert x_sample.shape == (3, 1, 64, 64)

model = UNet().to(device)
model.eval()

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("Imagens do plano Z:", len(dataset))
print("Batch para hooks:", tuple(x_sample.shape))
print("Parametros da U-Net:", parametros)


# ============================================================
# 3 - HOOKS DAS CAMADAS
# ============================================================

print("\n2 - REGISTRANDO HOOKS")

CAMADAS = [
    "down1.block1.conv1",
    "bot1.conv1",
    "out"
]

activations = defaultdict(list)
shapes = []
handles = []

for nome, modulo in model.named_modules():

    if isinstance(
        modulo,
        (
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.GroupNorm
        )
    ):

        def hook(m, entradas, saida, nome=nome):

            shapes.append((
                nome,
                type(m).__name__,
                tuple(saida.shape)
            ))

            if (
                nome in CAMADAS
                and not activations[nome]
            ):
                activations[nome].append(
                    saida.detach().float().cpu()
                )

        handles.append(
            modulo.register_forward_hook(hook)
        )


# ============================================================
# 4 - FORWARD E CAPTURA DAS ATIVACOES
# ============================================================

print("\n3 - EXECUTANDO FORWARD")

try:

    with torch.no_grad():

        tempos = torch.randint(
            0,
            1000,
            (3,),
            device=device
        )

        pred = model(
            x_sample,
            tempos
        )

finally:

    # Evita acumular hooks em execucoes futuras.
    for handle in handles:
        handle.remove()


assert pred.shape == (3, 1, 64, 64)

for camada in CAMADAS:

    assert len(activations[camada]) == 1

    print(
        camada,
        "->",
        tuple(activations[camada][0].shape)
    )

print("Camadas inspecionadas:", len(shapes))

print(
    "Timesteps sorteados:",
    tempos.cpu().tolist()
)


# ============================================================
# 5 - SALVAR TODOS OS SHAPES
# ============================================================

print("\n4 - SALVANDO SHAPES DAS CAMADAS")

arquivo_csv = (
    SAIDA / "aula13_formatos_camadas.csv"
)

with arquivo_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as arq:

    writer = csv.writer(arq)

    writer.writerow([
        "camada",
        "tipo",
        "shape"
    ])

    for nome, tipo, shape in shapes:

        writer.writerow([
            nome,
            tipo,
            str(shape)
        ])

print("CSV de shapes: OK")


# ============================================================
# 6 - VISUALIZAR FEATURE MAPS
# ============================================================

print("\n5 - VISUALIZANDO FEATURE MAPS")

list_img = [0, 1, 2]

for camada in CAMADAS:

    tensor = activations[camada][0]

    num_channels = tensor.shape[1]
    n = min(20, num_channels)

    for indice in list_img:

        fmap = tensor[indice]

        fig, axs = plt.subplots(
            1,
            n,
            figsize=(max(3, 1.65 * n), 2.7),
            squeeze=False
        )

        for canal, ax in enumerate(axs[0]):

            ax.imshow(
                fmap[canal].numpy(),
                cmap="inferno"
            )

            ax.axis("off")

            ax.set_title(
                str(canal),
                fontsize=8
            )

        fig.suptitle(
            f"{camada} | imagem {indice}"
            f" | {n} canais"
        )

        fig.tight_layout()

        nome_seguro = camada.replace(
            ".", "_"
        )

        arquivo_png = (
            SAIDA
            / f"aula13_feature_{nome_seguro}"
              f"_imagem_{indice}.png"
        )

        fig.savefig(
            arquivo_png,
            dpi=120
        )

        plt.close(fig)

    print(
        "Paineis salvos:",
        camada,
        "| canais:",
        n
    )


# ============================================================
# 7 - SCHEDULER LINEAR
# ============================================================

print("\n6 - CONFIGURANDO SCHEDULER LINEAR")

T = 1000

beta_start = 0.001
beta_end = 0.02

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

assert torch.all(
    alpha_bar[1:] < alpha_bar[:-1]
)

print("Timesteps:", T)
print("Beta inicial:", betas[0].item())
print("Beta final:", betas[-1].item())
print("Alpha_bar final:", alpha_bar[-1].item())

print("Scheduler linear: OK")


# ============================================================
# 8 - FUNCAO Q_SAMPLE
# ============================================================

print("\n7 - IMPLEMENTANDO Q_SAMPLE")


def q_sample(x0, t, noise):

    a = alpha_bar[t].view(
        -1, 1, 1, 1
    )

    return (
        torch.sqrt(a) * x0
        + torch.sqrt(1.0 - a) * noise
    )


# ============================================================
# 9 - PROCESSO REVERSO DDPM
# ============================================================

print("\n8 - IMPLEMENTANDO P_SAMPLE_DDPM")


@torch.no_grad()
def p_sample_ddpm(
    modelo,
    x,
    t,
    add_noise=True
):

    tempo = torch.full(
        (x.shape[0],),
        t,
        dtype=torch.long,
        device=x.device
    )

    pred_noise = modelo(
        x,
        tempo
    )

    media = (
        x
        - (
            betas[t]
            / torch.sqrt(1 - alpha_bar[t])
        ) * pred_noise
    ) / torch.sqrt(alphas[t])

    if add_noise and t > 0:

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

    return media


# ============================================================
# 10 - TESTAR A DIFUSAO
# ============================================================

print("\n9 - TESTANDO FORWARD E REVERSE")

with torch.no_grad():

    ruido = torch.randn_like(
        x_sample
    )

    timesteps_teste = torch.tensor(
        [0, 500, 999],
        device=device
    )

    xt = q_sample(
        x_sample,
        timesteps_teste,
        ruido
    )

    # Testa somente um passo reverso em t=500.
    # A rede ainda NAO foi treinada.
    x_prev = p_sample_ddpm(
        model,
        xt[1:2],
        t=500,
        add_noise=False
    )

assert xt.shape == x_sample.shape

assert x_prev.shape == (
    1, 1, 64, 64
)

assert torch.isfinite(x_prev).all()

print("q_sample:", tuple(xt.shape), "OK")

print(
    "p_sample_ddpm, um passo:",
    tuple(x_prev.shape),
    "OK"
)


# ============================================================
# 11 - VISUALIZAR A ADICAO DE RUIDO
# ============================================================

print("\n10 - GERANDO GRAFICOS")

fig, axes = plt.subplots(
    2, 3,
    figsize=(9, 6)
)

for i in range(3):

    original = (
        (
            x_sample[i, 0]
            .detach()
            .cpu()
            .numpy()
            + 1
        ) / 2
    ).clip(0, 1)

    ruidosa = (
        (
            xt[i, 0]
            .detach()
            .cpu()
            .numpy()
            + 1
        ) / 2
    ).clip(0, 1)

    axes[0, i].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, i].imshow(
        ruidosa,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, i].set_title("Original")

    axes[1, i].set_title(
        f"Ruido t={timesteps_teste[i].item()}"
    )

    axes[0, i].axis("off")
    axes[1, i].axis("off")

fig.tight_layout()

fig.savefig(
    SAIDA / "aula13_q_sample_mri.png",
    dpi=150
)

plt.close(fig)


# ============================================================
# 12 - GRAFICO ALPHA BAR
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 4)
)

ax.plot(
    alpha_bar.detach().cpu().numpy()
)

ax.set_xlabel("Timestep")
ax.set_ylabel("Alpha acumulado")
ax.set_title(
    "Scheduler linear - 1000 passos"
)

ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    SAIDA / "aula13_scheduler_alpha_bar.png",
    dpi=150
)

plt.close(fig)


# ============================================================
# 13 - SALVAR METADADOS
# ============================================================

resumo = {
    "aula": 13,
    "dataset": "brain-mri-images",
    "imagens_plano_z": len(dataset),
    "modelo_parametros": parametros,
    "imagens_inspecionadas": 3,
    "total_camadas_registradas": len(shapes),
    "camadas_visualizadas": CAMADAS,
    "timesteps": T,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "treinamento_executado": False,
    "processo_reverso_completo_executado": False
}

arquivo_json = (
    SAIDA / "aula13_resumo.json"
)

arquivo_json.write_text(
    json.dumps(
        resumo,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# 14 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 66)
print("VALIDACAO FINAL")
print("=" * 66)

assert arquivo_csv.exists()
assert arquivo_json.exists()

for camada in CAMADAS:

    nome_seguro = camada.replace(
        ".", "_"
    )

    for indice in list_img:

        caminho = (
            SAIDA
            / f"aula13_feature_{nome_seguro}"
              f"_imagem_{indice}.png"
        )

        assert caminho.exists()

for nome in [
    "aula13_q_sample_mri.png",
    "aula13_scheduler_alpha_bar.png"
]:

    assert (SAIDA / nome).exists()

print("Dataset MRI plano Z: OK")
print("UNet residual Aula 12: OK")
print("Hooks removidos apos forward: OK")
print("Formatos das camadas em CSV: OK")
print("9 paineis de feature maps: OK")
print("Scheduler linear 1000 passos: OK")
print("q_sample: OK")
print("p_sample_ddpm, um passo: OK")
print("Graficos e JSON: OK")

print("\nTREINAMENTO EXECUTADO: NAO")
print("AULA 13 EXECUTADA COM SUCESSO!")
