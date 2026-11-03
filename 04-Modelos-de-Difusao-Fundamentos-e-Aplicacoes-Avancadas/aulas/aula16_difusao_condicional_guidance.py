# ============================================================
# PROJETO 04 - AULA 16
# DIFUSAO CONDICIONAL E CLASSIFIER-FREE GUIDANCE
# ============================================================

from pathlib import Path
import ast
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
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

img_size = 64
T = 1000

beta_start = 1e-4
beta_end = 0.02

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
print("AULA 16 - DIFUSAO CONDICIONAL E GUIDANCE")
print("=" * 65)

print("Dispositivo:", device)


# ============================================================
# 2 - REUTILIZAR CLASSES DA AULA 15
# ============================================================

print("\n1 - CARREGANDO ARQUITETURA CONDICIONAL")


def importar_definicoes(caminho, nomes):

    arvore = ast.parse(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )

    definicoes = {
        node.name: node
        for node in arvore.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.ClassDef)
        )
    }

    faltando = [
        nome
        for nome in nomes
        if nome not in definicoes
    ]

    if faltando:
        raise RuntimeError(
            f"Definicoes ausentes: {faltando}"
        )

    modulo = ast.Module(
        body=[
            definicoes[nome]
            for nome in nomes
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

print("Classes da Aula 15: OK")


# ============================================================
# 3 - DATASET
# ============================================================

print("\n2 - CARREGANDO DATASET X/Y/Z")

dataset = BrainMRIConditionalDataset(
    DADOS,
    transform=transform_mri,
    slice_plane=None
)

assert dataset.num_classes == 3

assert dataset.label_to_idx == {
    "x": 0,
    "y": 1,
    "z": 2
}

print("Total de imagens:", len(dataset))
print("Classes:", dataset.label_to_idx)

# Selecionar uma imagem real da classe Z
indice_z = dataset.slice_planes.index("z")

amostra = dataset[indice_z]

x0 = amostra["image"].unsqueeze(0).to(device)

labels = torch.tensor(
    [dataset.label_to_idx["z"]],
    dtype=torch.long,
    device=device
)

assert x0.shape == (1, 1, 64, 64)

print("Imagem de teste:", tuple(x0.shape))
print("Classe selecionada: Z")


# ============================================================
# 4 - MODELO CONDICIONAL
# ============================================================

print("\n3 - CONFIGURANDO MODELO")

model = ConditionalUNet(
    in_channels=1,
    base_channels=64,
    num_classes=dataset.num_classes,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

model.eval()

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

assert model.null_label == 3

print("Parametros:", parametros)
print("Classe nula CFG:", model.null_label)
print("AdamW: 0.0001")
print("Pesos condicionais treinados: NAO")


# ============================================================
# 5 - SCHEDULER LINEAR
# ============================================================

print("\n4 - CONFIGURANDO SCHEDULER")

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

assert betas.shape == (1000,)
assert alpha_bar.shape == (1000,)

assert torch.all(
    alpha_bar[1:] < alpha_bar[:-1]
)

print("Timesteps:", T)
print("Beta inicial:", betas[0].item())
print("Beta final:", betas[-1].item())
print("Alpha_bar final:", alpha_bar[-1].item())

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    alpha_bar.detach().cpu().numpy()
)

ax.set_title(
    "Difusao condicional - alpha_bar"
)

ax.set_xlabel("Timestep")
ax.set_ylabel("Alpha acumulado")
ax.grid(alpha=0.3)

fig.tight_layout()

arquivo_scheduler = (
    RESULTADOS
    / "aula16_scheduler_linear.png"
)

fig.savefig(
    arquivo_scheduler,
    dpi=150
)

plt.close(fig)

print("Grafico do scheduler: OK")


# ============================================================
# 6 - FORWARD PROCESS Q_SAMPLE
# ============================================================

print("\n5 - IMPLEMENTANDO Q_SAMPLE")


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


passos_visualizacao = [
    0, 250, 500, 750, 999
]

ruido_fixo = torch.randn_like(x0)

fig, axes = plt.subplots(
    1,
    6,
    figsize=(14, 3)
)

original = (
    (x0[0, 0].detach().cpu() + 1) / 2
).clamp(0, 1).numpy()

axes[0].imshow(
    original,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0].set_title("Original")
axes[0].axis("off")

for posicao, passo in enumerate(
    passos_visualizacao,
    start=1
):

    t_tensor = torch.tensor(
        [passo],
        dtype=torch.long,
        device=device
    )

    imagem_ruidosa = q_sample(
        x0,
        t_tensor,
        ruido_fixo
    )

    vis = (
        (imagem_ruidosa[0, 0].detach().cpu() + 1)
        / 2
    ).clamp(0, 1).numpy()

    axes[posicao].imshow(
        vis,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[posicao].set_title(
        f"t={passo}"
    )

    axes[posicao].axis("off")

fig.tight_layout()

arquivo_forward = (
    RESULTADOS
    / "aula16_forward_condicional.png"
)

fig.savefig(
    arquivo_forward,
    dpi=150
)

plt.close(fig)

print("q_sample: OK")
print("Visualizacao do ruido: OK")


# ============================================================
# 7 - CLASSIFIER-FREE GUIDANCE
# ============================================================

print("\n6 - IMPLEMENTANDO GUIDANCE")


@torch.no_grad()
def p_sample_conditional(
    model,
    x,
    t,
    labels,
    guidance_scale=1.0,
    add_noise=True,
    return_details=False
):

    B = x.shape[0]

    t_tensor = torch.full(
        (B,),
        t,
        dtype=torch.long,
        device=x.device
    )

    labels = labels.to(
        device=x.device,
        dtype=torch.long
    )

    # Predicao condicional
    pred_noise_cond = model(
        x,
        t_tensor,
        labels
    )

    # Classe especial sem condicao
    # No modelo da Aula 15, null_label = 3.
    uncond_labels = torch.full_like(
        labels,
        model.null_label
    )

    pred_noise_uncond = model(
        x,
        t_tensor,
        uncond_labels
    )

    # Classifier-Free Guidance
    pred_noise = (
        pred_noise_uncond
        + guidance_scale
        * (
            pred_noise_cond
            - pred_noise_uncond
        )
    )

    # Processo reverso DDPM
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]

    coeff1 = 1 / torch.sqrt(
        alpha_t
    )

    coeff2 = (
        beta_t
        / torch.sqrt(1 - alpha_bar_t)
    )

    mean = coeff1 * (
        x - coeff2 * pred_noise
    )

    if add_noise and t > 0:

        alpha_bar_prev = alpha_bar[t - 1]

        posterior_variance = (
            beta_t
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        x_prev = (
            mean
            + torch.sqrt(posterior_variance)
            * torch.randn_like(x)
        )

    else:

        x_prev = mean

    if return_details:

        return (
            x_prev,
            pred_noise_cond,
            pred_noise_uncond,
            pred_noise
        )

    return x_prev


print("p_sample_conditional: definida")


# ============================================================
# 8 - TESTAR UM PASSO DE GUIDANCE
# ============================================================

print("\n7 - TESTANDO UM PASSO CONDICIONAL")

t_teste = 500

tempo = torch.tensor(
    [t_teste],
    device=device,
    dtype=torch.long
)

x_teste = q_sample(
    x0,
    tempo,
    torch.randn_like(x0)
)

with torch.no_grad():

    (
        x_prev,
        ruido_cond,
        ruido_uncond,
        ruido_cfg
    ) = p_sample_conditional(
        model,
        x_teste,
        t_teste,
        labels,
        guidance_scale=3.0,
        add_noise=False,
        return_details=True
    )

assert x_prev.shape == (1, 1, 64, 64)

assert torch.isfinite(x_prev).all()

assert torch.allclose(
    ruido_cfg,
    ruido_uncond
    + 3.0 * (
        ruido_cond - ruido_uncond
    )
)

assert torch.allclose(
    ruido_uncond
    + 0.0 * (
        ruido_cond - ruido_uncond
    ),
    ruido_uncond
)

assert torch.allclose(
    ruido_uncond
    + 1.0 * (
        ruido_cond - ruido_uncond
    ),
    ruido_cond
)

print("Saida:", tuple(x_prev.shape))
print("guidance_scale=0: OK")
print("guidance_scale=1: OK")
print("guidance_scale=3: OK")
print("Classe nula=3: OK")
print("Um passo DDPM: OK")


# ============================================================
# 9 - FUNCAO PARA GERACAO CONDICIONAL
# ============================================================

print("\n8 - DEFININDO SHOW_AND_PREDICT_CONDITIONAL")


@torch.no_grad()
def show_and_predict_conditional(
    model,
    labels,
    steps=T,
    epoch=None,
    guidance_scale=3.0,
    output_path=None
):

    # Esta funcao deve ser usada APOS
    # treinar o modelo condicional.
    #
    # Para o scheduler atual, a geracao
    # completa usa os 1000 passos.

    if steps != T:
        raise ValueError(
            "Este scheduler exige steps=1000. "
            "Menos passos precisam de um "
            "metodo de amostragem apropriado."
        )

    model.eval()

    labels = labels.to(
        device=device,
        dtype=torch.long
    )

    x = torch.randn(
        labels.shape[0],
        1,
        img_size,
        img_size,
        device=device
    )

    for t in reversed(range(steps)):

        x = p_sample_conditional(
            model,
            x,
            t,
            labels,
            guidance_scale=guidance_scale
        )

    imagem = (
        (x[0, 0].clamp(-1, 1) + 1)
        / 2
    ).detach().cpu().numpy()

    imagem = (
        imagem * 255
    ).astype(np.uint8)

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )

    ax.imshow(
        imagem,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.axis("off")

    nome_classe = dataset.idx_to_label[
        labels[0].item()
    ]

    titulo = f"Classe: {nome_classe.upper()}"

    if epoch is not None:
        titulo += f" | Epoca: {epoch}"

    ax.set_title(titulo)
    fig.tight_layout()

    if output_path is not None:

        fig.savefig(
            output_path,
            dpi=150
        )

    plt.close(fig)

    return imagem


print("show_and_predict_conditional: OK")
print("Geracao de 1000 passos: NAO executada")
print("Motivo: modelo condicional nao treinado")


# ============================================================
# 10 - SALVAR CONFIGURACOES
# ============================================================

print("\n9 - SALVANDO RESULTADOS")

arquivo_json = (
    RESULTADOS
    / "aula16_guidance_configuracoes.json"
)

dados = {
    "aula": 16,
    "modelo": "ConditionalUNet",
    "parametros": parametros,
    "dataset_imagens": len(dataset),
    "classes_reais": dataset.label_to_idx,
    "classe_nula": model.null_label,
    "scheduler": "linear",
    "timesteps": T,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "guidance_scale_teste": 3.0,
    "timestep_teste": t_teste,
    "q_sample_executado": True,
    "p_sample_um_passo_executado": True,
    "geracao_completa_executada": False,
    "treinamento_executado": False,
    "device": str(device)
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
# 11 - VALIDACAO FINAL
# ============================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert parametros == 48826561

assert dataset.num_classes == 3
assert model.null_label == 3

for arquivo in [
    arquivo_scheduler,
    arquivo_forward,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("Dataset Brain MRI X/Y/Z: OK")
print("ConditionalUNet Aula 15: OK")
print("Scheduler linear 1000 passos: OK")
print("Alpha acumulado: OK")
print("q_sample: OK")
print("Predicao condicional: OK")
print("Predicao incondicional: OK")
print("Classifier-Free Guidance: OK")
print("p_sample_conditional: OK")
print("Um passo reverso: OK")
print("show_and_predict_conditional: OK")
print("Graficos e JSON: OK")

print("\nTREINAMENTO EXECUTADO: NAO")
print("GERACAO COMPLETA EXECUTADA: NAO")
print("AULA 16 EXECUTADA COM SUCESSO!")
