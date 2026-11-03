# ==========================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 04 - ENTROPIA E RUIDO
# ==========================================================

from pathlib import Path
import json

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

print("=" * 65)
print("AULA 04 - ENTROPIA E RUIDO")
print("=" * 65)

print("\nPyTorch:", torch.__version__)
print("Dispositivo: CPU")
print("Seeds fixadas: 0")


# ==========================================================
# 2 - CRIAR ONDA SENOIDAL
# ==========================================================

print("\n1 - CRIANDO ONDA SENOIDAL")

x0 = torch.sin(
    torch.linspace(
        0,
        8 * np.pi,
        1000
    )
).unsqueeze(1)

print("Formato do tensor:", tuple(x0.shape))
print("Total de pontos:", x0.shape[0])

assert x0.shape == (1000, 1)


# ==========================================================
# 3 - CONFIGURAR SCHEDULER LINEAR
# ==========================================================

print("\n2 - CONFIGURANDO SCHEDULER")

T = 5

betas = torch.linspace(
    0.01,
    0.2,
    T
)

alphas = 1.0 - betas

alphas_cumprod = torch.cumprod(
    alphas,
    dim=0
)

print("\nBETAS:")
print(betas)

print("\nALPHAS:")
print(alphas)

print("\nALPHAS ACUMULADOS:")
print(alphas_cumprod)


# ==========================================================
# 4 - APLICAR RUIDO GAUSSIANO
# ==========================================================

print("\n3 - ADICIONANDO RUIDO")

xt_list = [x0]

for t in range(T):

    noise = torch.randn_like(x0)

    xt = (
        torch.sqrt(alphas_cumprod[t]) * x0
        + torch.sqrt(1 - alphas_cumprod[t]) * noise
    )

    xt_list.append(xt)

    print(
        f"Estagio {t + 1}/{T}: "
        f"alpha acumulado = "
        f"{alphas_cumprod[t].item():.6f}"
    )

assert len(xt_list) == 6

print("\nCinco estagios de ruido: OK")


# ==========================================================
# 5 - ESTIMAR ENTROPIA DIFERENCIAL
# ==========================================================

print("\n4 - CALCULANDO ENTROPIA")


def estimate_entropy(data, bins=50):

    valores = (
        data.detach()
        .cpu()
        .numpy()
        .flatten()
    )

    hist, bin_edges = np.histogram(
        valores,
        bins=bins,
        density=True
    )

    hist += 1e-8

    largura = (
        bin_edges[1]
        - bin_edges[0]
    )

    entropia = -np.sum(
        hist * np.log(hist) * largura
    )

    return float(entropia)


entropias = []

for i, xt in enumerate(xt_list):

    H = estimate_entropy(xt)

    entropias.append(H)

    print(
        f"t={i} | "
        f"Entropia estimada = {H:.6f}"
    )

assert len(entropias) == 6
assert np.isfinite(entropias).all()


# ==========================================================
# 6 - VISUALIZAR OS SEIS ESTAGIOS
# ==========================================================

print("\n5 - CRIANDO GRAFICOS")

fig, axes = plt.subplots(
    1,
    T + 1,
    figsize=(17, 4),
    sharey=True
)

for i, xt in enumerate(xt_list):

    valores = xt.detach().cpu().numpy()

    axes[i].plot(
        valores,
        color="steelblue",
        linewidth=0.8
    )

    axes[i].set_title(
        f"t={i}\nH={entropias[i]:.2f}"
    )

    axes[i].set_xticks([])

    axes[i].grid(
        alpha=0.2
    )

axes[0].set_ylabel("Amplitude")

fig.suptitle(
    "Difusao: onda senoidal com ruido gaussiano"
)

fig.tight_layout()

arquivo_png = (
    RESULTADOS
    / "aula04_evolucao_ruido_entropia.png"
)

fig.savefig(
    arquivo_png,
    dpi=160,
    bbox_inches="tight"
)

plt.close(fig)

print("Grafico dos seis estagios: OK")


# ==========================================================
# 7 - GRAFICO DA ENTROPIA
# ==========================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    range(T + 1),
    entropias,
    marker="o",
    linewidth=2
)

ax.set_title(
    "Entropia diferencial estimada por estagio"
)

ax.set_xlabel("Estagio")
ax.set_ylabel("Entropia estimada")

ax.set_xticks(
    range(T + 1)
)

ax.grid(alpha=0.3)

fig.tight_layout()

arquivo_entropia = (
    RESULTADOS
    / "aula04_grafico_entropia.png"
)

fig.savefig(
    arquivo_entropia,
    dpi=160
)

plt.close(fig)

print("Grafico da entropia: OK")


# ==========================================================
# 8 - SALVAR VALORES EM CSV
# ==========================================================

print("\n6 - SALVANDO RESULTADOS")

arquivo_csv = (
    RESULTADOS
    / "aula04_entropias.csv"
)

dados_csv = np.column_stack([
    np.arange(T + 1),
    entropias
])

np.savetxt(
    arquivo_csv,
    dados_csv,
    delimiter=",",
    header="estagio,entropia_estimada",
    comments="",
    fmt=["%d", "%.8f"]
)

arquivo_json = (
    RESULTADOS
    / "aula04_parametros_resultados.json"
)

dados_json = {

    "aula": 4,
    "tema": "Entropia e ruido",
    "pontos_senoide": 1000,
    "estagios_ruido": T,
    "seed": 0,
    "beta_inicial": 0.01,
    "beta_final": 0.2,
    "betas": betas.tolist(),
    "alphas_cumprod": alphas_cumprod.tolist(),
    "entropias": entropias,
    "metodo_entropia": "histograma com 50 bins"
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        dados_json,
        arquivo,
        indent=4,
        ensure_ascii=False
    )

print("CSV salvo: OK")
print("JSON salvo: OK")


# ==========================================================
# 9 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert x0.shape == (1000, 1)
assert len(xt_list) == T + 1

assert all(
    xt.shape == x0.shape
    for xt in xt_list
)

assert torch.isfinite(
    torch.stack(xt_list)
).all()

for arquivo in [
    arquivo_png,
    arquivo_entropia,
    arquivo_csv,
    arquivo_json
]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("PyTorch: OK")
print("NumPy: OK")
print("Onda senoidal: OK")
print("Scheduler linear: OK")
print("Cinco estagios de ruido: OK")
print("Estimativa de entropia: OK")
print("Graficos PNG: OK")
print("Resultados CSV: OK")
print("Parametros JSON: OK")

print("\nAULA 04 EXECUTADA COM SUCESSO!")
