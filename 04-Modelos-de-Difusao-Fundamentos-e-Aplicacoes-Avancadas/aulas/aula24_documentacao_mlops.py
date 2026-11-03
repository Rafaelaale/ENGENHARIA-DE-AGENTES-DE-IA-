# ================================================================
# PROJETO 04 - AULA 24
# DOCUMENTACAO E EXPORTACAO PARA MLOPS
# Complementa o experimento validado na Aula 23.
# ================================================================

from pathlib import Path
from datetime import datetime

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image


# ================================================================
# 1 - CONFIGURACOES
# ================================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"

EXPERIMENTO = (
    "model_artifacts_conditional_diffusion_"
    "20261102_235758_73aa15d9"
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["teste", "exportar"],
    default="teste"
)

parser.add_argument(
    "--experimento",
    default=EXPERIMENTO
)

parser.add_argument(
    "--exemplos",
    type=int,
    default=5
)

args = parser.parse_args()

if not 1 <= args.exemplos <= 20:
    parser.error(
        "--exemplos deve estar entre 1 e 20"
    )

PASTA = RESULTADOS / args.experimento

CACHE = (
    RESULTADOS
    / "aula22_avaliacao_final"
    / "avaliar_ddim50_n60_g3"
    / "amostras.pt"
)

MODELO_COMPLETO = (
    PASTA
    / "models"
    / "conditional_unet_final.pth"
)

PESOS_ORIGINAIS = (
    PASTA
    / "models"
    / "conditional_unet_best_weights.pth"
)

PESOS_INFERENCIA = (
    PASTA
    / "models"
    / "model_weights_only.pth"
)

ARQUITETURA = (
    PASTA
    / "models"
    / "conditional_unet_architecture.py"
)

CONFIG = (
    PASTA
    / "configs"
    / "model_config.json"
)

METADADOS = (
    PASTA
    / "configs"
    / "experiment_metadata.json"
)

AVALIACAO = (
    PASTA
    / "metrics"
    / "final_evaluation.json"
)

HISTORICO_CSV = (
    PASTA
    / "logs"
    / "historico_treinamento.csv"
)


# ================================================================
# 2 - FUNCOES AUXILIARES
# ================================================================

def sha256(caminho):

    dig = hashlib.sha256()

    with caminho.open("rb") as arquivo:

        for bloco in iter(
            lambda: arquivo.read(1024 * 1024),
            b""
        ):
            dig.update(bloco)

    return dig.hexdigest()


def ler_json(caminho):

    return json.loads(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )


def gravar_texto(caminho, texto):

    if caminho.exists():

        atual = caminho.read_text(
            encoding="utf-8-sig"
        )

        if atual == texto:
            return

        data = datetime.now().strftime(
            "%Y%m%d-%H%M%S-%f"
        )

        backup = caminho.with_name(
            caminho.name + ".backup-" + data
        )

        shutil.copy2(
            caminho,
            backup
        )

    temporario = caminho.with_name(
        caminho.name + ".tmp"
    )

    temporario.write_text(
        texto,
        encoding="utf-8"
    )

    os.replace(
        temporario,
        caminho
    )


def gravar_json(caminho, valor):

    texto = json.dumps(
        valor,
        ensure_ascii=False,
        indent=2
    ) + "\n"

    gravar_texto(
        caminho,
        texto
    )


def numerico(valor):

    if valor is None or valor == "":
        return None

    try:

        numero = float(valor)

        if numero.is_integer():
            return int(numero)

        return numero

    except (ValueError, TypeError):

        return valor


# ================================================================
# 3 - VALIDAR ARQUIVOS EXISTENTES
# ================================================================

print("=" * 64)
print("AULA 24 - DOCUMENTACAO E EXPORTACAO PARA MLOPS")
print("=" * 64)

print("Modo:", args.modo)
print("Experimento:", PASTA.name)

necessarios = [
    PASTA,
    CACHE,
    MODELO_COMPLETO,
    PESOS_ORIGINAIS,
    ARQUITETURA,
    CONFIG,
    METADADOS,
    AVALIACAO,
    HISTORICO_CSV
]

for caminho in necessarios:

    if not caminho.exists():

        raise FileNotFoundError(
            "Artefato ausente: "
            + str(caminho)
        )

model_config = ler_json(CONFIG)

metadata = ler_json(METADADOS)

final_evaluation = ler_json(AVALIACAO)

model_params = model_config[
    "model_params"
]

assert (
    model_config["architecture"]
    == "ConditionalUNet"
)

assert model_params["num_classes"] == 3

assert model_params["null_label"] == 3

assert (
    model_config["dataset_info"]["class_mapping"]
    == {"x": 0, "y": 1, "z": 2}
)


# ================================================================
# 4 - RECUPERAR CACHE DA AULA 22
# ================================================================

cache = torch.load(
    CACHE,
    map_location="cpu",
    weights_only=True
)

real = cache["real"]

fake = cache["fake"]

labels = cache["labels"].long()

assert (
    real.shape
    == fake.shape
    == (60, 1, 64, 64)
)

assert (
    torch.bincount(
        labels,
        minlength=3
    ).tolist()
    == [20, 20, 20]
)

assert bool(
    torch.isfinite(real).all()
    and torch.isfinite(fake).all()
)

assert (
    cache["config"]["pesos_sha256"]
    == sha256(PESOS_ORIGINAIS)
)


# ================================================================
# 5 - RECUPERAR HISTORICO
# ================================================================

with HISTORICO_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as arquivo:

    history = [
        {
            k: numerico(v)
            for k, v in linha.items()
        }
        for linha in csv.DictReader(arquivo)
    ]

if not history:
    raise RuntimeError(
        "Historico de treinamento vazio."
    )


# ================================================================
# 6 - VALIDAR O MODELO EXPORTADO
# ================================================================

pacote = torch.load(
    MODELO_COMPLETO,
    map_location="cpu",
    weights_only=True
)

assert (
    pacote["diffusion_params"]["T"]
    == 1000
)

assert (
    pacote["model_config"]
    ["model_params"]["num_classes"]
    == 3
)

# Carrega a arquitetura exportada na Aula 23.

spec = importlib.util.spec_from_file_location(
    "conditional_unet_aula24",
    ARQUITETURA
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(mod)

campos = (
    "in_channels",
    "base_channels",
    "num_classes",
    "time_dim",
    "label_dim",
    "dropout_prob"
)

kwargs = {
    k: model_params[k]
    for k in campos
}

model = mod.ConditionalUNet(
    **kwargs
).cpu()

model.load_state_dict(
    torch.load(
        PESOS_ORIGINAIS,
        map_location="cpu",
        weights_only=True
    ),
    strict=True
)

model.eval()

with torch.inference_mode():

    saida = model(
        torch.zeros(1, 1, 64, 64),
        torch.tensor([0]),
        torch.tensor([0])
    )

assert saida.shape == (
    1, 1, 64, 64
)


# ================================================================
# 7 - RESUMO DAS VALIDACOES
# ================================================================

print("\n1 - VALIDACAO DOS ARTEFATOS")

print("Modelo e pesos: OK")
print("Forward da U-Net: OK")
print("Cache: 60 imagens reais + 60 geradas")
print("Classes: 20 por classe")

print(
    "Historico:",
    len(history),
    "epocas"
)

metricas = final_evaluation[
    "overall_metrics"
]

print(
    "FID64:",
    metricas["fid"]
)

print(
    "IS:",
    metricas["inception_score_media"]
)

print(
    "LPIPS diversidade:",
    metricas["lpips_diversidade_sintetica"]
)


# ================================================================
# 8 - MODO TESTE
# ================================================================

if args.modo == "teste":

    print(
        "\nAULA 24 - TESTE CONCLUIDO!"
    )

    sys.exit(0)


# ================================================================
# 9 - SALVAR PESOS PARA INFERENCIA
# ================================================================

inicio_exportacao = (
    datetime.now().astimezone()
)

print(
    "\n2 - SALVANDO PESOS PARA INFERENCIA"
)

if PESOS_INFERENCIA.exists():

    if (
        sha256(PESOS_INFERENCIA)
        != sha256(PESOS_ORIGINAIS)
    ):

        raise RuntimeError(
            "Ja existem pesos de inferencia "
            "diferentes. Arquivo preservado."
        )

else:

    shutil.copy2(
        PESOS_ORIGINAIS,
        PESOS_INFERENCIA
    )

assert (
    sha256(PESOS_INFERENCIA)
    == sha256(PESOS_ORIGINAIS)
)

print(
    "models/model_weights_only.pth: OK"
)


# ================================================================
# 10 - CONFIGURACAO DO DATASET
# ================================================================

print(
    "\n3 - SALVANDO CONFIGURACAO DO DATASET"
)

mapeamento = (
    model_config["dataset_info"]["class_mapping"]
)

dataset_config = {

    "class_mapping": mapeamento,

    "idx_to_label": {
        str(v): k
        for k, v in mapeamento.items()
    },

    "class_descriptions": {
        "x": "sagital",
        "y": "coronal",
        "z": "axial"
    },

    "num_classes": 3,

    "dataset_name": (
        model_config["dataset_info"]
        .get("name", "Brain MRI")
    ),

    "total_samples": (
        model_config["dataset_info"]
        .get("total_samples")
    ),

    "preprocessing": {
        "resize": [64, 64],
        "channels": 1,
        "normalize_mean": 0.5,
        "normalize_std": 0.5,
        "range": [-1, 1]
    }
}

gravar_json(
    PASTA
    / "configs"
    / "dataset_config.json",
    dataset_config
)

print(
    "configs/dataset_config.json: OK"
)


# ================================================================
# 11 - HISTORICO DE TREINAMENTO EM JSON
# ================================================================

print(
    "\n4 - SALVANDO HISTORICO E LOG"
)

training_history = {

    "source": (
        "logs/historico_treinamento.csv"
    ),

    "epochs_recorded": len(history),

    "records": history,

    "note": (
        "FID, IS e LPIPS foram calculados "
        "separadamente na Aula 22."
    )
}

gravar_json(
    PASTA
    / "metrics"
    / "training_history.json",
    training_history
)


# ================================================================
# 12 - LOG ESTRUTURADO
# ================================================================

training_log = {

    "experiment_id": (
        metadata["experiment_id"]
    ),

    "model_version": (
        metadata["model_version"]
    ),

    "operation": (
        "documentation_and_inference_export"
    ),

    "status": "completed",

    "training_scope": (
        "reduced_experiment"
    ),

    "training_start_time": None,

    "export_start_time": (
        inicio_exportacao.isoformat()
    ),

    "export_end_time": (
        datetime.now()
        .astimezone()
        .isoformat()
    ),

    "epochs_recorded": len(history),

    "final_metrics": (
        final_evaluation["overall_metrics"]
    ),

    "model_path": (
        "models/conditional_unet_final.pth"
    ),

    "inference_weights": (
        "models/model_weights_only.pth"
    ),

    "artifacts_directory": ".",

    "note": (
        "O inicio do treinamento original "
        "nao foi registrado."
    )
}

gravar_json(
    PASTA
    / "logs"
    / "training_log.json",
    training_log
)

print(
    "metrics/training_history.json: OK"
)

print(
    "logs/training_log.json: OK"
)


# ================================================================
# 13 - SALVAR EXEMPLOS DAS TRES CLASSES
# ================================================================

print(
    "\n5 - SALVANDO CINCO EXEMPLOS POR CLASSE"
)

demo_samples = {}

for classe, label in enumerate(
    ["x", "y", "z"]
):

    posicoes = torch.where(
        labels == classe
    )[0][:args.exemplos]

    demo_samples[label] = []

    for numero, i in enumerate(
        posicoes,
        1
    ):

        destino = (
            PASTA
            / "samples"
            / f"{label}_sample_{numero}.png"
        )

        matriz = (
            (
                fake[int(i), 0]
                .float()
                .clamp(-1, 1)
                + 1
            ) / 2 * 255
        ).to(
            torch.uint8
        ).numpy()

        if destino.exists():

            with Image.open(destino) as img:
                existente = np.asarray(
                    img.convert("L")
                )

            if not np.array_equal(
                existente,
                matriz
            ):

                raise RuntimeError(
                    "Imagem existente diferente: "
                    + str(destino)
                )

        else:

            Image.fromarray(
                matriz,
                mode="L"
            ).save(destino)

        demo_samples[label].append(
            "samples/" + destino.name
        )

    print(
        f"Classe {label}: "
        f"{len(demo_samples[label])} "
        "exemplos reutilizados do cache"
    )

gravar_json(
    PASTA
    / "samples"
    / "demo_samples.json",
    demo_samples
)


# ================================================================
# 14 - DOCUMENTACAO MARKDOWN
# ================================================================

print(
    "\n6 - GERANDO DOCUMENTACAO MARKDOWN"
)

readme = f"""
# Modelo de Difusao Condicional

Experimento: {metadata["experiment_id"]}

## Visao geral

Modelo educacional ConditionalUNet em PyTorch
para geracao de imagens de ressonancia magnetica
de 64x64 pixels.

Treinamento reduzido:
{len(history)} epocas registradas.

Sem validacao clinica.

## Arquitetura

- Modelo: ConditionalUNet
- Base channels: {model_params["base_channels"]}
- Classes reais: 3
- X = sagital = 0
- Y = coronal = 1
- Z = axial = 2
- Classe incondicional CFG = 3
- Resolucao: 64x64
- Canais: 1
- Normalizacao: [-1, 1]
- Timesteps: 1000
- Beta inicial: 0.0001
- Beta final: 0.02

## Metricas da Aula 22

- FID64: {metricas["fid"]}
- Inception Score: {metricas["inception_score_media"]}
- LPIPS diversidade: {metricas["lpips_diversidade_sintetica"]}
- LPIPS real/gerada: {metricas.get("lpips_real_gerada_nao_pareadas")}

LPIPS calculado com SqueezeNet.

FID64 nao equivale ao FID2048.

## Arquivos

### models/

- conditional_unet_final.pth
- model_weights_only.pth
- conditional_unet_architecture.py

### configs/

- model_config.json
- dataset_config.json
- experiment_metadata.json

### metrics/

- final_evaluation.json
- training_history.json

### logs/

- training_log.json
- historico_treinamento.csv

### samples/

- demo_samples.json
- x_sample_1.png ate x_sample_5.png
- y_sample_1.png ate y_sample_5.png
- z_sample_1.png ate z_sample_5.png

### docs/

- documentation.md
- checksums_sha256.json

## Recarregar o modelo

Execute no diretorio do experimento.

```python
import json
import sys
from pathlib import Path
import torch

base = Path.cwd()

sys.path.insert(
    0,
    str(base / "models")
)

from conditional_unet_architecture import ConditionalUNet

cfg = json.loads(
    (
        base / "configs" / "model_config.json"
    ).read_text(encoding="utf-8")
)

params = cfg["model_params"]

campos = (
    "in_channels",
    "base_channels",
    "num_classes",
    "time_dim",
    "label_dim",
    "dropout_prob"
)

model = ConditionalUNet(
    **{{k: params[k] for k in campos}}
)

pesos = torch.load(
    base / "models" / "model_weights_only.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    pesos,
    strict=True
)

model.eval()
```

## Inferencia

O carregamento do modelo nao gera imagens
automaticamente.

Para gerar imagens, utilizar DDIM ou DDPM
com os mesmos parametros de difusao,
mapeamento de classes e guidance do treinamento.

Nao existe modelo ONNX nesta exportacao.

## Limitacoes

- Treinamento reduzido e educacional.
- Amostras de demonstracao reutilizadas do cache.
- Avaliacao feita com 60 imagens sinteticas.
- Possivel sobreposicao com dados de treinos anteriores.
- Metricas perceptuais fora do dominio MRI.
- Nenhuma validacao anatomica ou clinica.
- Nao utilizar para diagnostico.
"""

gravar_texto(
    PASTA
    / "docs"
    / "documentation.md",
    readme
)


# ================================================================
# 15 - HASHES DOS ARTEFATOS
# ================================================================

print(
    "\n7 - ATUALIZANDO INTEGRIDADE SHA-256"
)

arquivo_hashes = (
    PASTA
    / "docs"
    / "checksums_sha256.json"
)

checksums = {

    str(
        caminho.relative_to(PASTA)
    ).replace("\\", "/"): sha256(caminho)

    for caminho in sorted(
        PASTA.rglob("*")
    )

    if (
        caminho.is_file()
        and caminho != arquivo_hashes
        and ".backup-" not in caminho.name
        and not caminho.name.endswith(".tmp")
    )
}

gravar_json(
    arquivo_hashes,
    checksums
)


# ================================================================
# 16 - VALIDACAO DO MODELO DE INFERENCIA
# ================================================================

pesos_recarregados = torch.load(
    PESOS_INFERENCIA,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    pesos_recarregados,
    strict=True
)

assert len(demo_samples) == 3

assert all(
    len(v) == args.exemplos
    for v in demo_samples.values()
)

assert (
    PASTA
    / "docs"
    / "documentation.md"
).exists()


# ================================================================
# 17 - VALIDACAO FINAL
# ================================================================

print("\n" + "=" * 64)
print("VALIDACAO FINAL - AULA 24")
print("=" * 64)

for nome in (
    "models",
    "configs",
    "metrics",
    "logs",
    "samples",
    "docs"
):

    assert (PASTA / nome).is_dir()

    print(nome + "/: OK")

print("Pesos somente para inferencia: OK")
print("Arquivos de configuracao e metadados: OK")
print("Historico JSON e log estruturado: OK")

print(
    f"{args.exemplos} demonstracoes "
    "por classe: OK"
)

print("Documentacao Markdown: OK")
print("Hashes SHA-256: OK")
print("Modelo de inferencia recarregado: OK")

print("\nEXPERIMENTO:")
print(PASTA)

print(
    "\nAULA 24 - EXPORTACAO CONCLUIDA!"
)
