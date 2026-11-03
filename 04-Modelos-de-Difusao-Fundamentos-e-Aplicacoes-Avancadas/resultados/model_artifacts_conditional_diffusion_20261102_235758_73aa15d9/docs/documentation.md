
# Modelo de Difusao Condicional

Experimento: conditional_diffusion_20261102_235758_73aa15d9

## Visao geral

Modelo educacional ConditionalUNet em PyTorch
para geracao de imagens de ressonancia magnetica
de 64x64 pixels.

Treinamento reduzido:
3 epocas registradas.

Sem validacao clinica.

## Arquitetura

- Modelo: ConditionalUNet
- Base channels: 16
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

- FID64: 19.956939697265625
- Inception Score: 1.0891852378845215
- LPIPS diversidade: 0.03768837860409628
- LPIPS real/gerada: 0.6658141801754633

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
    **{k: params[k] for k in campos}
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
