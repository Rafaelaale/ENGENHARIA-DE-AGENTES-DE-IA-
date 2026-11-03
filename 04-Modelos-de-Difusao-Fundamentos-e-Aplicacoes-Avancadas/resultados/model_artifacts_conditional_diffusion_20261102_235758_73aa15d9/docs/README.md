
# Modelo Condicional de Difusao - Projeto 04

Experimento: conditional_diffusion_20261102_235758_73aa15d9

Arquitetura: ConditionalUNet

Base channels: 16

Pesos originais: aula21_reduzido_melhor_modelo.pth

## Estrutura

- models/: modelo, arquitetura e pesos
- configs/: configuracoes e metadados
- metrics/: metricas finais
- logs/: historico de treinamento
- samples/: imagens reais e sinteticas
- docs/: documentacao e hashes

## Classes

- x = 0 = sagital
- y = 1 = coronal
- z = 2 = axial
- classe nula CFG = 3

## Carregar pacote

import torch

pacote = torch.load(
    "models/conditional_unet_final.pth",
    map_location="cpu",
    weights_only=True
)

## Recriar a rede

Importar ConditionalUNet de:

models/conditional_unet_architecture.py

Usar os parametros em:

pacote["model_config"]["model_params"]

O campo null_label e informativo.
Nao deve ser passado ao construtor.

Depois executar:

model.load_state_dict(
    pacote["model_state_dict"]
)

model.eval()

## Limitacoes

Treinamento reduzido e educativo.

FID64 nao equivale a FID2048.

Amostras de avaliacao podem ter sido
utilizadas em treinamentos anteriores.

Nao utilizar as imagens para diagnostico.

O modelo nao possui validacao clinica.
