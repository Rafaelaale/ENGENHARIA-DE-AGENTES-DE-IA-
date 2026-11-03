# 04 - Modelos de Difusao: Fundamentos e Aplicacoes Avancadas

Trilha educacional completa em Python e PyTorch.

## Conteudos

- Fundamentos de DDPM e processos de difusao
- Arquitetura U-Net e embeddings temporais
- ConditionalUNet e Classifier-Free Guidance
- Geracao de imagens com DDPM e DDIM
- Condicionamento espacial inspirado em ControlNet
- Treinamento e validacao reduzidos em CPU
- Metricas FID64, Inception Score e LPIPS
- Exportacao e documentacao MLOps
- Interface de inferencia interativa

## Arquivos

- `aulas/`: todos os codigos completos das aulas
- `resultados/`: metricas, configuracoes e exemplos de imagens
- `resultados/model_artifacts_conditional_diffusion_20261102_235758_73aa15d9/`: experimento exportado

Os checkpoints intermediarios e o dataset bruto
permanecem no ambiente local de desenvolvimento.

## Resultados experimentais

- Modelo reduzido: 3.982.513 parametros
- FID64: 19.9569
- Inception Score: 1.0892
- LPIPS diversidade: 0.0377

Os resultados sao exploratorios, com treinamento reduzido.
As imagens sinteticas nao possuem validacao clinica.
