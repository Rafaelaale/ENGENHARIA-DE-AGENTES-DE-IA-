# Aula 01 - Historico das Redes Difusoras

Curso: Modelos de Difusao - Fundamentos e Aplicacoes Avancadas

## 1. Objetivo

Compreender os principais modelos generativos
e a evolucao historica dos modelos de difusao.

## 2. Modelos Generativos

### VAE - Variational Autoencoder

- Trabalha com representacoes latentes.
- Aprende a codificar e reconstruir dados.
- Permite amostragem de novos exemplos.
- Algumas implementacoes produzem imagens suavizadas.

### GAN - Generative Adversarial Network

- Possui gerador e discriminador.
- As duas redes sao treinadas competitivamente.
- Pode gerar imagens nitidas.
- Pode apresentar instabilidade e mode collapse.

### Modelos de Difusao

- Processo direto: adiciona ruido aos dados.
- Processo reverso: aprende a remover ruido.
- Gera novas amostras a partir de ruido.
- Pode apresentar alta qualidade de imagem.
- A geracao pode exigir muitas etapas.

## 3. Evolucao Historica

2015 - Introducao dos modelos de difusao.
2019 - Avancos nos modelos baseados em score.
2020 - DDPM.
2021 - Melhorias na amostragem e variancia.
2022 - Modelos de difusao latente e Stable Diffusion.
2023 - ControlNet e avancos em video.
2024 - Maior consistencia temporal e geracao de video.

## 4. Conceitos Fundamentais

- Processo direto (forward diffusion)
- Processo reverso (reverse diffusion)
- Ruido
- Amostragem (sampling)
- DDPM
- DDIM
- Espaco latente
- Stable Diffusion
- ControlNet

## 5. Ferramenta da Trilha

Biblioteca prevista: Hugging Face Diffusers.

A instalacao sera realizada quando chegar
a primeira aula pratica correspondente.

## 6. Conclusao

Os modelos de difusao aprendem a reverter
um processo de adicao gradual de ruido.

Diferentemente das GANs, seu treinamento
nao depende da competicao entre duas redes.

O conhecimento desta aula prepara o estudo
dos algoritmos de difusao e de suas aplicacoes.

Status: aula teorica registrada.
