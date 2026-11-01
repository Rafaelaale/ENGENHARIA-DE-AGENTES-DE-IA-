# Redes Neurais Recorrentes com PyTorch

### Deep Learning | NLP | GRU | LSTM | Classificação de Sequências

Projeto de portfólio técnico com implementação e validação de arquiteturas recorrentes utilizando Python e PyTorch.

**Área:** Machine Learning, Ciência de Dados e Engenharia de IA.

**Autor:** Rafael Alexandre Oliveira Araujo

[Visualizar dashboard](dashboard.html)

---

## Visão geral

Este projeto explora redes neurais recorrentes para processamento de dados sequenciais, com ênfase em classificação de textos e análise de sentimentos.

A implementação parte de conceitos fundamentais e evolui até um classificador binário com:

- Embeddings de palavras
- GRU e LSTM
- Sequências de comprimento variável
- Padding e BucketIterator
- `pack_padded_sequence`
- Camada linear de classificação
- CrossEntropyLoss e Adam
- Predições probabilísticas com Softmax

## Arquitetura

```text
Texto
  |
Tokenização
  |
Índices do vocabulário
  |
Embedding (100 dimensões)
  |
Empacotamento das sequências
  |
GRU (256 unidades ocultas)
  |
Hidden final válido
  |
Linear (256 -> 2)
  |
Logits das classes
  |
Softmax para inferência
  |
Negativo / Positivo
```

A classificação utiliza o último estado oculto válido da GRU, evitando que tokens de padding interfiram na representação final.

## Tecnologias

| Tecnologia | Aplicação |
|---|---|
| Python | Implementação dos laboratórios |
| PyTorch | Construção e treinamento das redes |
| RNN | Modelagem de sequências |
| GRU | Processamento recorrente |
| LSTM | Estudo de memória de longo prazo |
| Embedding | Representação vetorial de tokens |
| Adam | Otimização |
| CrossEntropyLoss | Classificação binária com dois logits |

## Laboratórios

| Tema | Competência demonstrada |
|---|---|
| Arquiteturas recorrentes | Compreensão de diferentes modelos |
| Vanishing Gradient | Análise do fluxo de gradientes |
| Gates | Mecanismos de controle de memória |
| LSTM e GRU | Comparação de arquiteturas |
| Camada RNN | Manipulação de tensores e estados ocultos |
| RNN no PyTorch | Classificação de sequências |
| Redes profundas e bidirecionais | Arquiteturas multicamadas |
| Representação do dado | Vocabulário e embeddings |
| BucketIterator | Organização de batches |
| Sequências empacotadas | Processamento sem padding inválido |
| Classificando sequências | Treinamento e inferência |

Os arquivos Python e relatórios estão organizados na pasta `RNN_Alura_Automacao`.

## Resultados observados

### Classificação de sentimentos — laboratório didático

| Métrica | Resultado |
|---|---|
| Amostras de treinamento | 6 |
| Épocas | 30 |
| Acurácia de treinamento | 100% |
| Testes automatizados | 11/11 |
| Dispositivo | CPU |
| Loss inicial registrada | 0.7790 |
| Loss final exibida | 0.0000 (arredondada) |

**Limitação importante:** os resultados acima são obtidos sobre seis avaliações didáticas. Não representam acurácia de generalização nem desempenho no IMDb oficial.

O classificador ainda pode cometer erros em frases novas. Por exemplo, a avaliação "Terrible movie." recebeu 98,19% de probabilidade positiva no experimento registrado.

### Sequências empacotadas

- 21/21 verificações aprovadas.
- Embedding com 100 dimensões.
- GRU com 256 unidades ocultas.
- Preservação dos comprimentos reais das sequências.
- Verificação do último estado oculto válido.

### BucketIterator

- 10/10 testes aprovados.
- Padding global: 13 posições.
- Padding com agrupamento: 4 posições.
- Redução observada de aproximadamente 69,2% nesse conjunto.

## Execução

Requisitos:

- Python
- PyTorch
- Ambiente virtual recomendado

Exemplo no PowerShell:

```powershell
.\.venv\Scripts\python.exe ".\RNN_Alura_Automacao\IMDb_Classificando_Sequencias\laboratorio_classificando_sequencias.py"
```

Alguns laboratórios recebem caminhos de arquivos anteriores como argumentos. Consulte o código correspondente antes de executar.

## Próximas melhorias

- Carregar o IMDb oficial.
- Separar treino, validação e teste.
- Avaliar precisão, recall e F1-score.
- Comparar GRU, LSTM e arquiteturas bidirecionais.
- Avaliar embeddings pré-treinados.
- Publicar métricas de generalização reproduzíveis.

## Objetivo profissional

Demonstrar competências práticas em:

- Desenvolvimento de modelos de Deep Learning
- Processamento de Linguagem Natural
- Manipulação de tensores
- Treinamento e avaliação de redes neurais
- Organização de experimentos
- Documentação e validação técnica

**Projeto de estudo e portfólio.** Resultados experimentais são apresentados com suas limitações.