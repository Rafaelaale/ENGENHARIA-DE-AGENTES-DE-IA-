# Calibracao de thresholds e flakiness

Trilha: LLMOps com GitHub Actions e DeepEval.

## Experimento

- Metricas: AnswerRelevancy e Faithfulness
- Modelo avaliador: Ollama llama3.2
- Thresholds: 0.25, 0.50 e 0.75
- Repeticoes: 3 por caso e metrica

## Resultados observados

- AnswerRelevancy: 1.0 nos dois casos
- Faithfulness: 0.0 nos dois casos na ultima execucao
- Variacao de Faithfulness entre execucoes
- Possivel falso negativo na resposta sobre o preco da consulta

## Aprendizado

A calibracao deve considerar a confiabilidade do modelo
avaliador. Diminuir o threshold nao corrige falsos negativos.

Os arquivos registram um experimento de avaliacao,
nao uma garantia de qualidade do chatbot.
