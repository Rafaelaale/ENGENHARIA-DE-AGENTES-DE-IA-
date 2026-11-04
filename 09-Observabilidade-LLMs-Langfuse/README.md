# Observabilidade de LLMs com Langfuse

## Portfolio de Engenharia de IA

**Autor:** Rafael Alexandre Oliveira Araujo

Projeto desenvolvido durante a trilha Alura de observabilidade para modelos de linguagem.

### Tecnologias

- Python
- Langfuse
- Ollama
- Llama 3.2
- LLM-as-a-Judge
- Streamlit

## Funcionalidades implementadas

- Tracing de chamadas a modelos de linguagem.
- Monitoramento com spans e eventos.
- Gerenciamento de prompts.
- Avaliacao automatica com LLM-as-a-Judge.
- Classificacao de severidade das respostas.
- Fallback automatico para casos criticos.
- Reavaliacao das respostas alternativas.
- Identificacao de casos prioritarios.
- Consolidacao de metricas e consumo de tokens.

## Dashboard de resultados

| Indicador | Resultado |
|---|---:|
| Exemplos avaliados | 5 |
| Score medio | 1,8 / 5 |
| Casos criticos | 5 |
| Casos para revisao | 5 |
| Fallbacks aplicados | 5 |
| Tokens de entrada | 5.032 |
| Tokens de saida | 1.779 |

Os resultados correspondem a uma execucao experimental local.

O pipeline foi executado com sucesso, mas a qualidade das respostas permaneceu baixa.
Todos os casos continuaram criticos apos o fallback.
Isso demonstra a importancia da observabilidade para identificar falhas antes de utilizar modelos em producao.

## Dashboard interativo

O arquivo dashboard.py apresenta indicadores de qualidade, casos criticos e consumo de tokens.

Para executar, instale Streamlit e Pandas e utilize:

    python -m streamlit run dashboard.py

## Demonstracao final

A implementacao modularizada esta em aulas/demo_producao/.

Para executar a avaliacao local:

    python aulas/demo_producao/main.py

Requer Ollama, modelo llama3.2:latest e configuracao local do Langfuse.

## Objetivo profissional

Demonstrar competencias praticas em Engenharia de IA, Python, avaliacao de LLMs,
observabilidade, monitoramento e integracao de modelos de linguagem.

[Repositorio principal](https://github.com/Rafaelaale/ENGENHARIA-DE-AGENTES-DE-IA-)

[LinkedIn](https://www.linkedin.com/in/rafael-araujo-aa10a4232/)
