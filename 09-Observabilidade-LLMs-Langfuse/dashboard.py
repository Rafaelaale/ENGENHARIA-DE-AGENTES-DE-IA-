import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Observabilidade de LLMs | Portfolio",
    page_icon="📊",
    layout="wide"
)

st.title("Observabilidade de LLMs")
st.caption("Python | Langfuse | Ollama | LLM-as-a-Judge")
st.write("Projeto experimental de Engenharia de IA — Rafael Alexandre Oliveira Araujo")

st.info(
    "Resultados de uma execucao local com 5 exemplos. "
    "Nao representam metricas em tempo real nem desempenho de producao."
)

a, b, c, d = st.columns(4)

a.metric("Exemplos avaliados", 5)
b.metric("Score medio", "1,8 / 5")
c.metric("Casos criticos", 5)
d.metric("Fallbacks aplicados", 5)

st.subheader("Qualidade e monitoramento")

dados = pd.DataFrame({
    "Indicador": [
        "Casos avaliados",
        "Casos criticos",
        "Casos para revisao",
        "Fallbacks aplicados"
    ],
    "Quantidade": [5, 5, 5, 5]
})

st.bar_chart(dados.set_index("Indicador"))

st.subheader("Consumo de tokens")

tokens = pd.DataFrame({
    "Tipo": ["Entrada", "Saida"],
    "Tokens": [5032, 1779]
})

st.bar_chart(tokens.set_index("Tipo"))

st.subheader("Analise de qualidade")

st.warning(
    "Todos os casos permaneceram criticos apos o fallback. "
    "O pipeline identificou as falhas, mas a estrategia de mitigacao "
    "ainda precisa ser aprimorada."
)

st.subheader("Competencias demonstradas")

st.markdown("""
- Instrumentacao e tracing com Langfuse
- Avaliacao automatica LLM-as-a-Judge
- Classificacao de severidade
- Fallback automatico
- Reavaliacao de respostas
- Analise de casos prioritarios
- Metricas de tokens
- Modularizacao Python
- Integracao com Ollama
""")

st.subheader("Arquitetura implementada")

st.code(
    "Dataset -> Geracao LLM -> Julgamento -> Severidade -> "
    "Fallback -> Reavaliacao -> Metricas -> Langfuse",
    language="text"
)

st.markdown(
    "[Repositorio principal](https://github.com/Rafaelaale/ENGENHARIA-DE-AGENTES-DE-IA-)"
)
