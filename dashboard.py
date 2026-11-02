from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
GITHUB = "https://github.com/Rafaelaale/ENGENHARIA-DE-AGENTES-DE-IA-"
CNN = "03-Redes-Neurais-Convolucionais-PyTorch"

st.set_page_config(
    page_title="Engenharia de Agentes de IA | Portfólio",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.6rem;
}
.hero {
    padding: 30px;
    border-radius: 14px;
    background: linear-gradient(115deg, #102338, #17576a);
    color: white;
}
.hero h1 {
    color: white;
}
.hero p {
    color: #e1f0f4;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Engenharia de Agentes de IA</h1>
<p>Python · Machine Learning · Deep Learning · NLP · RAG · LangGraph</p>
<p><strong>Rafael Alexandre Oliveira Araujo</strong></p>
<p>Portfólio técnico | Análise e Desenvolvimento de Sistemas</p>
</div>
""", unsafe_allow_html=True)

st.write("")

a, b, c, d = st.columns(4)

a.metric("Etapas da trilha", "7")
b.metric("LeNet-5: teste completo", "50,90%")
c.metric("Imagens CIFAR-10 testadas", "10.000")
d.metric("VGG16 + SVM: amostra", "70%")

st.caption(
    "Acurácia da VGG16 + SVM medida em apenas 50 imagens; "
    "não é diretamente comparável ao teste completo da LeNet-5."
)

projetos = [
    (
        "01",
        "Fundamentos de Redes Neurais",
        "redes_neurais_pytorch",
        "PyTorch, MLP, gradientes e otimização"
    ),
    (
        "02",
        "Redes Neurais Recorrentes",
        "Redes_Neurais_Recorrentes_PyTorch_2",
        "RNN, GRU, LSTM e classificação de sequências"
    ),
    (
        "03",
        "Redes Neurais Convolucionais",
        CNN,
        "LeNet-5, CIFAR-10, VGG16-BN, SVM e fine-tuning"
    ),
    (
        "04",
        "Processamento de Linguagem Natural",
        "nlp_analise_sentimento",
        "Análise de sentimentos e classificação"
    ),
    (
        "05",
        "RAG e LangChain",
        "rag-foundations-langchain",
        "Embeddings, Chroma, Ollama e busca semântica"
    ),
    (
        "06",
        "Orquestração de Agentes",
        "langgraph-orchestration-agent",
        "ReAct, HITL, persistência e multiagentes"
    ),
    (
        "07",
        "Integração de Sistemas",
        "md-bank-protocols",
        "Agentes, componentes e Docker Compose"
    ),
]

aba1, aba2, aba3 = st.tabs([
    "Projetos",
    "Métricas e gráficos",
    "Perfil profissional"
])

with aba1:
    st.header("Trilha de Engenharia de IA")

    st.write(
        "Todos os projetos pertencem à mesma trilha, "
        "com progressão dos fundamentos até agentes inteligentes."
    )

    for numero, titulo, pasta, tecnologias in projetos:

        with st.container(border=True):
            st.subheader(f"{numero} | {titulo}")

            st.write(tecnologias)

            st.markdown(
                f"[Examinar código no GitHub]"
                f"({GITHUB}/tree/main/{pasta})"
            )

with aba2:
    st.header("Resultados experimentais")

    resultados = pd.DataFrame([
        {
            "Projeto": "LeNet-5",
            "Resultado": "50,90%",
            "Amostra": "10.000 imagens de teste",
            "Observação": "5 épocas completas"
        },
        {
            "Projeto": "LeNet-5: validação",
            "Resultado": "51,46%",
            "Amostra": "5.000 imagens",
            "Observação": "Melhor época: 4"
        },
        {
            "Projeto": "VGG16-BN + SVM",
            "Resultado": "70%",
            "Amostra": "50 imagens",
            "Observação": "Teste exploratório"
        },
        {
            "Projeto": "VGG16-BN fine-tuning",
            "Resultado": "Pesos atualizados",
            "Amostra": "Teste reduzido",
            "Observação": "Validação técnica"
        },
        {
            "Projeto": "NLP",
            "Resultado": "91,85%",
            "Amostra": "Experimento documentado",
            "Observação": "Consultar README do projeto"
        },
    ])

    st.dataframe(
        resultados,
        hide_index=True,
        use_container_width=True
    )

    historico = (
        BASE
        / CNN
        / "resultados"
        / "aula21_historico_treinamento.csv"
    )

    if historico.exists():

        df = pd.read_csv(historico)

        df = df.sort_values("epoca").set_index("epoca")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loss por época")

            st.line_chart(
                df[["loss_treino", "loss_validacao"]]
            )

        with col2:
            st.subheader("Acurácia por época")

            st.line_chart(
                df[
                    [
                        "acuracia_treino",
                        "acuracia_validacao"
                    ]
                ]
            )

        with st.expander("Ver histórico completo"):
            st.dataframe(
                df,
                use_container_width=True
            )

    else:
        st.info(
            "Histórico CSV da LeNet-5 não encontrado."
        )

    fine = (
        BASE
        / CNN
        / "resultados"
        / "aula23_historico_fine_tuning.csv"
    )

    if fine.exists():

        st.subheader("Fine-tuning: teste funcional")

        dados_fine = pd.read_csv(fine)

        st.line_chart(
            dados_fine.set_index("epoca")[
                ["loss_treino", "loss_validacao"]
            ]
        )

        st.caption(
            "Dados de uma amostra muito pequena; "
            "não representam desempenho de produção."
        )

    st.markdown(
        f"[Abrir resultados completos no GitHub]"
        f"({GITHUB}/tree/main/{CNN}/resultados)"
    )

with aba3:
    st.header("Apresentação para recrutadores")

    st.markdown("""
    **Perfil**

    Estudante de Análise e Desenvolvimento de Sistemas,
    em transição profissional da área de saúde
    para tecnologia.

    **Competências práticas**

    - Python e programação aplicada.
    - Treinamento e validação de redes neurais.
    - PyTorch, Deep Learning e visão computacional.
    - NLP e classificação de textos.
    - Transfer Learning e Fine-tuning.
    - RAG, LangChain e Ollama.
    - Agentes e fluxos multiagentes com LangGraph.
    - Git, GitHub e documentação técnica.

    **Objetivo profissional**

    Atuar em desenvolvimento Python,
    Inteligência Artificial aplicada,
    Machine Learning, dados e automação.
    """)

    st.markdown(
        "[GitHub completo](https://github.com/Rafaelaale)"
    )

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/rafael-araujo-aa10a4232/)"
    )

st.divider()

st.caption(
    "Portfólio de Engenharia de Agentes de IA. "
    "Resultados experimentais apresentados com suas limitações."
)
