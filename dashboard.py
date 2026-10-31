import streamlit as st

st.set_page_config(page_title="Trilha de Engenharia de IA", page_icon="🤖")

assuntos = [
    "Redes neurais com PyTorch",
    "Funções de perda",
    "Classificação e regressão com PyTorch",
    "Otimização de redes neurais e taxa de aprendizado",
    "Otimizadores: SGD, momentum, Nesterov, AdaGrad, AdaDelta, RMSprop e Adam",
    "Batches, iterações e épocas",
    "Datasets e DataLoader, incluindo MNIST",
    "Dataset customizado de aluguel de bicicletas",
    "Fluxos de treinamento e validação",
    "Dashboard com LangGraph",
]

st.title("Trilha de Engenharia de IA")
st.write("Painel de consulta dos assuntos registrados neste repositório.")

assunto = st.selectbox("Selecione um assunto", assuntos)

st.subheader(assunto)
st.write("Consulte os arquivos e materiais correspondentes a este assunto no repositório.")
st.caption(f"Total de assuntos listados: {len(assuntos)}")
