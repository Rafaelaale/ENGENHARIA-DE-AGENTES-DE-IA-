import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Painel de análise de sentimentos", page_icon="💬", layout="wide")

st.title("Análise de sentimentos em avaliações")
st.caption("Painel exploratório para arquivos CSV. Não classifica textos nem substitui os experimentos dos notebooks.")

arquivo = st.file_uploader("Carregue um CSV autorizado", type=["csv"])

if arquivo is None:
    st.info("Carregue um CSV contendo a coluna `avaliacao`. Se houver uma coluna `sentimento`, também será exibida sua distribuição.")
    st.stop()

try:
    dados = pd.read_csv(arquivo)
except Exception as erro:
    st.error(f"Não foi possível ler o CSV: {erro}")
    st.stop()

if "avaliacao" not in dados.columns:
    st.error("O CSV precisa conter uma coluna chamada `avaliacao`.")
    st.write("Colunas encontradas:", ", ".join(map(str, dados.columns)))
    st.stop()

dados["avaliacao"] = dados["avaliacao"].fillna("").astype(str)
total = len(dados)
preenchidas = dados["avaliacao"].str.strip().ne("").sum()

col1, col2, col3 = st.columns(3)
col1.metric("Registros", f"{total:,}".replace(",", "."))
col2.metric("Avaliações preenchidas", f"{preenchidas:,}".replace(",", "."))
col3.metric("Colunas no arquivo", len(dados.columns))

st.subheader("Prévia das avaliações")
st.dataframe(dados.head(20), use_container_width=True)

if "sentimento" in dados.columns:
    rotulos = dados["sentimento"].fillna("Sem rótulo").astype(str).value_counts().rename_axis("sentimento").reset_index(name="quantidade")
    st.subheader("Distribuição dos rótulos existentes")
    st.plotly_chart(px.bar(rotulos, x="sentimento", y="quantidade", color="sentimento"), use_container_width=True)
    st.caption("O gráfico apresenta os rótulos já existentes no CSV; não são previsões geradas pelo painel.")
else:
    st.warning("Não há coluna `sentimento`; a distribuição de rótulos não pode ser exibida.")

st.subheader("Buscar avaliações")
termo = st.text_input("Filtrar pelo texto")
if termo:
    resultado = dados[dados["avaliacao"].str.contains(termo, case=False, na=False)]
    st.write(f"{len(resultado)} resultado(s)")
    st.dataframe(resultado, use_container_width=True)

st.caption("Privacidade: utilize apenas dados autorizados e remova informações pessoais antes de compartilhar o arquivo.")
