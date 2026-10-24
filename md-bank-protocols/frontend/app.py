import streamlit as st
import requests

API_URL = "http://supervisor:8000/chat"

st.set_page_config(page_title="MDBank Assistente", page_icon="🏦")

st.title("MDBank Assistente")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Digite sua pergunta")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "message": user_input,
        "session_id": "123",
        "client_id": "123"
    }

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    resposta = data.get("resposta", "Erro na resposta do servidor")
                else:
                    resposta = f"Erro ao chamar API ({response.status_code})"
            except Exception as e:
                resposta = f"Erro de conexão com o supervisor: {e}"

            st.markdown(resposta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta
    })
