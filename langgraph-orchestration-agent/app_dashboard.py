import streamlit as st
import json
from langchain_core.messages import HumanMessage
from email_assistant_memory import graph, memory_store

st.set_page_config(page_title="AI Agent Orchestrator | LangGraph", page_icon="🤖", layout="wide")

st.title("🤖 Enterprise Email Agent Copilot")
st.caption("Arquitetura Corporativa com LangGraph: Semantic Router, HITL & Memória Semântica Local")

with st.sidebar:
    st.header("⚙️ Painel de Controle")
    st.markdown("**Modelo Ativo:** `Ollama / llama3.2:1b`")
    st.markdown("**Arquitetura:** `StateGraph + Memory Store`")
    st.markdown("---")
    st.subheader("📚 Memórias Semânticas Ativas")
    memories = memory_store.search(("emails", "contacts"))
    if memories:
        for m in memories:
            st.info(f"**{m.key}:** {m.value.get('data')}")
    else:
        st.write("Nenhuma memória persistida ainda.")

col1, col2 = st.subplots([1, 1])

with col1:
    st.subheader("📥 Simulação de Entrada de E-mail")
    sender = st.text_input("Remetente", value="Alice Smith <alice.smith@company.com>")
    recipient = st.text_input("Destinatário", value="Sarah Chen <sarah.chen@company.com>")
    subject = st.text_input("Assunto", value="Acompanhamento")
    body = st.text_area("Corpo do E-mail", value="Olá Sarah, podemos agendar uma reunião rápida de alinhamento para amanhã?")

    run_btn = st.button("🚀 Processar com LangGraph", type="primary")

with col2:
    st.subheader("📊 Rastreabilidade e Decisões do Agente")
    if run_btn:
        email_data = {
            "from": sender,
            "to": recipient,
            "subject": subject,
            "body": body
        }

        initial_state = {
            "email_input": email_data,
            "messages": [HumanMessage(content=f"De: {sender}\nAssunto: {subject}\n\n{body}")]
        }

        with st.spinner("Executando nós do Grafo..."):
            result = graph.invoke(initial_state)

        st.success("Fluxo executado com sucesso!")

        tab1, tab2 = st.tabs(["📝 Resposta do Agente", "🔍 Inspeção de Estado"])

        with tab1:
            last_msg = result["messages"][-1] if result.get("messages") else None
            if last_msg:
                st.markdown("### Ação / E-mail Gerado")
                st.code(last_msg.content, language="json" if "{" in last_msg.content else "markdown")

        with tab2:
            st.json({
                "total_mensagens": len(result.get("messages", [])),
                "estado_final": str(result)
            })