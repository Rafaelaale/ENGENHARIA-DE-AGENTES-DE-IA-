import streamlit as st
import os
from email_assistant_memory import graph

st.set_page_config(page_title='Executive AI Agent Dashboard', layout='wide', page_icon='🤖')
st.title('🤖 LangGraph Agent Dashboard & Executive Monitor')
st.markdown('Painel executivo para monitoramento de agentes, controle Human-in-the-Loop (HITL) e auditoria de decisoes.')

tab1, tab2, tab3 = st.tabs(['📊 Topologia do Grafo', '📥 Triagem de E-mails', '⚙️ Auditoria & HITL'])

with tab1:
    st.subheader('Topologia do Grafo de Decisao')
    img = os.path.join(os.path.dirname(__file__), 'grafo_email.png')
    if os.path.exists(img):
        st.image(img, caption='Topologia LangGraph com HITL e Memoria Semantica', use_container_width=True)
    else:
        st.warning('grafo_email.png nao encontrado.')

with tab2:
    st.subheader('Triagem e Execucao')
    remetente = st.text_input('Remetente', value='alice.smith@empresa.com')
    assunto = st.text_input('Assunto', value='Reuniao de Alinhamento')
    conteudo = st.text_area('Corpo do E-mail', value='Podemos agendar a reuniao amanha com 2 participantes?')
    if st.button('Executar Agente de Triagem', type='primary'):
        estado = {'email_sender': remetente, 'email_subject': assunto, 'email_body': conteudo, 'messages': []}
        cfg = {'configurable': {'thread_id': 'dashboard-thread-1'}}
        res = graph.invoke(estado, config=cfg)
        st.success('Processamento concluido!')
        st.json(res)

with tab3:
    st.subheader('Auditoria e Metricas')
    st.metric(label='Status do Modelo Local', value='Ollama (llama3.2:1b)', delta='Operacional')
    st.info('Controle estrito HITL ativo para acoes de agendamento.')
