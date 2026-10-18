from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios. Responda em português e de forma curta."),
    ("placeholder", "{historico}"),
    ("human", "{query}"),
])

cadeia = prompt_sugestao | modelo | StrOutputParser()
memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao_id: str):
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

perguntas = [
    "Quero visitar um lugar famoso no Brasil. Sugira apenas um.",
    "Qual é a melhor época para visitar esse lugar?",
]

for pergunta in perguntas:
    resposta = cadeia_com_memoria.invoke(
        {"query": pergunta},
        config={"configurable": {"session_id": sessao}},
    )
    print("Você:", pergunta)
    print("Sr. Passeios:", resposta)
    print()
