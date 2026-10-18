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
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
    ("placeholder", "{historico}"),
    ("human", "{query}"),
])

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

print("Digite sair para encerrar.")

while True:
    uma_pergunta = input("Você: ")
    if uma_pergunta.strip().lower() == "sair":
        break

    resposta = cadeia_com_memoria.invoke(
        {"query": uma_pergunta},
        config={"configurable": {"session_id": sessao}},
    )

    print("Sr. Passeios:", resposta)
