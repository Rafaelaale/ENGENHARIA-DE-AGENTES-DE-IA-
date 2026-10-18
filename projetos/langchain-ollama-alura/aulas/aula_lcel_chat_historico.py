from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
historico = []

print("Digite sair para encerrar.")

while True:
    query = input("Você: ")
    if query.strip().lower() == "sair":
        break

    resposta = cadeia.invoke({
        "historico": historico,
        "query": query,
    })

    print("Sr. Passeios:", resposta)
    historico.extend([
        HumanMessage(content=query),
        AIMessage(content=resposta),
    ])
