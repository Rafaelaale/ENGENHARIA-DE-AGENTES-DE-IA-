from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de roteiro de viagens. Responda em português."),
    ("human", "Crie um roteiro de {numero_dias} dias, para uma família com {numero_criancas} crianças, que gosta de {atividade}.")
])

resposta = (prompt | modelo).invoke({
    "numero_dias": 7,
    "numero_criancas": 2,
    "atividade": "praia"
})

print(resposta.content)
