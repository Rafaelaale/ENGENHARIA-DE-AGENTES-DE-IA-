from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

prompt_consultor = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor de viagens. Responda em português."),
    ("human", "{query}"),
])

assistente = prompt_consultor | modelo | StrOutputParser()

print(assistente.invoke({"query": "Quero férias em praias no Brasil."}))
