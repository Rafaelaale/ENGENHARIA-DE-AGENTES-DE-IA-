from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

modelo = ChatOllama(model="llama3.2:1b", temperature=0.5)

mensagens = [
    SystemMessage(content="Você é um assistente de viagens. Responda em português e use o histórico da conversa.")
]

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for uma_pergunta in lista_perguntas:
    mensagens.append(HumanMessage(content=uma_pergunta))
    resposta = modelo.invoke(mensagens)
    mensagens.append(AIMessage(content=resposta.content))
    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta.content, "\n")
