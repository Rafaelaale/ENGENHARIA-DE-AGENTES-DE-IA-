from langchain_ollama import ChatOllama

modelo = ChatOllama(model="llama3.2:1b", temperature=0.5)

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for uma_pergunta in lista_perguntas:
    resposta = modelo.invoke(uma_pergunta)
    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta.content, "\n")
