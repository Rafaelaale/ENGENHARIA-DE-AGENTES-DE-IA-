from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

modelo = ChatOllama(model="llama3.2:1b", temperature=0)
modelo_json = ChatOllama(model="llama3.2:1b", temperature=0, format="json")

prompt_cidade = PromptTemplate.from_template(
"""Interesse: {interesse}
Responda somente JSON.
Chaves obrigatorias: cidade e motivo.
As duas devem ser texto, nao lista.
{{"cidade":"Rio de Janeiro","motivo":"praias famosas"}}
Nao use foo, bar ou baz.
JSON:"""
)

prompt_restaurantes = PromptTemplate.from_template(
"""Cidade recebida: {cidade}
Responda somente JSON.
Copie exatamente essa cidade.
Chaves obrigatorias: cidade e restaurantes.
{{"cidade":"Rio de Janeiro","restaurantes":"Aprazivel e Zaza Bistro"}}
Nao use foo, bar ou baz.
JSON:"""
)

prompt_cultural = PromptTemplate.from_template(
"Sugira atividades e locais culturais em {cidade}"
)

def texto(valor):
    if isinstance(valor, list):
        return ", ".join(map(str, valor))
    return "" if valor is None else str(valor)

def limpar_destino(dados):
    if not isinstance(dados, dict) or not dados.get("cidade") or dados.get("cidade") in ("foo", "bar", "baz"):
        dados = {"cidade": "Rio de Janeiro", "motivo": "praias famosas"}
    return {"cidade": texto(dados.get("cidade")), "motivo": texto(dados.get("motivo"))}

def limpar_restaurantes(dados, cidade):
    if not isinstance(dados, dict) or "foo" in dados or not dados.get("restaurantes"):
        dados = {"cidade": cidade, "restaurantes": "restaurantes locais a beira-mar"}
    return {"cidade": cidade, "restaurantes": texto(dados.get("restaurantes"))}

def mostrar(nome, valor):
    print(f"\n=== {nome} ===")
    print(valor)
    return valor

cadeia_1 = prompt_cidade | modelo_json | JsonOutputParser()
cadeia_2 = prompt_restaurantes | modelo_json | JsonOutputParser()
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

destino = mostrar("destino", limpar_destino(cadeia_1.invoke({"interesse": "praias"})))
restaurantes = mostrar("restaurantes", limpar_restaurantes(cadeia_2.invoke(destino), destino["cidade"]))
cultural = mostrar("cultural", cadeia_3.invoke(restaurantes))
