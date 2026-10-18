from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
)

query_model = OllamaLLM(model="gemma3:1b", temperature=0)
pergunta = "Qual é a política de férias?"

# Rewrite: reescreve a pergunta para facilitar a busca
rewriter_prompt_template = """
Gere uma consulta de pesquisa para um banco de dados vetorial (Vector DB)
a partir da pergunta de um usuário, permitindo uma resposta mais precisa
por meio da busca semântica.
Retorne somente a consulta revisada.

Pergunta do usuário: {user_question}

Consulta revisada do Vector DB:
"""

rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)
rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

consulta_reescrita = rewriter_chain.invoke({"user_question": pergunta})
print("\nConsulta reescrita:")
print(consulta_reescrita)

# Multiple Query: gera consultas alternativas
multi_query_prompt_template = """Você é um assistente de modelo de linguagem de IA.
Gere cinco versões diferentes da pergunta para recuperar documentos relevantes
de um banco de dados vetorial. Mantenha o mesmo assunto da pergunta original.
Forneça as perguntas alternativas separadas por quebras de linha.

Pergunta original: {question}
"""

multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template)
multi_query_chain = (
    multi_query_prompt
    | query_model
    | CommaSeparatedListOutputParser()
)

consultas_alternativas = multi_query_chain.invoke({"question": pergunta})
print("\nConsultas alternativas:")
for consulta in consultas_alternativas:
    print(f"- {consulta}")

# HyDE: gera um documento hipotético relacionado à pergunta
hyde_prompt_template = """
Responda à pergunta da melhor forma possível em no mínimo dois parágrafos.
Mantenha o assunto da pergunta, sem introduzir temas diferentes.
Pergunta: {user_question}
Resposta:
"""

hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | query_model | StrOutputParser()

documento_hipotetico = hyde_chain.invoke({"user_question": pergunta})
print("\nDocumento hipotético (HyDE):")
print(documento_hipotetico)