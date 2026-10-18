# Arquitetura geral do RAG

RAG significa Retrieval-Augmented Generation: recuperar informacao externa e gerar a resposta com esse contexto.

Fluxo:
1. Reunir PDFs, planilhas, documentos ou bases de dados.
2. Padronizar e dividir o texto.
3. Transformar o texto em embeddings, que sao vetores numericos.
4. Salvar os vetores em um vector store.
5. Transformar a pergunta do usuario em vetor e buscar os trechos mais parecidos.
6. Entregar esses trechos ao LLM para gerar a resposta final.

O LangChain orquestra esse fluxo. O RAG nao retreina o modelo; ele amplia o contexto no momento da consulta.
