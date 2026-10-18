# Prompt tradicional versus prompt com RAG

No prompt tradicional, o LLM responde apenas com o conhecimento geral do treinamento. Regras internas, como cesta de Natal ou home office, nao estao nessa base, entao a resposta tende a ser generica ou alucinada.

No prompt com RAG, os documentos da empresa sao transformados em embeddings e salvos em um banco vetorial. A pergunta recupera os trechos relevantes e o LLM gera a resposta com esse contexto.

O objetivo do RAG e aumentar a precisao e o embasamento, nao apenas ampliar o texto enviado ao modelo.
