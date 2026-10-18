# Fine-tuning versus RAG

Fine-tuning retreina um modelo existente com dados próprios, geralmente nas últimas camadas. Exige GPU, tempo e custo, e altera o comportamento do modelo.

RAG não retreina o modelo. Ele consulta uma base externa no momento da pergunta, reduz custo e facilita a atualização dos documentos.

O curso usa RAG com LangChain para ampliar o contexto do LLM.
