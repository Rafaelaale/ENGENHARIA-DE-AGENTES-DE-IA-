# Conceitos de LLM

Um LLM recebe um texto e projeta a continuação com base em bilhões ou trilhões de parâmetros.

A arquitetura transformer usa atenção, encoder e decoder para interpretar o contexto. Em "Lucas está próximo do ponto", o sentido depende da frase completa, não da palavra isolada.

Limitações: conhecimento desatualizado, falta de regras específicas do negócio e alucinação, quando a resposta não tem base suficiente.

O curso usa RAG com LangChain para acrescentar documentos externos e tornar o modelo especialista nesses dados.
