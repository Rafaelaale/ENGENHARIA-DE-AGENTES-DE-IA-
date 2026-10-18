# Arquiteturas RAG com LLMs

Estudos de RAG com LangChain: embeddings, busca semantica e agentes.

Stack: Python, LangChain, Chroma e Ollama (`nomic-embed-text` para embeddings e `gemma3:1b` para respostas). A versao local nao usa credito da OpenAI.

- Embeddings e busca semantica
- Agente farmaceutico com PDFs sinteticos, nao bulas oficiais
- Agente de RH com politicas ficticias
- Chat de regras de futebol com texto didatico, nao regulamento oficial

## Como rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
ollama serve
python referencias\professor\projeto_04_regras_futebol_local.py
```

Os indices locais (Chroma e FAISS) sao gerados ou reutilizados durante a execucao e nao ficam no Git.