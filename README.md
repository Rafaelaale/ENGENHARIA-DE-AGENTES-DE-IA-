# Engenharia de Agentes de IA

Portfólio de estudos e projetos de curso em Python, com **LangChain, Ollama e RAG**.
Os arquivos dos projetos estão reunidos neste repositório, não apenas apontados por links externos.

## Para quem avalia este portfólio

Os exemplos mostram a evolução de chamadas a modelos e composição de prompts para
memória de conversa, roteamento e recuperação de documentos. Permitem avaliar o
código, os dados de demonstração e as escolhas de implementação.
São exercícios educacionais, não sistemas de produção nem evidência de resultados comerciais.

## Projetos

| Projeto | O que demonstra | Onde começar |
| --- | --- | --- |
| [LangChain + Ollama](projetos/langchain-ollama-alura/) | Saída JSON, LCEL, memória, roteamento com LangGraph e RAG sobre PDFs/TXT | [Guia de execução](projetos/langchain-ollama-alura/README.md) · [Grafo local](projetos/langchain-ollama-alura/aulas/aula_langgraph.py) |
| [Fundamentos e aplicações RAG](projetos/rag-langchain/) | Embeddings, recuperação semântica, metadados, reranking e comparação com/sem RAG | [Guia de execução](projetos/rag-langchain/README.md) · [Interface de RH](projetos/rag-langchain/projetos/agente-rh/agente_rh.py) |

O segundo projeto incorpora o conteúdo que já existia em `rag-foundations-langchain/`,
incluindo as aulas adicionais de busca e avaliação, sem manter uma cópia duplicada.

## Stack

- Python 3.12, LangChain e LangGraph.
- Ollama para modelos e embeddings locais; OpenAI em exemplos opcionais.
- FAISS para busca vetorial, PyPDF para leitura e Streamlit para a interface de RH.
- Chroma aparece apenas nas referências históricas, fora do ambiente recomendado.

## Como executar

Comece na raiz deste repositório. Use um ambiente virtual separado para cada projeto:

```bash
cd projetos/langchain-ollama-alura
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
python aulas/aula_06_chat_com_memoria_ollama.py
```

Para o RAG, em um novo terminal na raiz do repositório:

```bash
cd projetos/rag-langchain
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull gemma3:1b
python projetos/agente-futebol/agente_futebol.py
```

**Pré-requisito:** instalar o [Ollama](https://ollama.com/) e mantê-lo em execução
em `http://127.0.0.1:11434` (`ollama serve`, em outro terminal, se necessário).
No PowerShell, substitua a ativação por `.\.venv\Scripts\Activate.ps1`.
Modelos são baixados separadamente; tempo e memória necessários dependem da máquina.
As demos acima não exigem chave da OpenAI. Os guias de cada pasta trazem as outras opções.

## Dados, limites e origem

- Políticas de RH e PDFs de medicamentos são sintéticos; não são orientação médica ou jurídica.
- O texto de futebol é didático, não um regulamento oficial. Os documentos GTB são material de estudo, não orientação atual sobre seguros.
- Exemplos OpenAI exigem chave pessoal e podem gerar custos. Nunca envie credenciais ao Git.
- Ambientes, segredos, caches, índices vetoriais e modelos baixados ficam fora do versionamento.
- [Proveniência da consolidação](docs/PROVENIENCIA.txt): fontes, commits, arquivos preservados e adaptações.
