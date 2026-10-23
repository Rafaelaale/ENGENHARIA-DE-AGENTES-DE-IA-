# 🧠 Engenharia de Agentes de IA — Portfolio Técnico

Portfólio de engenharia de software e inteligência artificial voltado a arquiteturas avançadas de agentes autônomos, sistemas RAG e orquestração de grafos orientados a estado com execução local (custo zero).

---

## 📂 Projetos no Repositório

### 1. [🤖 LangGraph Orchestration Agent](./langgraph-orchestration-agent)
- **Foco:** Orquestração de agentes corporativos com controle de estado determístico.
- **Destaques:**
  - Máquina de estados finita com **LangGraph** (`StateGraph`).
  - **Human-in-the-Loop (HITL)** para aprovação manual de ações sensíveis (agendamento).
  - Roteamento semântico de intenções (Triagem autônoma de e-mails).
  - **Memória Semântica de Longo Prazo** para persistência de contatos.
  - Interface interativa em **Streamlit** e execução 100% on-premises via **Ollama (`llama3.2:1b`)**.

### 2. [📚 RAG Foundations & Local Agents](./rag-foundations-langchain)
- **Foco:** Arquitetura de recuperação de informação aumentada por geração (RAG).
- **Destaques:**
  - Embeddings locais com **Ollama** (`nomic-embed-text`) e bases vetoriais locais (**Chroma**).
  - Padrão **ReAct** (Reasoning + Acting) para busca semântica em documentos confidenciais.
  - Respostas fundamentadas com controle de contexto e redução de alucinações.

---

## 🛠️ Stack Tecnológica Consolidada

| Categoria | Tecnologias |
| :--- | :--- |
| **Orquestração & Frameworks** | LangGraph, LangChain, LangChain-Core |
| **Modelos Locais (LLM & Embeddings)** | Ollama (`llama3.2:1b`, `nomic-embed-text`) |
| **Bases Vetoriais & Persistência** | Chroma, MemoryStore |
| **Interfaces de Apresentação** | Streamlit |
| **Linguagem & Ambiente** | Python 3.11, PowerShell, Git |

---

## 🎯 Objetivo de Engenharia

Demonstrar a aplicação prática de padrões corporativos de IA Generativa:
- Governança e auditoria com Human-in-the-Loop.
- Eliminação de custos de API e compliance de dados com modelos locais.
- Arquiteturas orientadas a eventos e grafos auditáveis.