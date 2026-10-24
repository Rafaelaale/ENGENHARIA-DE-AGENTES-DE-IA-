# 🤖 Engenharia de Agentes Autônomos: LangGraph, RAG & Executive Dashboards

## 📌 Visão Geral do Repositório
Solução corporativa voltada para a arquitetura de agentes autônomos utilizando grafos de decisão com LangGraph, recuperação contextual via RAG e interfaces executivas com Streamlit. Foco em execução local com Ollama, determinismo e Human-in-the-Loop (HITL).

---

## 🏗️ Módulos Principais

### 1. `langgraph-orchestration-agent/` (Core do Agente & HITL)
- **LangGraph State Management:** Grafos de decisão, controle de estado e memória semântica com checkpoints.
- **Human-in-the-Loop (HITL):** Pontos de interrupção determinísticos para aprovação humana em ações críticas.
- **LLM Local:** Processamento via Ollama (`llama3.2:1b`), garantindo privacidade e custo zero.
- **Executive Dashboard (`app_dashboard.py`):** Visualização do grafo, simulador de triagem e painel de auditoria.

### 2. `rag-foundations-langchain/` (Fundamentos de RAG)
- **Base Vetorial:** ChromaDB local para recuperação semântica de documentos.
- **Embeddings:** Execução local com Nomic-Embed-Text e Gemma via Ollama CLI.

---

## ⚙️ Instruções de Execução

```powershell
# Execução do agente via terminal (com HITL interativo)
cd langgraph-orchestration-agent
py -3.11 email_assistant_memory.py

# Execução do Dashboard Executivo
streamlit run app_dashboard.py
```
