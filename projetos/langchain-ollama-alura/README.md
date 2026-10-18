# LangChain + Ollama: prompts, memória e roteamento

Exercícios do curso de LangChain e Python da Alura, com exemplos OpenAI e
adaptações para modelos locais. O código reúne composição de cadeias (LCEL),
saída JSON, histórico de conversa, roteamento com LangGraph e RAG com FAISS.

## Preparação

Execute os comandos **nesta pasta**, com Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

No PowerShell: `.\.venv\Scripts\Activate.ps1`.
Instale o [Ollama](https://ollama.com/) separadamente e mantenha o serviço em
`http://127.0.0.1:11434`; se necessário, execute `ollama serve` em outro terminal.
Não é preciso chave OpenAI para os exemplos locais.

## Exemplos locais

| Comando | O que observar |
| --- | --- |
| `python aulas/aula_04_json_ollama.py` | Prompt e parsing de saída estruturada |
| `python aulas/aula_05_sequencia_ollama.py` | Encadeamento de respostas |
| `python aulas/aula_06_chat_sem_memoria_ollama.py` | Chamadas sem histórico |
| `python aulas/aula_06_chat_com_memoria_ollama.py` | Histórico de mensagens enviado ao modelo |
| `python aulas/aula_langgraph.py` | Roteamento entre consultores de praia e montanha |
| `python aulas/aula_rag_txt.py` | Recuperação de trechos de TXT |
| `python aulas/aula_rag_pdfs.py` | Leitura, divisão, embeddings e recuperação de PDFs |

O chat com memória faz duas perguntas predefinidas para permitir comparar com
a versão sem memória; não é um terminal interativo.
As demos RAG usam os arquivos em [documentos/](documentos/), com materiais GTB
de seguros de novembro de 2023. Eles não representam condições atuais nem
recomendação de contratação. O índice é recriado em memória a cada execução;
não carregue índices pickle recebidos de terceiros.

## Exemplos opcionais com OpenAI

`main.py`, `mainchat.py`, `main_chat.py`, `main_langgraph.py`, `main_rag.py`
e as aulas com sufixo `_professor.py` usam a API OpenAI. Configure
`OPENAI_API_KEY` no ambiente ou em um `.env` local **não versionado**.
Há custos de API e a disponibilidade dos modelos depende da sua conta.
Exemplo, após configurar a chave: `python main_rag.py`.

Apesar do nome, `mainLangGraph.py` usa Ollama e uma cadeia simples;
o grafo local está em `aulas/aula_langgraph.py`.

## Limites e atribuição

Os exercícios não têm avaliação automatizada de qualidade das respostas.
Saídas JSON e decisões de roteamento podem variar com o modelo.
Os nomes dos arquivos didáticos foram mantidos para facilitar a comparação com a fonte.
Consulte a [proveniência](../../docs/PROVENIENCIA.txt) para o commit de origem
e as adaptações de dependências e segurança.
