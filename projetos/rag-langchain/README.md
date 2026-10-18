# Arquiteturas RAG com LLMs

Estudos de embeddings, busca semântica e aplicações RAG com LangChain e Ollama.
Esta pasta preserva o projeto anteriormente chamado `rag-foundations-langchain`,
incluindo as aulas de avaliação e busca adicionais ao repositório de origem.

## O que o projeto demonstra

- **Futebol:** fluxo de TXT → chunks → embeddings → busca FAISS → resposta local.
- **RH:** PDFs, classificação de metadados, reranking por LLM e interface Streamlit.
- **Medicamentos:** recuperação de trechos de PDFs sintéticos com exibição de fontes.
- **Aulas:** embeddings, reescrita de consultas, múltiplas consultas, HyDE e
  comparação de respostas com/sem RAG usando um avaliador LLM.

As demos usam Python, LangChain, FAISS e Ollama (`nomic-embed-text` para embeddings
e `gemma3:1b` para respostas), sem créditos OpenAI.
Chroma permanece apenas nos exemplos históricos em `referencias/professor/`.

## Preparação

Execute os comandos **nesta pasta**, com Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull gemma3:1b
```

No PowerShell: `.\.venv\Scripts\Activate.ps1`.
Instale o [Ollama](https://ollama.com/) separadamente e mantenha o serviço em
`http://127.0.0.1:11434`; se necessário, execute `ollama serve` em outro terminal.

## Como rodar

```bash
# Demonstração de busca e respostas sobre futebol
python projetos/agente-futebol/agente_futebol.py

# Interface de políticas de RH; abra o endereço local indicado pelo Streamlit
python -m streamlit run projetos/agente-rh/agente_rh.py --server.address 127.0.0.1

# Demonstração de recuperação de trechos de medicamentos sintéticos
python projetos/agente-bulas/agente_bulas.py
```

Futebol e medicamentos imprimem perguntas predefinidas, respostas e/ou contexto.
Na interface de RH, experimente: “Com quanto tempo de antecedência devo solicitar férias?”.
O exemplo de futebol executa `ollama stop` após consultas para liberar o modelo.
As três demos recriam os índices FAISS em memória; não leem índices pickle do disco.
O Streamlit mantém recursos em cache durante a sessão.

### Aulas complementares preservadas

```bash
python aulas/aula_03_embeddings_local.py
python aulas/aula_buscas_vetoriais.py
python aulas/aula_avaliacao_local.py
python aulas/aula_validacao_comparativa_local.py
```

As duas últimas usam `langchain-classic` e avaliação por LLM sobre poucos exemplos,
não um benchmark nem comprovação de ganho de precisão.
Os geradores `gerar_bulas.py` e `gerar_pdfs_rh.py` usam ReportLab e escrevem no
diretório corrente; os PDFs de demonstração já estão em `documentos/`, portanto
não é necessário executá-los.

## Dados e limites

- PDFs de medicamentos são sintéticos, **não bulas oficiais ou orientação médica**.
- Políticas de RH são fictícias; textos de futebol são resumos didáticos,
  não regulamentos oficiais.
- RAG pode recuperar trechos inadequados e gerar respostas incorretas;
  sempre confira as fontes. O reranking também depende do modelo.
- `referencias/professor/` foi preservada para atribuição e comparação, mas contém
  imports antigos, exemplos OpenAI e Chroma. Esses arquivos **não fazem parte do
  ambiente executável recomendado**. Chroma não foi incluído por alertas de
  segurança sem versão corrigida na verificação da migração.
- Segredos, ambientes, caches, modelos e índices locais não devem ser versionados.

Veja a [proveniência](../../docs/PROVENIENCIA.txt) para fontes e alterações.