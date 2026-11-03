# Transformers com PyTorch

Projeto de estudo desenvolvido durante a formacao em
Transformers com PyTorch da Alura.

## Conteudos implementados

- Carregamento e limpeza de textos da Wikipedia
- Tokenizacao com BERT em portugues
- Positional Encoding
- Self-Attention
- Multi-Head Attention
- Feed-Forward Network
- Modelo Transformer autorregressivo
- Treinamento com PyTorch e AdamW
- Avaliacao de loss e perplexidade
- Visualizacao dos pesos de atencao
- Interface interativa de geracao de texto

## Organizacao

- `aulas/`: codigos Python das aulas
- `dados/`: amostra textual utilizada nos experimentos
- `resultados/`: metricas, relatorios e graficos

## Resultados

O experimento de treinamento utilizou uma amostra pequena,
com aproximadamente 1.156 tokens e 80 iteracoes.

A arquitetura e os procedimentos de inferencia foram
executados, mas o modelo ainda nao produz textos coerentes.

Os resultados demonstram o funcionamento tecnico da
arquitetura, nao um modelo de linguagem pronto para producao.

## Tecnologias

Python, PyTorch, Transformers, Hugging Face, NumPy
e Matplotlib.

## Execucao

Instalar as dependencias utilizadas pelas aulas e executar
os scripts Python na sequencia.

Os checkpoints `.pt` e o ambiente virtual `.venv`
nao estao incluidos no repositorio.

Para executar a interface com os pesos treinados, e
necessario gerar o checkpoint nas aulas de treinamento
e avaliacao.

## Objetivo

Demonstrar conhecimentos praticos em arquiteturas
Transformer, Deep Learning, NLP, treinamento,
avaliacao e inferencia de modelos de linguagem.
