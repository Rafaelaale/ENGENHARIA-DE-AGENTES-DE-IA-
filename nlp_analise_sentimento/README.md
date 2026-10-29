# Análise de sentimentos em avaliações

Projeto educacional de Processamento de Linguagem Natural (PLN) para explorar, preparar e classificar avaliações de texto.

## Conteúdo

- `Notebooks/Aula_1.ipynb` a `Notebooks/Aula_5.ipynb`: sequência de estudos e experimentos de análise de sentimentos.
- `dashboard.py`: painel interativo para explorar um arquivo CSV de avaliações.

## Resultado registrado

No experimento documentado no notebook Aula 5, foi registrada acurácia de 91,85% usando todas as features e n-grams. Esse valor corresponde à configuração testada no notebook; não é garantia de desempenho em outros dados. Para uma avaliação mais completa, considere também precisão, recall, F1-score e matriz de confusão.

## Executar o dashboard

Requer Python e as bibliotecas `streamlit`, `pandas` e `plotly`.

```bash
pip install streamlit pandas plotly
streamlit run dashboard.py
```

No painel, carregue um CSV com uma coluna `avaliacao`. Se houver uma coluna `sentimento`, o painel também exibirá a distribuição dos rótulos existentes. O dashboard não treina nem executa um modelo de classificação.

## Dados e privacidade

Os arquivos de dados não foram incluídos. Use um CSV autorizado para análise e remova informações pessoais antes de compartilhá-lo ou publicá-lo.

## Reprodutibilidade

Os notebooks registram os procedimentos e resultados dos experimentos. Acurácia depende dos dados, da preparação do texto e da configuração utilizada.
