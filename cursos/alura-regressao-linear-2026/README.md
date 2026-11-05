# Regressao Linear - Alura

## Aula: Identificando homocedasticidade

Analise estatistica de precos imobiliarios com Python.

### Tecnologias
- Python
- Pandas
- NumPy
- Statsmodels
- Seaborn
- Matplotlib

### Modelo
Regressao linear multipla utilizando:
- Area do primeiro andar
- Existencia de segundo andar
- Quantidade de banheiros
- Cozinha excelente

### Resultados
- 1.006 imoveis no treinamento
- R2: 0.651129
- Teste de Breusch-Pagan
- Evidencia estatistica de heterocedasticidade

### Executar
Instalar dependencias:

pip install pandas numpy statsmodels seaborn matplotlib

Executar a partir desta pasta:

python aulas/aula_03_homocedasticidade.py

### Observacoes
O modelo apresenta maior dispersao dos residuos
nos imoveis de preco previsto elevado.

Os resultados sao educacionais.