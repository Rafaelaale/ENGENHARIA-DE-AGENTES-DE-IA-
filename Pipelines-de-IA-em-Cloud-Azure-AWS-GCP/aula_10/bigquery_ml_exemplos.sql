-- EXEMPLOS PARA GOOGLE BIGQUERY
-- Substituir SEU_PROJETO pelo ID real.
-- Requer tabela carregada no BigQuery e faturamento configurado.
-- NAO executado neste laboratorio.

CREATE OR REPLACE VIEW
  `SEU_PROJETO.churn_dataset.churn_features`
AS
SELECT
  CreditScore,
  Age,
  Tenure,
  Balance,
  NumOfProducts,
  Exited
FROM `SEU_PROJETO.churn_dataset.churn_data`;

CREATE OR REPLACE MODEL
  `SEU_PROJETO.churn_dataset.churn_model`
OPTIONS (
  MODEL_TYPE = 'RANDOM_FOREST_CLASSIFIER',
  INPUT_LABEL_COLS = ['Exited']
)
AS
SELECT
  CreditScore,
  Age,
  Tenure,
  Balance,
  NumOfProducts,
  Exited
FROM `SEU_PROJETO.churn_dataset.churn_features`;

SELECT *
FROM ML.EVALUATE(
  MODEL `SEU_PROJETO.churn_dataset.churn_model`
);

SELECT *
FROM ML.PREDICT(
  MODEL `SEU_PROJETO.churn_dataset.churn_model`,
  (
    SELECT
      CreditScore,
      Age,
      Tenure,
      Balance,
      NumOfProducts
    FROM `SEU_PROJETO.churn_dataset.churn_data`
  )
);