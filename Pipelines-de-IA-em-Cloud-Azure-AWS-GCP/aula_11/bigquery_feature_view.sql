-- Exemplo de view para BigQuery.
-- Substituir SEU_PROJETO pelo projeto real.
-- Nao executado na Google Cloud.

CREATE OR REPLACE VIEW
  `SEU_PROJETO.churn_dataset.my_churn_features`
AS
SELECT
  CAST(RowNumber AS STRING) AS entity_id,
  CreditScore AS credit_score,
  Age AS age,
  Balance AS balance,
  Exited AS target_variable,
  CURRENT_TIMESTAMP() AS feature_timestamp
FROM `SEU_PROJETO.churn_dataset.churn_data`;