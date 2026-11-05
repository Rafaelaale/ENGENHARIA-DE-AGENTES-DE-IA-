from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn
import numpy as np

from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

print("===== ALURA - AULA 4 =====")
print("CONFIGURANDO MLFLOW COM SQLITE")

PASTA_AULA = Path(__file__).resolve().parent
PROJETO = PASTA_AULA.parent

BANCO = PROJETO / "mlflow.db"

EXPERIMENT_NAME = "demo_mlflow_educativo"
MODEL_NAME = "iris_classifier_educativo"

TRACKING_URI = "sqlite:///" + BANCO.as_posix()

mlflow.set_tracking_uri(TRACKING_URI)

print("\nTracking URI:", mlflow.get_tracking_uri())

client = MlflowClient()

experimento = client.get_experiment_by_name(EXPERIMENT_NAME)

if experimento is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print("Experimento criado.")
else:
    experiment_id = experimento.experiment_id
    print("Experimento ja existente. Reutilizando.")

print("Nome:", EXPERIMENT_NAME)
print("Experiment ID:", experiment_id)
print("Nome reservado para o modelo:", MODEL_NAME)

# Carregando o dataset Iris
iris = load_iris()

# Separacao treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42
)

feature_names = iris.feature_names
target_names = iris.target_names

print("\n===== DATASET IRIS =====")
print("Total de amostras:", len(iris.data))
print("Amostras de treinamento:", len(X_train))
print("Amostras de teste:", len(X_test))
print("Quantidade de variaveis:", len(feature_names))

print("\nVariaveis:")
for nome in feature_names:
    print("-", nome)

print("\nClasses:")
for nome in target_names:
    print("-", nome)

print("\n===== VALIDACAO =====")

assert len(X_train) == 120
assert len(X_test) == 30
assert len(feature_names) == 4
assert len(target_names) == 3

print("Divisao dos dados validada.")
print("Experimento configurado no SQLite.")
print("Nested runs e autolog serao utilizados na proxima aula.")

print("\n===== AULA 4 CONCLUIDA =====")
