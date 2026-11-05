from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np

from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

PROJETO = Path(__file__).resolve().parent.parent
BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///" + BANCO.as_posix())
mlflow.set_registry_uri(mlflow.get_tracking_uri())

EXPERIMENT_NAME = "demo_mlflow_educativo"
MODEL_NAME = "iris_classifier_educativo"

mlflow.set_experiment(EXPERIMENT_NAME)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42
)

print("\n===== AULA 5 - NESTED RUNS =====")

best_accuracy = -1
best_run_id = None
best_model_type = None

rf_params = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 7}
]

lr_params = [
    {"C": 0.1, "max_iter": 1000},
    {"C": 1.0, "max_iter": 1000},
    {"C": 10.0, "max_iter": 1000}
]

with mlflow.start_run(run_name="parent_run_model_comparison") as parent:

    mlflow.log_param("experiment_type", "model_comparison")
    mlflow.log_param("models_tested", 2)
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("num_classes", len(iris.target_names))
    mlflow.set_tag("feature_names", ", ".join(iris.feature_names))

    print("Parent Run ID:", parent.info.run_id)

    configuracoes = [
        (RandomForestClassifier(**p, random_state=42), p, "rf")
        for p in rf_params
    ] + [
        (LogisticRegression(**p, random_state=42), p, "lr")
        for p in lr_params
    ]

    for model, params, tipo in configuracoes:

        with mlflow.start_run(
            run_name=f"{tipo}_{len(configuracoes)}_{params}",
            nested=True
        ) as child:

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("n_features", X_train.shape[1])
            mlflow.set_tag("algorithm", type(model).__name__)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="modelo", skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            print(
                f"{type(model).__name__}: "
                f"{params} | Accuracy: {accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = child.info.run_id
                best_model_type = type(model).__name__

    mlflow.log_metric("best_accuracy", float(best_accuracy))
    mlflow.set_tag("best_child_run_id", best_run_id)
    mlflow.set_tag("best_model_type", best_model_type)

print("\nMelhor accuracy:", round(best_accuracy, 4))
print("Melhor Run ID:", best_run_id)
print("Melhor algoritmo:", best_model_type)

print("\n===== AUTOLOG E GRIDSEARCH =====")

mlflow.sklearn.autolog(
    log_models=True,
    max_tuning_runs=5
)

try:
    with mlflow.start_run(run_name="autolog_gridsearch") as run:

        grid_run_id = run.info.run_id

        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        predictions = grid_search.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )

        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_precision", float(precision))
        mlflow.log_metric("test_recall", float(recall))
        mlflow.log_metric("test_f1", float(f1))

        print("Melhores parametros:", grid_search.best_params_)
        print("Score CV:", round(grid_search.best_score_, 4))
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("Grid Run ID:", grid_run_id)

finally:
    mlflow.sklearn.autolog(disable=True)

print("\n===== MODEL REGISTRY =====")

# Registro explicito do melhor estimador do GridSearch.
# Isso evita depender da localizacao interna do autolog.

with mlflow.start_run(run_name="registro_modelo_iris") as run:

    model_info = mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        name="modelo_iris", skops_trusted_types=["sklearn.tree._tree.Tree"],
        input_example=X_train[:3]
    )

    print("Model URI:", model_info.model_uri)

    versao = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

client = MlflowClient()
version = versao.version

client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description="Random Forest Iris otimizado com GridSearchCV"
)

for chave, valor in {
    "algorithm": "RandomForest",
    "dataset": "iris",
    "framework": "scikit-learn",
    "optimization": "GridSearchCV",
    "environment": "educativo_local"
}.items():

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key=chave,
        value=valor
    )

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="candidate",
    version=version
)

print("Versao registrada:", version)
print("Alias: candidate")

candidate = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@candidate"
)

candidate_accuracy = accuracy_score(
    y_test, candidate.predict(X_test)
)

print("Accuracy candidate:", round(candidate_accuracy, 4))

if candidate_accuracy >= 0.90:

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version
    )

    print("Alias champion configurado (simulacao educativa).")

    champion = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    amostras = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 4.2, 1.5]
    ])

    predicoes = champion.predict(amostras)
    probabilidades = champion.predict_proba(amostras)

    print("\n===== PREDICOES =====")

    for i, (classe, probs) in enumerate(
        zip(predicoes, probabilidades), start=1
    ):
        print(f"Amostra {i}: {iris.target_names[classe]}")
        print("Probabilidades:", np.round(probs, 3))

else:
    print("Modelo nao atingiu o criterio educativo de 90%.")

print("\n===== AULA 5 CONCLUIDA =====")

