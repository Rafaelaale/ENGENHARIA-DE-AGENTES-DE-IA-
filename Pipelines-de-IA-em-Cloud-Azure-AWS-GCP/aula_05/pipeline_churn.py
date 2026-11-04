from pathlib import Path
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2,<3"]
)
def load_data(path: str, df_art: Output[Dataset]):
    import pandas as pd

    df = pd.read_csv(path)

    required = ["CreditScore", "Balance", "Exited"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    df.to_json(df_art.path, orient="split")
    print(f"Dados carregados: {len(df)} registros")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2,<3",
        "scikit-learn>=1.5,<2",
        "joblib>=1.4,<2"
    ]
)
def train_component(
    df_art: Input[Dataset],
    numeric_cols: list,
    target_col: str,
    metrics: Output[Metrics],
    model_art: Output[Model],
):
    import joblib
    import pandas as pd

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_json(df_art.path, orient="split")

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=2,
        stratify=y
    )

    numeric_pipeline = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols)
    ])

    model = SkPipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("recall", float(recall))
    metrics.log_metric("f1", float(f1))

    joblib.dump(model, model_art.path)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print("Modelo salvo como artefato")


@dsl.pipeline(name="training-churn-pipeline")
def training_pipeline(
    path: str,
    numeric_cols: list,
    target_col: str = "Exited"
):
    data_task = load_data(path=path)

    train_component(
        df_art=data_task.outputs["df_art"],
        numeric_cols=numeric_cols,
        target_col=target_col
    )


def main():
    destino = Path(__file__).resolve().parent
    arquivo = destino / "pipeline_real_data.json"

    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=str(arquivo)
    )

    print("=== AULA 05 ===")
    print("Pipeline compilado.")
    print(f"JSON: {arquivo}")
    print("Componentes: load-data e train-component")
    print("Execucao na Google Cloud: nao realizada")


if __name__ == "__main__":
    main()