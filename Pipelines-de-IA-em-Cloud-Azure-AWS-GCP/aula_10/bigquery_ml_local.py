from pathlib import Path
import json
import sqlite3

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent
DADOS = BASE.parent / "aula_05" / "Churn_Modelling_demo.csv"

FEATURES_CANDIDATAS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
]

ALVO = "Exited"


def main():
    print("=== AULA 10 - BIGQUERY ML LOCAL ===")

    df = pd.read_csv(DADOS)

    features = [
        coluna
        for coluna in FEATURES_CANDIDATAS
        if coluna in df.columns
    ]

    if not features:
        raise ValueError("Nenhuma feature encontrada.")

    if ALVO not in df.columns:
        raise ValueError("Coluna Exited nao encontrada.")

    if df[features + [ALVO]].isna().any().any():
        raise ValueError("Existem valores ausentes.")

    if df[ALVO].nunique() != 2:
        raise ValueError("O alvo precisa ter duas classes.")

    print("Clientes:", len(df))
    print("Features utilizadas:", features)

    banco = BASE / "churn_local.db"

    with sqlite3.connect(banco) as conexao:
        df.to_sql(
            "churn_data",
            conexao,
            if_exists="replace",
            index=False
        )

        colunas_sql = ", ".join(
            f'"{coluna}"'
            for coluna in features + [ALVO]
        )

        conexao.execute("DROP VIEW IF EXISTS churn_features")

        conexao.execute(
            f"CREATE VIEW churn_features AS "
            f"SELECT {colunas_sql} FROM churn_data"
        )

        dados_modelo = pd.read_sql_query(
            "SELECT * FROM churn_features",
            conexao
        )

    X = dados_modelo[features]
    y = dados_modelo[ALVO].astype(int)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3
    )

    modelo.fit(X_treino, y_treino)

    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    previsoes = (probabilidades >= 0.5).astype(int)

    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=[0, 1]
    )

    metricas = {
        "modelo": "RandomForestClassifier",
        "origem": "SQLite + Scikit-learn",
        "total_clientes": len(df),
        "clientes_treino": len(X_treino),
        "clientes_teste": len(X_teste),
        "features": features,
        "accuracy": float(accuracy_score(y_teste, previsoes)),
        "precision": float(
            precision_score(y_teste, previsoes, zero_division=0)
        ),
        "recall": float(
            recall_score(y_teste, previsoes, zero_division=0)
        ),
        "f1": float(
            f1_score(y_teste, previsoes, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_teste, probabilidades)
        ),
        "matriz_confusao": matriz.tolist(),
        "previsoes_classe_0": int((previsoes == 0).sum()),
        "previsoes_classe_1": int((previsoes == 1).sum())
    }

    resultado = X_teste.copy()
    resultado["Exited_real"] = y_teste.to_numpy()
    resultado["Exited_previsto"] = previsoes
    resultado["probabilidade_churn"] = probabilidades

    resultado.to_csv(
        BASE / "previsoes_churn.csv",
        index=False
    )

    joblib.dump(
        modelo,
        BASE / "modelo_random_forest.joblib"
    )

    (BASE / "metricas.json").write_text(
        json.dumps(metricas, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=== RESULTADOS ===")

    for nome in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
    ]:
        print(f"{nome}: {metricas[nome]:.4f}")

    print("Matriz de confusao:", matriz.tolist())
    print("Previsoes classe 0:", metricas["previsoes_classe_0"])
    print("Previsoes classe 1:", metricas["previsoes_classe_1"])
    print("Arquivos gerados com sucesso.")


if __name__ == "__main__":
    main()