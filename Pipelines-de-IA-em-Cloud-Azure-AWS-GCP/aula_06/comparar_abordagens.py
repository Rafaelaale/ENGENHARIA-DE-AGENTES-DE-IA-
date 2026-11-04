from pathlib import Path
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DADOS = BASE.parent / "aula_05" / "Churn_Modelling_demo.csv"
COLUNAS = ["CreditScore", "Balance"]


def carregar_dados():
    df = pd.read_csv(DADOS)
    print(f"Clientes carregados: {len(df)}")
    return df


def aprendizado_supervisionado(df):
    X = df[COLUNAS]
    y = df["Exited"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    return {
        "accuracy": float(accuracy_score(y_teste, previsoes)),
        "recall": float(recall_score(y_teste, previsoes, zero_division=0)),
        "f1": float(f1_score(y_teste, previsoes, zero_division=0))
    }


def aprendizado_nao_supervisionado(df):
    X = df[COLUNAS]
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
    grupos = modelo.fit_predict(X_normalizado)

    resultado = df.copy()
    resultado["grupo"] = grupos
    resultado.to_csv(BASE / "clientes_segmentados.csv", index=False)

    return {
        "quantidade_grupos": 3,
        "silhouette": float(silhouette_score(X_normalizado, grupos)),
        "clientes_por_grupo": {
            str(k): int(v)
            for k, v in resultado["grupo"].value_counts().sort_index().items()
        }
    }


def main():
    print("=== AULA 06 - ABORDAGENS DE MACHINE LEARNING ===")
    df = carregar_dados()

    print("\n=== APRENDIZADO SUPERVISIONADO ===")
    supervisionado = aprendizado_supervisionado(df)
    print(supervisionado)

    print("\n=== APRENDIZADO NAO SUPERVISIONADO ===")
    nao_supervisionado = aprendizado_nao_supervisionado(df)
    print(nao_supervisionado)

    relatorio = {
        "supervisionado": supervisionado,
        "nao_supervisionado": nao_supervisionado
    }

    destino = BASE / "resultado_comparacao.json"
    destino.write_text(
        json.dumps(relatorio, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nRelatorio salvo: {destino}")


if __name__ == "__main__":
    main()
