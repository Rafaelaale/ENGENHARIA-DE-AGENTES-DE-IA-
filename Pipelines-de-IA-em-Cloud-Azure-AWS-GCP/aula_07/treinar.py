from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DADOS = BASE.parent / "aula_05" / "Churn_Modelling_demo.csv"
FEATURES = ["CreditScore", "Balance"]

def main():
    df = pd.read_csv(DADOS)
    X = df[FEATURES]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": float(accuracy_score(y_test, previsoes)),
        "recall": float(recall_score(y_test, previsoes, zero_division=0)),
        "f1": float(f1_score(y_test, previsoes, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probabilidades)),
        "matriz_confusao": confusion_matrix(y_test, previsoes).tolist()
    }

    joblib.dump(modelo, BASE / "modelo_churn.joblib")
    (BASE / "metricas.json").write_text(
        json.dumps(metricas, indent=4), encoding="utf-8"
    )

    print("=== MODELO TREINADO ===")
    for chave, valor in metricas.items():
        print(f"{chave}: {valor}")
    print("Modelo salvo com sucesso.")

if __name__ == "__main__":
    main()
