from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pasta = Path(__file__).resolve().parent
projeto = pasta.parent

mlflow.set_tracking_uri(
    "sqlite:///" + (projeto / "mlflow.db").as_posix()
)

mlflow.set_experiment("Hello-World-MLFlow")

np.random.seed(42)

for ruido in [5, 10, 20]:

    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 * X + 5
    y = y_true + ruido * np.random.randn(100, 1)

    with mlflow.start_run(run_name=f"ruido_{ruido}") as run:

        modelo = LinearRegression()
        modelo.fit(X, y)

        y_pred = modelo.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        mlflow.log_param("nivel_ruido", ruido)
        mlflow.log_param("n_amostras", 100)

        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric(
            "coeficiente", float(modelo.coef_[0][0])
        )
        mlflow.log_metric(
            "intercepto", float(modelo.intercept_[0])
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(X.ravel(), y.ravel(), alpha=0.5, label="Dados")
        plt.plot(X.ravel(), y_pred.ravel(), color="red", label="Previsao")
        plt.plot(X.ravel(), y_true.ravel(), color="green",
                 linestyle="--", label="Verdadeiro")

        plt.title(f"Regressao Linear - Ruido {ruido}")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        plt.text(
            0.05, 0.95,
            f"MSE: {mse:.4f}\nR2: {r2:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8)
        )

        grafico = pasta / f"regressao_ruido_{ruido}.png"
        plt.savefig(grafico, dpi=150)
        plt.close()

        mlflow.log_artifact(str(grafico))

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo"
        )

        print(f"\nRUIDO: {ruido}")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"RUN ID: {run.info.run_id}")
        print("-" * 40)

print("\n===== AULA 2 CONCLUIDA =====")
