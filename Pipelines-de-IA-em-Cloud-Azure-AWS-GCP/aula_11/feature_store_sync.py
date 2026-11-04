from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
DADOS = BASE.parent / "aula_05" / "Churn_Modelling_demo.csv"
BANCO = BASE / "feature_store_local.db"


def sincronizar(conexao):
    conexao.execute("DELETE FROM online_store")

    conexao.execute("""
        INSERT INTO online_store
        SELECT
            entity_id,
            credit_score,
            age,
            balance,
            target_variable,
            feature_timestamp
        FROM my_churn_features
    """)

    conexao.commit()

    quantidade = conexao.execute(
        "SELECT COUNT(*) FROM online_store"
    ).fetchone()[0]

    return quantidade


def consultar(conexao, entity_id):
    conexao.row_factory = sqlite3.Row

    registro = conexao.execute("""
        SELECT *
        FROM online_store
        WHERE entity_id = ?
    """, (str(entity_id),)).fetchone()

    return dict(registro) if registro else None


def main():
    print("=== AULA 11 - FEATURE STORE ===")

    df = pd.read_csv(DADOS)

    obrigatorias = {
        "CreditScore",
        "Balance",
        "Exited"
    }

    ausentes = obrigatorias - set(df.columns)

    if ausentes:
        raise ValueError(
            f"Colunas ausentes: {sorted(ausentes)}"
        )

    if "Age" not in df.columns:
        df["Age"] = 40
        print(
            "AVISO: Age ausente no CSV. "
            "Valor demonstrativo 40 aplicado."
        )

    df = df.copy()
    df["entity_id"] = [
        str(i) for i in range(1, len(df) + 1)
    ]

    df["feature_timestamp"] = (
        datetime.now(timezone.utc).isoformat()
    )

    with sqlite3.connect(BANCO) as conexao:
        df.to_sql(
            "churn_data",
            conexao,
            if_exists="replace",
            index=False
        )

        conexao.execute(
            "DROP VIEW IF EXISTS my_churn_features"
        )

        conexao.execute("""
            CREATE VIEW my_churn_features AS
            SELECT
                entity_id,
                CreditScore AS credit_score,
                Age AS age,
                Balance AS balance,
                Exited AS target_variable,
                feature_timestamp
            FROM churn_data
        """)

        conexao.execute("""
            CREATE TABLE online_store (
                entity_id TEXT PRIMARY KEY,
                credit_score REAL,
                age REAL,
                balance REAL,
                target_variable INTEGER,
                feature_timestamp TEXT
            )
        """)

        quantidade = sincronizar(conexao)

        print("Features sincronizadas:", quantidade)

        cliente = "1"

        antes = consultar(conexao, cliente)

        print("Cliente consultado:", cliente)
        print("Saldo inicial:", antes["balance"])

        novo_saldo = float(antes["balance"]) + 10000

        conexao.execute("""
            UPDATE churn_data
            SET Balance = ?
            WHERE entity_id = ?
        """, (novo_saldo, cliente))

        conexao.commit()

        sem_sincronizar = consultar(conexao, cliente)

        print(
            "Saldo na Online Store antes da nova sincronizacao:",
            sem_sincronizar["balance"]
        )

        assert (
            sem_sincronizar["balance"] == antes["balance"]
        )

        sincronizar(conexao)

        depois = consultar(conexao, cliente)

        print(
            "Saldo na Online Store apos sincronizacao:",
            depois["balance"]
        )

        assert abs(
            depois["balance"] - novo_saldo
        ) < 0.000001

        resultado = {
            "quantidade_entidades": quantidade,
            "entity_id": cliente,
            "saldo_inicial": antes["balance"],
            "saldo_origem_atualizado": novo_saldo,
            "saldo_online_antes_sync": (
                sem_sincronizar["balance"]
            ),
            "saldo_online_depois_sync": depois["balance"],
            "sincronizacao_validada": True,
            "features_cliente": depois
        }

        (BASE / "resultado_feature_store.json").write_text(
            json.dumps(
                resultado,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print("Sincronizacao validada:", True)
        print("Arquivo de resultados gerado.")


if __name__ == "__main__":
    main()