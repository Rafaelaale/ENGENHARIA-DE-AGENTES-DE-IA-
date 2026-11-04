import pandas as pd

VERSAO = "1.0.0"

CATALOGO = {
    "saldo_alto": {
        "versao": VERSAO,
        "origem": "Balance",
        "regra": "Balance >= 120000",
        "tipo": "int",
        "descricao": "Indica saldo igual ou superior a 120000"
    },
    "score_baixo": {
        "versao": VERSAO,
        "origem": "CreditScore",
        "regra": "CreditScore < 550",
        "tipo": "int",
        "descricao": "Indica score de credito inferior a 550"
    },
    "faixa_saldo": {
        "versao": VERSAO,
        "origem": "Balance",
        "regra": "0: baixo; 1: medio; 2: alto",
        "tipo": "int",
        "descricao": "Classificacao do saldo em tres faixas"
    }
}


def gerar_features(df: pd.DataFrame) -> pd.DataFrame:
    obrigatorias = {"CreditScore", "Balance"}

    ausentes = obrigatorias - set(df.columns)

    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")

    resultado = df.copy()

    if resultado[["CreditScore", "Balance"]].isna().any().any():
        raise ValueError("Dados ausentes nas colunas obrigatorias")

    resultado["saldo_alto"] = (
        resultado["Balance"] >= 120000
    ).astype(int)

    resultado["score_baixo"] = (
        resultado["CreditScore"] < 550
    ).astype(int)

    resultado["faixa_saldo"] = pd.cut(
        resultado["Balance"],
        bins=[-float("inf"), 50000, 120000, float("inf")],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)

    return resultado