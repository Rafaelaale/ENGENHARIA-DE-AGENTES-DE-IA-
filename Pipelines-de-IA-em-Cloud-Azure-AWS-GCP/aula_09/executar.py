from pathlib import Path
import json
import pandas as pd

from feature_store import CATALOGO, VERSAO, gerar_features

BASE = Path(__file__).resolve().parent
DADOS = BASE.parent / "aula_05" / "Churn_Modelling_demo.csv"


def main():
    print("=== AULA 09 - FEATURE STORE LOCAL ===")

    df = pd.read_csv(DADOS)

    print("Clientes carregados:", len(df))

    features_treino = gerar_features(df)
    features_producao = gerar_features(df.copy())

    colunas = list(CATALOGO.keys())

    consistentes = features_treino[colunas].equals(
        features_producao[colunas]
    )

    assert consistentes, "Inconsistencia entre as features"

    assert set(features_treino["saldo_alto"].unique()) <= {0, 1}
    assert set(features_treino["score_baixo"].unique()) <= {0, 1}
    assert set(features_treino["faixa_saldo"].unique()) <= {0, 1, 2}

    features_treino.to_csv(
        BASE / "features_clientes.csv",
        index=False
    )

    (BASE / "catalogo_features.json").write_text(
        json.dumps(CATALOGO, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    validacao = {
        "versao": VERSAO,
        "quantidade_clientes": len(df),
        "features": colunas,
        "consistencia_treino_producao": consistentes,
        "saldo_alto": int(features_treino["saldo_alto"].sum()),
        "score_baixo": int(features_treino["score_baixo"].sum()),
        "faixa_saldo_distribuicao": {
            str(k): int(v)
            for k, v in features_treino["faixa_saldo"]
            .value_counts()
            .sort_index()
            .items()
        }
    }

    (BASE / "validacao_features.json").write_text(
        json.dumps(validacao, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Versao:", VERSAO)
    print("Features:", colunas)
    print("Consistencia treino/producao:", consistentes)
    print("Clientes com saldo alto:", validacao["saldo_alto"])
    print("Clientes com score baixo:", validacao["score_baixo"])
    print("Distribuicao das faixas:", validacao["faixa_saldo_distribuicao"])
    print("Arquivos gerados com sucesso.")


if __name__ == "__main__":
    main()